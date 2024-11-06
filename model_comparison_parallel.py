import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize.treebank import TreebankWordTokenizer
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import nltk
import logging
from torch.utils.data.distributed import DistributedSampler


def initial_setup():
# Download NLTK data

    nltk.download('punkt', quiet=True)
    logging.basicConfig(level=logging.DEBUG)
    try:
        nltk.data.find('tokenizers/punkt')
        print("'punkt' tokenizer is available.")
    except LookupError:
        print("'punkt' tokenizer is not available.")

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    return device


class CodeDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# Initialize distributed environment
def init_distributed_mode(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Choose backend based on availability
    if torch.cuda.is_available():
        backend = 'nccl'
    else:
        backend = 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def gather_all_from_ranks(data):
    gathered_data = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_data, data)
    # Flatten the list if data is a list
    if isinstance(data, list):
        return [item for sublist in gathered_data for item in sublist]
    else:
        return gathered_data

# Step 1: Load the datasets
def load_and_sample_datasets(fraction=0.1, dataset_name='codeparrot/xlcost-text-to-code'):
    # Load datasets
    print(f"Loading dataset: {dataset_name}")
    datasets = []
    if dataset_name == 'codeparrot/xlcost-text-to-code':
        try:
            code_x_glue_dataset = load_dataset("codeparrot/xlcost-text-to-code", split='train')
            #code_x_glue_dataset = load_dataset('code_x_glue_ct_code_to_text', 'python', split='train')
            datasets.append(code_x_glue_dataset)
        except Exception as e:
            print(f"Failed to load xlcost-text-to-code dataset: {e}")
    elif dataset_name == 'codeparrot/apps':
        try:
            apps_dataset = load_dataset('codeparrot/apps', split='all', trust_remote_code=True)
            datasets.append(apps_dataset)
        except Exception as e:
            print(f"Failed to load codeparrot/apps dataset: {e}")
    elif dataset_name == 'codeparrot/codeparrot-clean': 
        try:
            codeparrot_clean_dataset = load_dataset('codeparrot/codeparrot-clean', split='train')
            datasets.append(codeparrot_clean_dataset)
        except Exception as e:
            print(f"Failed to load codeparrot/codeparrot-clean dataset: {e}")

    if not datasets:
        raise ValueError("No datasets were loaded successfully.")

    # Combine datasets
    combined_dataset = concatenate_datasets(datasets)

    # Sample 10% of the data
    num_samples = int(len(combined_dataset) * fraction)
    sampled_dataset = combined_dataset.shuffle(seed=42).select(range(num_samples))
    return sampled_dataset


def load_model_and_tokenizer_distributed(rank, world_size, model_name, sampled_dataset):
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if torch.cuda.is_available():
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)
    
    dataset = CodeDataset(sampled_dataset)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)
    
    # Tracking metrics
    total_examples = 0
    exact_matches = 0
    bleu_scores = []
    response_times = []
    no_text_or_code = 0
    smoothie = SmoothingFunction().method4
    nltk_treebank_tokenizer = TreebankWordTokenizer()

    # Process data
    for batch in tqdm(dataloader, desc=f"Processing on GPU {rank}"):
        example = batch[0]
        text_input = example['nl'] or example['question'] or example['text']
        reference_code = example['code'] or example['solutions'] or example['answer']
        
        if not text_input or not reference_code:
            no_text_or_code += 1
            continue
        
        prompt = f"Write code for the following description:\n{text_input[0]}"
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        # Generate code
        start_time = time.time()
        output_sequences = model.module.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask', None),
            max_length=inputs['input_ids'].shape[1] + 256,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        end_time = time.time()
        response_time = end_time - start_time
        response_times.append(response_time)
        
        generated_code = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        generated_code = generated_code[len(prompt):].strip()
        
        total_examples += 1
        is_exact_match = generated_code.strip() == reference_code[0].strip()
        if is_exact_match:
            exact_matches += 1
        
        reference_tokens = nltk_treebank_tokenizer.tokenize(reference_code[0])
        candidate_tokens = nltk_treebank_tokenizer.tokenize(generated_code)
        bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)
        bleu_scores.append(bleu_score)

    # Gather results from all ranks
    all_bleu_scores = gather_all_from_ranks(bleu_scores)
    all_exact_matches = gather_all_from_ranks(exact_matches)
    all_total_examples = gather_all_from_ranks(total_examples)
    all_response_times = gather_all_from_ranks(response_times)

    # Only rank 0 will calculate and print the final aggregated results
    if rank == 0:
        bleu_scores_combined = [score for scores in all_bleu_scores for score in scores]
        total_examples_combined = sum(all_total_examples)
        exact_matches_combined = sum(all_exact_matches)
        response_times_combined = [time for times in all_response_times for time in times]

        print(f"Processed {total_examples_combined} examples.")
        print(f"no_text_or_code: {no_text_or_code}")

        if bleu_scores_combined:
            average_bleu = sum(bleu_scores_combined) / len(bleu_scores_combined)
            print(f"\nAverage BLEU score: {average_bleu:.4f}")
            
            percentiles = [25, 50, 75, 90, 95, 99]
            percentile_values = np.percentile(bleu_scores_combined, percentiles)
            print("\nBLEU Score Percentiles:")
            for p, value in zip(percentiles, percentile_values):
                print(f"{p}th percentile: {value:.4f}")
        else:
            print("\nNo BLEU scores were calculated.")

        if response_times_combined:
            average_time = sum(response_times_combined) / len(response_times_combined)
            total_time = sum(response_times_combined)
            max_time = max(response_times_combined)
            min_time = min(response_times_combined)
            throughput = total_examples_combined / total_time if total_time > 0 else 0
            print(f"Average API response time: {average_time:.2f} seconds")
            print(f"Total API response time: {total_time:.2f} seconds")
            print(f"Max API response time: {max_time:.2f} seconds")
            print(f"Min API response time: {min_time:.2f} seconds")
            print(f"Throughput: {throughput:.2f} requests per second")
        else:
            print("\nNo response times were recorded.")

        if total_examples_combined > 0:
            exact_match_rate = exact_matches_combined / total_examples_combined * 100
            print(f"Exact match rate: {exact_match_rate:.2f}% ({exact_matches_combined}/{total_examples_combined})")
        else:
            print("\nNo examples were processed.")

    # Finalize the distributed process group
    dist.destroy_process_group()


def main(rank, world_size, model_name, dataset_name, fraction):
    init_distributed_mode(rank, world_size)
    print("point 2")
    print(f"Running on {world_size} devices, rank {rank}")
    print("point 3")
    initial_setup()
    print("point 4")
    device = get_device()
    print("point 5")
    sampled_dataset = load_and_sample_datasets(fraction, dataset_name=dataset_name)
    print("point 6")
    load_model_and_tokenizer_distributed(rank, world_size, model_name, sampled_dataset)


if __name__ == "__main__":
    model_name = 'Salesforce/codegen-350M-mono'
    # World size (number of processes to use)
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No GPUs detected. Using CPU with world_size = 1.")
        world_size = 1
    fraction = 0.1
    dataset_name = 'codeparrot/apps'
    # Start the multiprocessing
    print(f"Running on {world_size} devices.")
    print("point 1")
    mp.spawn(
        main,
        args=(world_size, model_name, dataset_name, fraction),
        nprocs=world_size,
        join=True
    )

