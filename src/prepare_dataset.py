from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
import os

def prepare_dataset():
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

    data_path = "data/all_books.txt"
    if not os.path.exists(data_path):
        print("Data file not found!")
        return

    print("Loading dataset from text file...")
    # Using load_dataset with 'text' builder handles large files efficiently
    # and provides progress bars automatically during mapping.
    dataset = load_dataset("text", data_files={"train": data_path})
    
    # Tokenization function that runs in parallel
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    print("Tokenizing (with progress bar)...")
    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=4, # Use 4 cores for speed
        remove_columns=["text"],
        desc="Tokenizing"
    )

    block_size = 128
    
    # Function to group tokens into chunks
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # Drop the small remainder
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
            
        # Split by chunks of max_len
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    print("Grouping into chunks...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
        desc="Grouping"
    )

    print(f"Saving dataset... Num samples: {len(lm_datasets['train'])}")
    lm_datasets['train'].save_to_disk("data/processed_dataset")
    print("Done.")

if __name__ == "__main__":
    prepare_dataset()
