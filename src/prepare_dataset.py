from datasets import Dataset
from transformers import PreTrainedTokenizerFast
import os

def prepare_dataset():
    # Load tokenizer
    # We need to wrap the raw tokenizer.json in a transformers class
    # to make it compatible with the rest of the ecosystem if needed, 
    # but for simple encoding here, we need it to handle the special tokens.
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    # GPT-2 uses the same token for eos and bos usually, or none.
    # We'll stick to <|endoftext|> as eos/pad.

    data_path = "data/all_books.txt"
    if not os.path.exists(data_path):
        print("Data file not found!")
        return

    print("Reading text...")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Tokenizing...")
    # Tokenize the entire text
    encodings = tokenizer(text, return_attention_mask=False)
    ids = encodings["input_ids"]
    
    print(f"Total tokens: {len(ids)}")

    block_size = 128
    print(f"Chunking into blocks of {block_size}...")
    
    # Create chunks
    chunks = []
    for i in range(0, len(ids), block_size):
        chunk = ids[i:i + block_size]
        if len(chunk) == block_size:
            chunks.append(chunk)
            
    # Convert to HF Dataset
    # We use the chunk as both input_ids and labels (standard for causal LM)
    # The Trainer will handle shifting labels automatically if we set labels=input_ids
    data = {
        "input_ids": chunks,
        "labels": chunks
    }
    
    dataset = Dataset.from_dict(data)
    
    save_path = "data/processed_dataset"
    dataset.save_to_disk(save_path)
    print(f"Dataset saved to {save_path}. Num samples: {len(dataset)}")

if __name__ == "__main__":
    prepare_dataset()
