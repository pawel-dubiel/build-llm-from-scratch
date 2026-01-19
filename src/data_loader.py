import os
from datasets import load_dataset
import tqdm

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "all_books.txt") # Keeping name for compatibility

def download_tinystories():
    print("Loading TinyStories dataset from Hugging Face...")
    # Load streaming to avoid massive RAM usage or downloading everything if we just want a subset
    # However, for TinyStories, it's not THAT huge (a few GBs). 
    # Let's load a subset 'train' and take a reasonable amount of data for a "small" training run.
    # We'll take 200,000 stories. TinyStories are short.
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    print(f"Saving to {OUTPUT_FILE}...")
    
    count = 0
    max_stories = 200_000 # enough for a good demo, not too heavy
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in tqdm.tqdm(dataset):
            text = sample['text']
            f.write(text)
            f.write("\n\n<|endoftext|>\n\n")
            count += 1
            if count >= max_stories:
                break
                
    print(f"Saved {count} stories to {OUTPUT_FILE}. Data preparation complete.")

if __name__ == "__main__":
    download_tinystories()
