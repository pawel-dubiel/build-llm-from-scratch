import os
import requests
import time

# List of books to download (Project Gutenberg IDs)
# 84: Frankenstein
# 1342: Pride and Prejudice
# 1661: The Adventures of Sherlock Holmes
# 11: Alice's Adventures in Wonderland
# 2701: Moby Dick
BOOKS = {
    "frankenstein": "https://www.gutenberg.org/files/84/84-0.txt",
    "pride_and_prejudice": "https://www.gutenberg.org/files/1342/1342-0.txt",
    "sherlock_holmes": "https://www.gutenberg.org/files/1661/1661-0.txt",
    "alice_in_wonderland": "https://www.gutenberg.org/files/11/11-0.txt",
    "moby_dick": "https://www.gutenberg.org/files/2701/2701-0.txt",
}

DATA_DIR = "data"

def download_book(name, url):
    """Downloads a book from the given URL and saves it to the data directory."""
    print(f"Downloading {name}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Gutenberg texts often have UTF-8 with BOM or other encodings, 
        # but modern files usually standardize on utf-8.
        # We'll try to decode as utf-8-sig to handle BOM if present.
        text = response.content.decode('utf-8-sig')
        
        # Basic cleanup: Remove header and footer (heuristic)
        # This is a rough heuristic for Gutenberg books
        start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
        
        start_idx = text.find(start_marker)
        if start_idx != -1:
            # Find the end of the line
            start_idx = text.find("\n", start_idx) + 1
        else:
            start_idx = 0
            
        end_idx = text.find(end_marker)
        if end_idx == -1:
            end_idx = len(text)
            
        clean_text = text[start_idx:end_idx].strip()
        
        output_path = os.path.join(DATA_DIR, f"{name}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(clean_text)
            
        print(f"Saved {name} to {output_path} ({len(clean_text)} characters)")
        return output_path
        
    except Exception as e:
        print(f"Failed to download {name}: {e}")
        return None

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    for name, url in BOOKS.items():
        download_book(name, url)
        time.sleep(1) # Be nice to the server

    # Combine all into one file for easier training
    print("Combining all books into all_books.txt...")
    with open(os.path.join(DATA_DIR, "all_books.txt"), "w", encoding="utf-8") as outfile:
        for filename in os.listdir(DATA_DIR):
            if filename.endswith(".txt") and filename != "all_books.txt":
                path = os.path.join(DATA_DIR, filename)
                with open(path, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
                    outfile.write("\n\n<|endoftext|>\n\n") # Separator
    
    print("Data preparation complete.")

if __name__ == "__main__":
    main()
