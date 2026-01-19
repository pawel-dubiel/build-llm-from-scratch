from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

def train_tokenizer():
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # And then train
    trainer = trainers.BpeTrainer(
        vocab_size=50257,  # GPT-2 size, maybe overkill for small data but standard
        min_frequency=2,
        special_tokens=["<|endoftext|>"]
    )
    
    # Train on the all_books.txt
    files = ["data/all_books.txt"]
    tokenizer.train(files, trainer)
    
    # Save the tokenizer
    tokenizer.save("tokenizer.json")
    print("Tokenizer trained and saved to tokenizer.json")

if __name__ == "__main__":
    train_tokenizer()
