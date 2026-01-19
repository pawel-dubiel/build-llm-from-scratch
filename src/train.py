import torch
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, PreTrainedTokenizerFast
from datasets import load_from_disk
import os

def train():
    print("Initializing model configuration...")
    # Configuration
    # Small model for demonstration (approx 124M params is standard GPT-2 Small)
    # n_layer=12, n_head=12, n_embd=768
    config = GPT2Config(
        vocab_size=50257, # matches tokenizer
        n_positions=128,  # matches dataset chunking
        n_ctx=128,
        n_embd=768,
        n_layer=12,
        n_head=12,
    )
    
    model = GPT2LMHeadModel(config)
    
    # Check availability (Trainer handles this internally usually, but good to verify)
    if torch.backends.mps.is_available():
        print("MPS (Apple Silicon) is available.")
    else:
        print("MPS not available. Using CPU (will be slow).")
        
    # Load tokenizer (needed for data collator padding)
    print("Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    
    # Ensure pad token is set in model config
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load dataset
    print("Loading dataset...")
    if not os.path.exists("data/processed_dataset"):
        raise FileNotFoundError("Processed dataset not found. Run prepare_dataset.py first.")
        
    dataset = load_from_disk("data/processed_dataset")
    
    # Split dataset
    print("Splitting dataset...")
    split_dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    # Training Arguments
    # Note: MPS usage is automatic in newer Transformers if available.
    training_args = TrainingArguments(
        output_dir="checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8, # Try 8 for M3, reduce if OOM
        per_device_eval_batch_size=8,
        max_steps=30, # For demo purposes, run short training
        save_steps=50,
        save_total_limit=2,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=50,
        learning_rate=5e-4,
        weight_decay=0.01,
        use_cpu=True,
        push_to_hub=False,
        report_to="none", # Disable wandb/etc for this simple setup
        dataloader_pin_memory=False,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model("checkpoints/final_model")
    tokenizer.save_pretrained("checkpoints/final_model")
    print("Done!")

if __name__ == "__main__":
    train()
