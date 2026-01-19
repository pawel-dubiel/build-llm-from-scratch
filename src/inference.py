import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import sys

def generate_text(prompt, model_path="checkpoints/final_model", max_length=100):
    print(f"Loading model from {model_path}...")
    try:
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Did you train the model yet?")
        return

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"Generating text for prompt: '{prompt}'")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print("-" * 40)
    print(generated_text)
    print("-" * 40)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "The quick brown fox"
        
    generate_text(prompt)
