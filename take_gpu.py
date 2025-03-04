import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Adjust based on your model
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
model.eval()

# Initial batch size
batch_size = 1
max_tokens = 32  # Number of tokens per input
prompt = "What is the meaning of life?"  # Example prompt

# Gradually increase batch size
while True:
    try:
        # Prepare input batch
        inputs = [prompt] * batch_size
        tokenized_inputs = tokenizer(inputs, return_tensors="pt")
        
        # Move tensors to the correct device
        input_ids = tokenized_inputs["input_ids"].to(device)
        attention_mask = tokenized_inputs["attention_mask"].to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_tokens)

        batch_size *= 2  # Double the batch size

    except torch.cuda.OutOfMemoryError:
        batch_size //= 2
