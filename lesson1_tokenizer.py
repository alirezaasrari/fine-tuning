
# Step 1: Import the tokenizer tools
from transformers import AutoTokenizer

# Step 2: Select our pretrained model
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Step 3: Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 4: Write a sample sentence
text = "What is an electrical transformer?"

# Step 5: Convert the sentence into tokens
tokens = tokenizer.tokenize(text)

# Step 6: Convert tokens into numerical IDs
token_ids = tokenizer.encode(text)

# Step 7: Print the results
print("Original text:", text)
print("Tokens:", tokens)
print("Token IDs:", token_ids)

# Step 8: Convert IDs back into readable text
decoded_text = tokenizer.decode(token_ids)

print("Decoded text:", decoded_text)
