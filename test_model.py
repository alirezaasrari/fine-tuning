
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel


# 1. Original pretrained model
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# 2. Our trained LoRA adapter
ADAPTER_PATH = "my_finetuned_model"


# 3. Select device and precision
device = "cuda"

use_bf16 = torch.cuda.is_bf16_supported()

dtype = torch.bfloat16 if use_bf16 else torch.float16


# 4. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH
)


# 5. Load original model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype
).to(device)


# 6. Load our trained adapter
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()


# 7. Prepare a question
messages = [
    {
        "role": "system",
        "content": "You are an engineering assistant."
    },
    {
        "role": "user",
        "content": "What is dissolved gas analysis?"
    }
]


# 8. Convert messages into model input
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(device)


# 9. Generate answer
with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False
    )


# 10. Extract generated answer
new_tokens = outputs[
    0, inputs["input_ids"].shape[-1]:
]

answer = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True
)


print("\nQuestion:")
print(messages[-1]["content"])

print("\nFine-tuned model answer:")
print(answer)
