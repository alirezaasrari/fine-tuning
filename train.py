
# ==========================================
# FINE-TUNING QWEN 0.5B USING LoRA
# ==========================================

# STEP 1: Import libraries

import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    get_peft_model
)


# STEP 2: Define model and dataset paths

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DATASET_PATH = "dataset/train.jsonl"

OUTPUT_DIR = "my_finetuned_model"


# STEP 3: Check GPU

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available!")

device = "cuda"

print("GPU:", torch.cuda.get_device_name(0))

# Choose a precision supported by the GPU
use_bf16 = torch.cuda.is_bf16_supported()

model_dtype = torch.bfloat16 if use_bf16 else torch.float16


# STEP 4: Load tokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# STEP 5: Load pretrained model

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=model_dtype
)

model.to(device)

model.config.use_cache = False


# STEP 6: Load our dataset

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train"
)

print("Number of examples:", len(dataset))


# STEP 7: Convert conversations into model inputs

def tokenize_example(example):

    # Apply Qwen's conversation template
    formatted_text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

    # Convert text into token IDs
    result = tokenizer(
        formatted_text,
        truncation=True,
        max_length=512,
        add_special_tokens=False
    )

    return result


tokenized_dataset = dataset.map(
    tokenize_example,
    remove_columns=dataset.column_names
)


# STEP 8: Prepare the data collator

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# STEP 9: Configure LoRA

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)


# STEP 10: Attach LoRA adapters to our model

model = get_peft_model(
    model,
    lora_config
)

# Show trainable parameters
model.print_trainable_parameters()


# STEP 11: Configure training

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    weight_decay=0.01,

    warmup_steps=0.03,

    logging_steps=1,

    save_strategy="epoch",

    save_total_limit=2,

    fp16=not use_bf16,

    bf16=use_bf16,

    report_to="none"
)


# STEP 12: Create the Trainer

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_dataset,

    data_collator=data_collator
)


# STEP 13: Start training

print("Starting fine-tuning...")

trainer.train()


# STEP 14: Save trained model

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuning completed!")

print("Model saved to:", OUTPUT_DIR)
