import transformers
from transformers import Trainer, TrainingArguments, GPT2Tokenizer, GPT2LMHeadModel,DataCollatorForLanguageModeling
from data import load_data, train_eval_dataset
import torch
from peft import LoraConfig, TaskType, get_peft_model

PW_WORD = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ "

dataset = load_data("rockyou")
checkpoint = "GPT2-Hacker-password-generator"
model = GPT2LMHeadModel.from_pretrained(f"./model/{checkpoint}")
tokenizer = GPT2Tokenizer.from_pretrained(f"./model/{checkpoint}")

# 將模型移到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# 設定 pad token (GPT2 預設沒有)
tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT2 是 causal LM，不是 masked LM
)

# 定義 tokenize 函數（不做 padding，交給 DataCollator 動態處理）
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
    )
    # labels 和 padding 由 DataCollatorForLanguageModeling 自動處理

# 對資料集進行 tokenize，透過map使用設定的tokenize_function套用到到每筆資料上
dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

#   {"text": "password123"}
#          ↓ tokenized by tokenize_function
#   {
#       "input_ids": [7, 34, 22, ..., 50256, 50256],      # token IDs (128個)
#       "attention_mask": [1, 1, 1, ..., 0, 0],           # 1=真實token, 0=padding
#       "labels": [7, 34, 22, ..., 50256, 50256]          # 同 input_ids
#   }

#keys_value_list = build_voc(model=checkpoint, WORD=PW_WORD)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    target_modules=['c_attn', 'c_proj'], 
    lora_alpha=32,
    lora_dropout=0.2,
    bias='none',
)

model = get_peft_model(model, lora_config)

train_ds, val_ds = train_eval_dataset(dataset["train"])

training_args = TrainingArguments(
    output_dir="./checkpoint",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=3,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="steps",
    eval_steps=1000,
    logging_dir="./logs",
    logging_steps=500,
    gradient_accumulation_steps=16,
    learning_rate=5e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    seed=42,
    report_to="tensorboard",
    bf16=True,  # 或 fp16=True，只能選一個
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args, 
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("./trained_model/rockyou-lora-password-generator")
tokenizer.save_pretrained("./trained_model/rockyou-lora-password-generator")