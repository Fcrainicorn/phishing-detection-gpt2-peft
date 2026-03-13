import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType, AutoPeftModelForSequenceClassification

# Project Setup
MODEL_SAVE_DIR = "./final_model_weights"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Dataset & Tokenize
dataset = load_dataset("shawhin/phishing-site-classification")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized = dataset.map(tokenize_function, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 2. Baseline Model
model = AutoModelForSequenceClassification.from_pretrained(
    "gpt2", num_labels=2,
    id2label={0: "SAFE", 1: "NOT_SAFE"},
    label2id={"SAFE": 0, "NOT_SAFE": 1}
).to(device)
model.config.pad_token_id = tokenizer.pad_token_id

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}

baseline_trainer = Trainer(
    model=model,
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)
baseline_results = baseline_trainer.evaluate()

# 3. LoRA Fine-Tuning
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=4, lora_alpha=32, lora_dropout=0.1,
    modules_to_save=["score"]
)
lora_model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./training_outputs",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="none"
)

lora_trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)
lora_trainer.train()
lora_model.save_pretrained(MODEL_SAVE_DIR)

# 4. Final Comparison
inference_model = AutoPeftModelForSequenceClassification.from_pretrained(
    MODEL_SAVE_DIR, num_labels=2
).to(device)
inference_model.config.pad_token_id = tokenizer.pad_token_id

inference_trainer = Trainer(model=inference_model, eval_dataset=tokenized["validation"], compute_metrics=compute_metrics)
finetuned_results = inference_trainer.evaluate()

# Save final results to JSON
results = {
    "baseline_accuracy": baseline_results['eval_accuracy'],
    "finetuned_accuracy": finetuned_results['eval_accuracy'],
    "improvement": finetuned_results['eval_accuracy'] - baseline_results['eval_accuracy']
}
with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)
