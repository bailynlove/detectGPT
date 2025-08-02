import torch, joblib, pandas as pd
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from utils import deberta_tok, deberta_model

df = pd.read_csv("data/full_data_0731_aug_4.csv")[["text", "prompt_type"]].dropna()
df["label"] = (df["prompt_type"] != 0).astype(int)

ds = Dataset.from_pandas(df[["text", "label"]])


def tok(batch):
    return deberta_tok(batch["text"], truncation=True, padding="max_length", max_length=512)


ds = ds.map(tok, batched=True).train_test_split(test_size=0.1, seed=42)

args = TrainingArguments(
    output_dir="models/deberta_ft",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
    evaluation_strategy="epoch",
    save_total_limit=1,
    report_to="none",
)
trainer = Trainer(
    model=deberta_model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
)
trainer.train()
trainer.save_model("models/deberta_ft")
