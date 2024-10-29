import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

raw_datasets = load_dataset("glue", "mrpc")
raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0]


batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()