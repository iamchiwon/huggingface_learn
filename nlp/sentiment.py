from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1

classifier = pipeline("sentiment-analysis", device=device)
result = classifier([
  "I've been waiting for a HuggingFace course my whole life.",
  "I hate this so much!"
])
print(result)