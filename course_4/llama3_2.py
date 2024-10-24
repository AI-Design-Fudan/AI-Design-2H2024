import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B-Instructed"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

pipe("The key to life is")
print(pipe("The key to life is"))
