from transformers import AutoTokenizer

# 替换为你感兴趣的模型名称
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
print(tokenizer.vocab)
