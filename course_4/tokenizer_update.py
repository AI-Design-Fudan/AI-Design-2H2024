from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-3.2-1B-Instruct"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# 定义新的 tokens
new_tokens = ["<|reasoning_start|>", "<|reasoning_end|>"]

# 增加 tokens 到 tokenizer
num_new_tokens = tokenizer.add_tokens(new_tokens)

print(f"Added {num_new_tokens} new tokens: {new_tokens}")

# 调整模型的嵌入层大小以适应新的 tokens
model.resize_token_embeddings(len(tokenizer))

# 检查 tokenizer 是否包含新的 tokens
print(f"<|reasoning_start|> token ID: {tokenizer.convert_tokens_to_ids('<|reasoning_start|>')}")
print(f"<|reasoning_end|> token ID: {tokenizer.convert_tokens_to_ids('<|reasoning_end|>')}")