import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型和分词器
model_id = "./final_model"
tokenizer_id = "./final_model"



tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
model = AutoModelForCausalLM.from_pretrained(model_id)


# 设置 pad_token_id 和 eos_token_id
pad_token = "<|pad_id|>"
eos_token = "<|end_of_text|>"

tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)

print("pad_token_id:", tokenizer.pad_token_id)
print("eos_token_id:", tokenizer.eos_token_id)

# 确保它们不同
if tokenizer.pad_token_id == tokenizer.eos_token_id:
    raise ValueError("pad_token_id 和 eos_token_id 不能相同！")



# 将模型移动到合适的设备
device = "cuda:1"  # 或 "cpu" 如果没有 GPU
model.to(device)

# 准备输入文本
input_text = "who are you"  # 替换为您想要推理的文本

# 使用分词器编码输入文本
encoding = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)

# 进行推理
with torch.no_grad():
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=1024, pad_token_id=tokenizer.eos_token_id)  # max_length 设置为生成的最大长度

# 解码生成的 token 为文本
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

print("生成的文本:", generated_text)
