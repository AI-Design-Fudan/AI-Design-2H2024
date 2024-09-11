import torch
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B')

model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B', torch_dtype=torch.float16)
model.to(device)

# Prepare data (this is just an example)
# inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
# inputs.to(device)
# # Training parameters
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
# loss_fn = torch.nn.CrossEntropyLoss()

# Eval loop
model.eval()



# 准备输入文本
input_text = "你好，Llama！请为我生成一些文本。"

# 使用分词器进行编码
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)


# print(input_ids)

# 使用模型生成输出
with torch.no_grad():
    output_ids = model.generate(input_ids, max_length=500, num_return_sequences=1)

# 解码生成的文本
# output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)


# 输出结果
print(output_text)
