import torch
from model import GPT, GPTConfig
from transformers import GPT2Tokenizer

# 加载分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 使用与训练时相同的配置初始化模型
config = GPTConfig(
    vocab_size=50304,
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    bias=True
)
model = GPT(config)

# 加载保存的模型权重
model_load_path = './GPT2_trained/trained_gpt_model_final.pth'
model.load_state_dict(torch.load(model_load_path))

# 将模型移动到 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# 定义文本生成函数
def generate_text(prompt, max_new_tokens=50, temperature=1.0, top_k=None):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text

# 示例用法
if __name__ == '__main__':
    prompt = "Once upon a time"
    generated_text = generate_text(prompt, max_new_tokens=200)
    print("输入提示：", prompt)
    print("生成文本：", generated_text)
