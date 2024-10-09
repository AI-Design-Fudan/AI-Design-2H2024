import torch
from gpt_model import GPTConfig, GPTModel
import sys
import sentencepiece as spm

# 设置设备（CUDA 或 CPU）
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_path, config):
    # 初始化模型
    model = GPTModel(config)
    # 加载模型参数
    model.load_state_dict(torch.load(model_path, map_location=device))
    # 将模型移动到指定设备
    model.to(device)
    # 设置模型为评估模式
    model.eval()
    return model

def main():
    # 加载配置
    config = GPTConfig()
    config.batch_size = 1  # 推理时通常使用 batch_size = 1
    config.dropout = 0.0   # 推理时将 dropout 设置为 0

    # 加载模型
    model_path = './gpt_model.pth'
    model = load_model(model_path, config)

    # 加载 tokenizer
    model_file = "tale.model"
    from data_set import load_tokenizer
    flag, sp = load_tokenizer(model_file)
    if not flag:
        print(f"load tokenizer model from: {model_file} failed")
        sys.exit(1)

    # 进行推理
    user_inputs = ["郭靖一掌挥出", "黄蓉突然想到", "周伯通好奇心大起", "洪七公哈哈大笑", "方文山大醉"]
    for user_input in user_inputs:
        # 将输入文本转换为 token id
        context = torch.tensor([sp.encode(user_input)], dtype=torch.int32, device=device)
        print(f"输入: {user_input}")

        # 调用生成函数
        gpt_output = model.generate(context, max_new_tokens=50)[0].tolist()

        # 将生成的 token id 转换回文本
        generated_text = sp.decode(gpt_output)
        print(f"生成的文本: {generated_text}")
        print("\n" + "="*50)

if __name__ == '__main__':
    main()
