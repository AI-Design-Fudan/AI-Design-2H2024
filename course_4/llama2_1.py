import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from datasets import load_dataset, Dataset
import json


# 加载数据集
dataset = load_dataset('SkunkworksAI/reasoning-0.01')
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer_id = "meta-llama/Llama-3.2-1B-Instruct"
# 加载模型和分词器，注意将torch_dtype改为float32以支持MPS
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,  # 修改为float32
    device_map="auto"
)

# 检查并设置 eos_token_id 和 pad_token_id，如果它们没有被设置
if tokenizer.eos_token_id is None:
    # 手动设置 eos_token_id，通常是模型的特殊标记
    tokenizer.eos_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else 0  # 假设0是合理的标记，需根据模型实际情况调整

if tokenizer.pad_token_id is None:
    # 设置 pad_token_id 为 eos_token_id 或指定其他合适的值
    tokenizer.pad_token_id = tokenizer.eos_token_id

# # 设置 Llama 模型的特殊标记 ID
# tokenizer.pad_token_id = 0
# tokenizer.eos_token_id = 2
# tokenizer.bos_token_id = 1

# 文件路径
file_path = "./formatted_reasoning_dataset.json"

# 加载 JSON 文件
with open(file_path, 'r') as f:
    raw_json = json.load(f)



# 遍历 JSON 数据并应用模板
processed_data = []

for entry in raw_json:
    # 假设您的数据是交替的 'user' 和 'assistant' 对话形式
    raw = tokenizer.apply_chat_template(entry, tokenize=False)
    processed_data.append(raw)

# 查看处理后的数据
print(processed_data[:2])  # 打印前两条处理后的数据


# 创建 Dataset 对象
dataset = Dataset.from_dict({'text': processed_data})

# 将数据集划分为训练集和验证集，按 90% 训练，10% 验证的比例
split_dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)

train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']


def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=4096,
        padding='max_length',
    )


# 对训练集和验证集进行分词
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 自回归模型不使用掩码语言模型
)


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # 根据您的显存大小进行调整
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # 如果批量大小较小，可以增加梯度累积步数
    evaluation_strategy='steps',
    eval_steps=500,  # 每500步进行一次评估
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    fp16=True,  # 如果使用 MPS（苹果芯片），请将 fp16 设为 False
    bf16=False,  # MPS 不支持 bf16
    dataloader_num_workers=2,  # 根据需要调整
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
