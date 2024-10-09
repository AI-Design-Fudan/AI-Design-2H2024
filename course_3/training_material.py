import os
import json

# 设置你的数据集文件夹路径
dataset_path = "./wiki_zh"
output_file = "tale.txt"

def extract_text_from_json_file(file_path):
    """ 从一个JSON文件中提取出所有的text字段并返回 """
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # 确保JSON数据包含"text"字段
                if 'text' in data:
                    texts.append(data['text'])
            except json.JSONDecodeError:
                print(f"JSON解码错误: {file_path}")
    return texts

def extract_texts_from_directory(directory_path):
    """ 递归遍历文件夹中的所有文件，提取text字段 """
    all_texts = []
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_name.startswith("wiki_"):  # 仅处理wiki开头的文件
                texts = extract_text_from_json_file(file_path)
                all_texts.extend(texts)
    return all_texts

def main():
    # 从数据集文件夹中提取所有的text数据
    all_texts = extract_texts_from_directory(dataset_path)

    # 打开 tale.txt 文件并将所有提取到的text追加进去
    with open(output_file, 'a', encoding='utf-8') as f:
        for text in all_texts:
            # 每个文本之间加入换行符
            f.write(text + "\n\n")

    print(f"成功将数据追加到 {output_file} 文件中，共处理了 {len(all_texts)} 个文本段。")

if __name__ == "__main__":
    main()
