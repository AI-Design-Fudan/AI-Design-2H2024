from datasets import load_dataset

dataset = load_dataset('lighteval/mmlu', 'abstract_algebra')
train_data = dataset["auxiliary_train"]
first_10_sample = train_data.select(range(10))
for i in first_10_sample:
    print(i['question'])
    print(i['subject'])
    print(i['choices'])
    print(i['answer'])
    