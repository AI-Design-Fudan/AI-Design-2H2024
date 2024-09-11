import transformers
import torch
import datasets
from datasets import load_dataset


model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
# <eos bos>

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=512,
)
print(outputs[0]["generated_text"][-1])


# def conversation(outputs, prompt):
#     prompt_input={"role":"user", "content": prompt}
#     outputs[0]["generated_text"].append(prompt_input)
#     outputs=pipeline(
#         outputs[0]["generated_text"],
#         max_new_tokens=512,
#     )
#     print(outputs[0]["generated_text"][-1])
#     return outputs


def eval(prompt):
    prompt_input = {"role":"user", "content": prompt}
    messages_input = [
            {"role": "system", "content": "You are a helpful assistant. you are going to solve a multiple choices question. Think about is question step by step then give your answer. the answer should be in this format:'your thinking, answer: your choice', example: 'answer: B'"},
            {"role": "user", "content": prompt},
    ]
    outputs = pipeline(
        messages_input,
        max_new_tokens=512,
    )
    # print(outputs[0]["generated_text"][-1])
    return outputs



# while(1):
#     prompt=input("ask me for anything\n")
#     if prompt == "q":
#         outputs = conversation(outputs, prompt)
#     else:
#         break






labels = ["A", "B", "C", "D"]


char_to_int = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3
}



correct_answer = 0
dataset = load_dataset('lighteval/mmlu', 'abstract_algebra')
train_data = dataset["auxiliary_train"]
sample_size = 10
first_10_sample = train_data.select(range(sample_size))
for i in first_10_sample:
    question = i["question"]
    choices = i["choices"]
    formatted_prompt = f"{question}. \n A: {choices[0]}, B: {choices[1]}, C: {choices[2]}, D: {choices[3]}"
    output = eval(prompt=formatted_prompt)
    answer = output[0]["generated_text"][-1]["content"]
    print(answer)
    pos_1 = answer.find("answer: ")
    pos_2 = answer.find("Answer: ")
    char_answer=""
    if pos_1 != -1:
        print(answer[pos_1+8])
        char_answer=answer[pos_1+8]
    elif pos_2 !=-1:
        print(answer[pos_2+8])
        char_answer=answer[pos_2+8]
    else:
        print("none")
    
    int_answer = char_to_int.get(char_answer, -1)
    # print(int_answer)
    # print(type(i["answer"]))
    if(int_answer==i["answer"]):
        correct_answer += 1

print(correct_answer/sample_size)

