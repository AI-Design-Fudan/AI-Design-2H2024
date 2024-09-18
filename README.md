# 2nd-half 2024 AI course from Patrick Chiang  
  
First, you need to install the requirements:  
```
cd AI-Design-2H2024
pip install -r requirements.txt
```



## Course 1
The first course is designed to provide an overview of the Transformers architecture and Language Models. You can experiment with various open-source models by replacing the model name in the `hf.py` file and running it.
```
cd course_1
python hf.py
```  
Please note that the example uses the LLaMa 3.1 8B Instruct model. This model requires both permission from Meta and a high-performance graphics card. You can replace it with the `Qwen/Qwen2-0.5B-Instruct` model to try out an alternative. Model evaluation is also available here.
```commandline
cd course_1
cd eval
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install .
lm_eval --model hf \
    --model_args meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```
You may also replace the model name with your own model.  
The `task` value specifies the dataset you use to evaluate your model.  
  
### Course_2
Course_2 will help you dive into the basics of how transformers get data ready for action. We'll cover how tokenizer and embedding work together to prep data, making it just right for transformers to use. It's all about turning words into a form that computers can understand and work with.

[//]: # (### Course_3)

[//]: # (Course_3 will assist you explore the full structure of realized transformers in this brief overview, where we'll cover everything from tokenization to attention mechanisms. Get a clear view of how these components work together to do text processing.)

[//]: # ()
[//]: # (### related papers)

[//]: # ([OneBit: Towards Extremely Low-bit Large Language Models]&#40;https://arxiv.org/abs/2402.11295&#41;)

[//]: # ()
[//]: # ([Efficient Large Language Models: A Survey]&#40;https://arxiv.org/abs/2312.03863&#41;)

[//]: # ()
[//]: # ([The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits]&#40;https://arxiv.org/abs/2402.17764&#41;)

[//]: # ()
[//]: # ([BitNet: Scaling 1-bit Transformers for Large Language Models]&#40;https://arxiv.org/abs/2310.11453&#41;)

[//]: # ()
[//]: # ([]&#40;&#41;)

[//]: # ()
[//]: # ()
[//]: # (## Week 2 Homework )

[//]: # (Your homework is run the code in course_3. I set a trap inside the code. Your target will be:)

[//]: # (1. successfully run the code.)

[//]: # (2. successfully train the model. &#40;loss will drop by training.&#41;)

[//]: # (3. be able to replace the training materials.)

[//]: # (4. modify the model.)

[//]: # ()
[//]: # (## Week 3 Homework )

[//]: # (Now you have the full version runable code, do some hand adjustment and see how these adjustments effect Hardware.)

[//]: # (Adjust the following config|structure of transformers model:)

[//]: # (1. number of heads)

[//]: # (2. transformer iterations)

[//]: # (3. context length)

[//]: # (4. embeddings)

[//]: # (5. quantization &#40;pending&#41;)

[//]: # ()
[//]: # ()
[//]: # (## Week 4 Content)

[//]: # (A new hand-made ViT model with MNIST datasets is ready to use. Feel free to use it and learn the structure of ViT. There is no homework left.)