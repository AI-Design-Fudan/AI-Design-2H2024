# 2nd-half 2024 AI course from Patrick Chiang

## Course 1
The first course meant to give every one of you a glance of how to use and Transformers Language Model.  
You can try some open source model by replace the model name on hf.py file and run it.
```
cd course_1
python hf.py
```  
Notice that example case use LLaMa 3.1 8B Instructed model. This mode need both admission from Meta and a high performance grapic card.  
You can replace that model with `Qwen/Qwen2-0.5B-Instruct` to give a try.  
Model evaluation is also available here.
```commandline
cd course_1
cd eval
cd lm-evaluation-harness
lm_eval --model hf \
    --model_args meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```
You may also replace the model name with your own model.  
The `task` value specifies the dataset you use to evaluate your model.  
