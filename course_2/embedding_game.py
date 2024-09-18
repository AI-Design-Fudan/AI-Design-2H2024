import gensim.downloader
import numpy as np

# 下载并加载GloVe模型
model = gensim.downloader.load("glove-wiki-gigaword-50")

# 计算经典的例子
# result_vector = model["father"] - model["man"] + model["woman"]
# result_vector = model["king"] - model["man"] + model["woman"]
result_vector = model["nephew"] + model["woman"] - model["man"]


# 找出与结果向量最接近的词
most_similar_word = model.similar_by_vector(result_vector, topn=2)

print(f"Most similar word: {most_similar_word}")


