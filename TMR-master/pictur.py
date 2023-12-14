import json

# 假设JSON文件名为data.json
with open('datasets/annotations/humanml3d/annotations.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

text_list = []
for segment_id, segment in data.items():

    text_list.append(segment['annotations'][0]['text'])

print(len(text_list))
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# 示例句子
sentence = "The cat is chasing the mouse, has been playing with a toy, and will do it again."

# 对句子进行分词
tokens = word_tokenize(sentence)

# 执行词性标注
tagged = pos_tag(tokens)

# 定义动词的词性标记
verb_tags = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

# 定义常见的非实义动词列表
non_lexical_verbs = {"be", "am", "is", "are", "was", "were", "been", "have", "has", "had", "do", "does", "did"}

# 计算实义动词的数量
lexical_verb_count = sum(1 for word, tag in tagged if tag in verb_tags and word.lower() not in non_lexical_verbs)

print(lexical_verb_count, tagged)