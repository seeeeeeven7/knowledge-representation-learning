import jieba
import jieba.posseg as pseg  

import os  
import sys  
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  
  

with open('FB15k_mid2description.txt', 'r', encoding='utf8') as infile:
    lines = infile.read().splitlines()

entity_ids = []
corpus = []
for line in lines:
    pos = line.find('\t')
    entity_ids.append(line[:pos])
    content = line[pos:].strip()[1:-5]
    corpus.append(content)

print(entity_ids[0])

vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
word = vectorizer.get_feature_names()
weight = tfidf.toarray()
entities = []
for i in range(len(weight)):
    candidates = []
    for j in range(len(word)):  
        if weight[i][j] > 0.1:
            candidates.append({'word': word[j], 'weight': weight[i][j]})
    candidates = sorted(candidates, key=lambda k: k['weight'], reverse = True)
    entities.append({'entity_id': entity_ids[i], 'keywords': candidates[:5]})

with open('FB15k_mid2description_tfidf.json', 'w', encoding='utf8') as outfile:
    print(entities, file=outfile)