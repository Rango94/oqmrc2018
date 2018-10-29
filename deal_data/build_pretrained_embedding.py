
import numpy as np
import json
import math
vectors_dic={}
with open('../../../../vector/vectors.txt','r',encoding='utf-8') as fo:
    for idx,line in enumerate(fo.readlines()):
        line=line.rstrip().split(' ')
        word,vector=[line[0],np.array([float(i) for i in line[1:]])]
        vectors_dic[word]=vector

with open('../../DATA/data/word_dic','r',encoding='utf-8') as fo:
    word_dic=json.load(fo)


embedding=[]
n=0
while len(word_dic)>0:
    for key in word_dic:
        if word_dic[key]==n:
            try:
                # norm=math.pow(np.sum(vectors_dic[key]**2),0.5)
                embedding.append(vectors_dic[key])
            except:
                embedding.append(np.random.random(128))
            n+=1
            word_dic.pop(key)
            break
embedding=np.array(embedding)
print(embedding.shape)

np.save('../../DATA/data/embedding',embedding)

