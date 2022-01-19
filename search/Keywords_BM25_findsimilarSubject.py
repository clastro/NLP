import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
from gensim import corpora
from gensim.models.fasttext import FastText
from konlpy.tag import Mecab
from rank_bm25 import BM25Okapi
import nmslib
import gensim
import time
import pymysql
from ftfy import fix_text, explain_unicode
from collections import Counter

#!pip install rank_bm25 --quiet #install BM25
#!pip install --no-binary :all: nmslib #install nmslib
#!pip install rank_bm25 --quiet #install BM25
#!pip install --no-binary :all: nmslib #install nmslib
#!pip install ftfy
#!pip install spacy
#!pip install nmslib

# if notebook
# pd.set_option('display.max_colwidth', -1)
# plt.style.use('fivethirtyeight')

conn  = pymysql.connect(host=host, port=port,
                     user=user,passwd=passwd,  
                     db=db)  

cursor = conn.cursor()
conn.set_charset('utf8')

mecab = Mecab(dicpath='c:/mecab/mecab-ko-dic/')

sql = "Select * FROM youtube_video_lists"
cursor.execute(sql)

columns = [i[0] for i in cursor.description]
df_video_lists = pd.DataFrame(cursor.fetchall(),columns=columns)

def clean_text(readData):
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', readData)
    return text
  
df_video_lists = df_video_lists.dropna(subset=['title'])

df_video_lists['morphs'] = df_video_lists['title'].apply(lambda x : clean_text(x))
df_video_lists['morphs'] = df_video_lists['morphs'].apply(lambda x : mecab.morphs(x))

ft_model = FastText(
    sg=1, # use skip-gram: usually gives better results
    vector_size=100, # embedding dimension (default)
    window=5, # window size: 10 tokens before and 10 tokens after to get wider context
    min_count=5, # only consider tokens with at least n occurrences in the corpus
    negative=15, # negative subsampling: bigger than default to sample negative examples more
    min_n=2, # min character n-gram
    max_n=5 # max character n-gram
)

ft_model.build_vocab(df_video_lists['morphs'])

ft_model.train(
    df_video_lists['morphs'],
    epochs=5,
    total_examples=ft_model.corpus_count, 
    total_words=ft_model.corpus_total_words)

ft_model.save('YT_fasttext.model')

#ft_model = FastText.load('YT_fasttext.model')

df_video_lists = df_video_lists[df_video_lists['morphs'].str.len()>=1]
df_video_lists.reset_index(drop=True,inplace=True)

bm25 = BM25Okapi(df_video_lists['morphs'])
weighted_doc_vects = []

for i,doc in tqdm(enumerate(df_video_lists['morphs'])):
    doc_vector = []
    for word in doc:
        vector = ft_model.wv.get_vector(word)
        weight = (bm25.idf[word] * ((bm25.k1 + 1.0)*bm25.doc_freqs[i][word])) 
        / 
        (bm25.k1 * (1.0 - bm25.b + bm25.b *(bm25.doc_len[i]/bm25.avgdl))+bm25.doc_freqs[i][word])
        weighted_vector = vector * weight
        doc_vector.append(weighted_vector)
    doc_vector_mean = np.mean(doc_vector,axis=0)
    weighted_doc_vects.append(doc_vector_mean)
    
    
pickle.dump( weighted_doc_vects, open( "weighted_doc_vects.p", "wb" ) )

with open( "weighted_doc_vects.p", "rb" ) as f:
    weighted_doc_vects = pickle.load(f)
    
 # initialize a new index, using a HNSW index on Cosine Similarity - can take a couple of mins
index = nmslib.init(method='hnsw', space='cosinesimil')
index.addDataPointBatch(data)
index.createIndex({'post': 2}, print_progress=True)

counter = Counter()
input_list = ['룩북','코디','봄','신상','원피스']

# querying the index:
#input = '매트 틴트 립스틱 mlbb 글로시'.lower().split()

query = [ft_model.wv.get_vector(vec) for vec in input_list]
query = np.mean(query,axis=0)

t0 = time.time()
ids, distances = index.knnQuery(query, k=100)
t1 = time.time()
print(f'Searched {df_video_lists.shape[0]} records in {round(t1-t0,4) } seconds \n')
for i,j in zip(ids,distances):
    print(1/round(j,2))
    print(df_video_lists.title.values[i])
    print(df_video_lists.channel_id.values[i])
    counter[df_video_lists.channel_id.values[i]] += 1/round(j,2)
    
 """
 Searched 1366829 records in 0.0 seconds 

33.33333407839141
여름원피스 추천 하울2 여행 휴양지룩 데일리 몽돌원피스 룩북/ 해온룩북
UCciz-iIa6yT-JGBlN7q_h6A
33.33333407839141
데일리 원피스 룩북????
UCvnCBA0eFMs3Hoiua-Kq7YA
33.33333407839141
두번째 미니원피스 룩북???? (feat. 악세사리) | 여름룩북 | 파티룩북 | 원피스추천 | 패션하울 | summer lookbook | fashion haul | 유피로그
UCCVsEn9ZHyesWG6rJVbiL9Q
25.00000055879356
여름 원피스 추천 하울 / 직장인 하객룩 코디 몽돌 룩북 / 해온 룩북
UCciz-iIa6yT-JGBlN7q_h6A
25.00000055879356

 """
    
dict(sorted(counter.most_common(), key=lambda x : x[1],reverse=True))

"""
 'UCciz-iIa6yT-JGBlN7q_h6A': 188.3333345626792,
 'UCg-dGGuwI6_H91qEmMfoWng': 135.00000078231102,
 'UCfcW__GMeF-sK1joelFbprw': 116.6666655490796,
 'UCI4XVFP2QP2EEX5W__qb-bg': 113.33333288629852,
 'UCa46gNXs22obCQe1JZnegSA': 113.33333288629852,
 'UCCVsEn9ZHyesWG6rJVbiL9Q': 75.00000167638068,
 'UCtQWzbv0mauBAQ0fO-diP8w': 75.00000167638066,
"""
