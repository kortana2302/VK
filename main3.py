import pandas as pd
import streamlit as st
import mpld3
import streamlit.components.v1 as components
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import numpy as np
import pickle
import re

def f2(x):
    return " ".join(map(lambda x:x.lower(),re.findall(r"(\w*[ЦцКкНнГгШшЩщЗзХхФфВвПпРрЛлДдЖжЧчСсМмТтБб])[УуЕеЫыАаОоЭэЯяИиЬьЮюЯяЙй]*",x)))
def vectorize(sentence):
    words = sentence.split()
    words_vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if len(words_vecs) == 0:
        return np.zeros(50)
    words_vecs = np.array(words_vecs)
    return words_vecs.mean(axis=0)

st.title('Поиск аудитории')


col1, col2 = st.columns(2)
text = col1.text_input(label = 'Ключевые слова')
df_new = pd.read_csv('data (2).csv')

if st.button('Расчет') and len(text)>0: 
    filename = 'model.sav'
    kmeans = pickle.load(open(filename, 'rb'))
    
    filename = 'w2v_model.sav'
    w2v_model = pickle.load(open(filename, 'rb'))

    text = f2(text)
    embedding = np.array(vectorize(text)).reshape(1, -1)
    pred = kmeans.predict(embedding.astype('double'))
    df_cluster = df_new[df_new['cluster_AP'] == pred[0]]
    if len(df_cluster)==0:
        col2.write('Нет рекомендаций') 
    else:
        df_sample = df_cluster.head(10)
        df_sample = df_sample[['Имя','Фамилия']].reset_index(drop = True)
        col2.write(df_sample)
