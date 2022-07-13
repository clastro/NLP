import numpy as np 
import pandas as pd
import re

from tensorflow.keras import Input,initializers, regularizers, constraints, optimizers
from tensorflow.keras.layers import LSTM , Embedding, Dropout , Activation, GRU, Flatten, Bidirectional, GlobalMaxPool1D, Convolution1D, concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow import keras

#!curl -s https://raw.githubusercontent.com/teddylee777/machine-learning/master/99-Misc/01-Colab/mecab-colab.sh | bash #colab에서 Mecab 설치

from konlpy.tag import Mecab
mecab = Mecab()
data = pd.read_csv('all2.csv')

df = data.sample(frac=1,random_state=0) # 랜덤하게 섞기

df['진단명2'] = df['진단명2'].map({"ADHD":0,"정신지체":1})

df['의뢰사유'] = df['의뢰사유'].astype(str) # 문자열로 형 변환
df['검사태도'] = df['검사태도'].astype(str) # 문자열로 형 변환
df['인지기능및적응기능'] = df['인지기능및적응기능'].astype(str) # 문자열로 형 변환

df["pos_cause"] = df["의뢰사유"].apply(lambda x : mecab.pos(x)) #품사 태깅
df["pos_attitude"] = df["검사태도"].apply(lambda x : mecab.pos(x)) #품사 태깅
df["pos_function"] = df["인지기능및적응기능"].apply(lambda x : mecab.pos(x)) #품사 태깅

df['combine'] = df['combine'].astype(str) # 문자열로 형 변환
df["pos_combine"] = df["combine"].apply(lambda x : mecab.pos(x)) #품사 태깅

def apply_pos(content_pos):    
    pos_list = []
    pos_tagging = ['NNG','NNP','MAG','XR','SL'] #명사 부사 어근 외국어 -> 품사 조절 해 볼 수 있을 듯
    for x in content_pos:#NNG,NNP,VA+ETM,MAG,NNBC
        if x[1] in pos_tagging:
            pos_list.append(x[0])
    return pos_list
  
df["cause"] = df["pos_cause"].apply(lambda x : apply_pos(x))
df["attitude"] = df["pos_attitude"].apply(lambda x : apply_pos(x))
df["function"] = df["pos_function"].apply(lambda x : apply_pos(x))
df["text_pos"] = df["pos_combine"].apply(lambda x : apply_pos(x)) #전체 어휘


adam = Adam(learning_rate=0.0001)
max_features = 8000 # 어휘 수 조절해 볼 필요 5000은 오히려 accuracy 떨어짐
tokenizer = Tokenizer(num_words=max_features) 
tokenizer.fit_on_texts(df['text_pos'])

df_train = df[:1458]
df_test = df[1458:]

maxlen = 256 

y_train = df_train['진단명2']

cause_tokenized_train = tokenizer.texts_to_sequences(df_train['cause'])
X_cause_train = pad_sequences(cause_tokenized_train, maxlen=maxlen)

attitude_tokenized_train = tokenizer.texts_to_sequences(df_train['attitude'])
X_attitude_train = pad_sequences(attitude_tokenized_train, maxlen=maxlen)

function_tokenized_train = tokenizer.texts_to_sequences(df_train['function'])
X_function_train = pad_sequences(function_tokenized_train, maxlen=maxlen)

embed_size = 256

input_cause = Input(shape=(None,),name='cause')
x = Embedding(max_features, embed_size)(input_cause)
x = Bidirectional(LSTM(32, return_sequences = True))(x)
x = BatchNormalization()(x)
x = Dense(20, activation='relu')(x)
x = Dropout(0.05)(x)
x = GlobalMaxPool1D()(x)

input_attitude = Input(shape=(None,),name='attitude')
y = Embedding(max_features, embed_size)(input_attitude)
y = Bidirectional(LSTM(32, return_sequences = True))(y)
y = BatchNormalization()(y)
y = Dense(20, activation='relu')(y)
y = Dropout(0.05)(y)
y = GlobalMaxPool1D()(y)

input_function = Input(shape=(None,),name='function')
z = Embedding(max_features, embed_size)(input_function)
z = Bidirectional(LSTM(32, return_sequences = True))(z)
z = BatchNormalization()(z)
z = Dense(20, activation='relu')(z)
z = Dropout(0.05)(z)
z = GlobalMaxPool1D()(z)

w = concatenate([x, y, z])
w = Dense(20, activation='relu')(w)
out =  Dense(1, activation='sigmoid')(w)

model = Model(inputs=[input_cause, input_attitude, input_function], outputs=out)
model.compile(loss='binary_crossentropy', 
                   optimizer='adam', 
                   metrics=['accuracy'])

#model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
model_check = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

batch_size = 128 
epochs = 20
history = model.fit([X_cause_train,X_attitude_train,X_function_train],y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2,callbacks=[es,model_check])

cause_tokenized_test = tokenizer.texts_to_sequences(df_test['cause'])
X_cause_test = pad_sequences(cause_tokenized_test, maxlen=maxlen)
attitude_tokenized_test = tokenizer.texts_to_sequences(df_test['attitude'])
X_attitude_test = pad_sequences(attitude_tokenized_test, maxlen=maxlen)
function_tokenized_test = tokenizer.texts_to_sequences(df_test['function'])
X_function_test = pad_sequences(function_tokenized_test, maxlen=maxlen)

pred = model.predict([X_cause_test,X_attitude_test,X_function_test])

from sklearn.metrics import *

confusion_matrix(df_test['진단명2'],np.round(pred))
f1_score(df_test['진단명2'],np.round(pred))

