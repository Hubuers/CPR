from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Others (文本、字符处理)
import numpy as np

# 导入数据集
## 读入概念-词条字典
import pandas as pd
df = pd.read_excel('E:/实验相关代码/m3代码/m3_code/data/概念前400单词.xlsx')
A_words = df['A_indexWords'].tolist()
B_words = df['B_indexWords'].tolist()


##预训练词向量导入与生成
### Create sequence
vocabulary_size = 20000

# 该类允许对文本语料库进行矢量化，方法是将每个文本转换为整数序列
#（每个整数是字典中标记的索引）或向量，其中每个标记的系数可以是二进制的，基于字数，基于tf idf。。。
tokenizer = Tokenizer(num_words= vocabulary_size) # 据词频保留的最大字数。只保留最常见的num_words-1单词。
tokenizer.fit_on_texts(df['A_indexWords'])
sequences = tokenizer.texts_to_sequences(df['A_indexWords'])
dataA = pad_sequences(sequences, maxlen=400, truncating='post')
embeddings_index = dict()
f = open('E:/实验相关代码/m3代码/m3_code/wikipedia2vec/enwiki_20180420_100d.txt',encoding='utf-8')
# f = open('D:/实验数据/glove/glove.6B/glove.6B.100d.txt',encoding='utf-8')
count = 0
for line in f:
    count += 1
    if count == 1:
        pass
    else:
        values = line.split()
        word = ''.join(values[:-100])
        coefs = np.asarray(values[-100:], dtype='float32')
        embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrixA = np.zeros((vocabulary_size, 100))
for word, index in tokenizer.word_index.items():
    if index > vocabulary_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrixA[index] = embedding_vector

### Create sequence
vocabulary_size = 20000
# 该类允许对文本语料库进行矢量化，方法是将每个文本转换为整数序列

#（每个整数是字典中标记的索引）或向量，其中每个标记的系数可以是二进制的，基于字数，基于tf idf。。。
tokenizer = Tokenizer(num_words= vocabulary_size) # 据词频保留的最大字数。只保留最常见的num_words-1单词。
tokenizer.fit_on_texts(df['B_indexWords'])
sequences = tokenizer.texts_to_sequences(df['B_indexWords'])
dataB = pad_sequences(sequences, maxlen=400,truncating='post')
embeddings_index = dict()
f = open('E:/实验相关代码/m3代码/m3_code/wikipedia2vec/enwiki_20180420_100d.txt',encoding='utf-8')
# f = open('D:/实验数据/glove/glove.6B/glove.6B.100d.txt',encoding='utf-8')
count = 0
for line in f:
    count+=1
    if count == 1:
        pass
    else:
        values = line.split()
        word = ''.join(values[:-100])
        coefs = np.asarray(values[-100:], dtype='float32')
        embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrixB = np.zeros((vocabulary_size, 100))
for word, index in tokenizer.word_index.items():
    if index > vocabulary_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrixB[index] = embedding_vector

# word2vec + LSTM + (output)

from keras.layers import LSTM, Embedding, Dense
from keras import Input, Model
from keras.layers import Concatenate
import keras

#M1

# first input model
Input_A = Input(batch_shape=(None,400))
Embedding_layer_A = Embedding(input_dim=vocabulary_size, output_dim=100, weights=[embedding_matrixA], input_length=400, trainable=False)(Input_A)
LSTM_A = LSTM(units=32, name='lstm_1')(Embedding_layer_A)

# second input model
Input_B = Input(batch_shape=(None, 400), name='input_B')
Embedding_layer_B = Embedding(input_dim=vocabulary_size, output_dim=100, weights=[embedding_matrixB], input_length=400, trainable=False)(Input_B)
LSTM_B=LSTM(units=32, name='lstm_2')(Embedding_layer_B)

added = Concatenate()([LSTM_A,LSTM_B])
drop = Dense(76)(added)
print(added.shape)
output = Dense(1,activation='sigmoid')(drop)

model = keras.Model(inputs=[Input_A, Input_B], outputs=[output],)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

## 导入概念的前400单词

representation_model = Model(inputs = model.inputs, outputs=model.get_layer('lstm_1').output)

LSTM_A_output = representation_model.predict([dataA, dataB])

print(LSTM_A_output.shape)
print(type(LSTM_A_output))
print(LSTM_A_output)

representation_model = Model(inputs = model.inputs, outputs=model.get_layer('lstm_2').output)

LSTM_B_output = representation_model.predict([dataA, dataB])

print(LSTM_B_output.shape)
print(type(LSTM_B_output))
print(LSTM_B_output)


import csv
with open("E:/实验相关代码/m3代码/m3_code/data/M1/A_32.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(LSTM_A_output)

import csv
with open("E:/实验相关代码/m3代码/m3_code/data/M1/B_32.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(LSTM_B_output)

## A_32.csv和B_32.csv就是拼起来到一个文件中就是64个特征值