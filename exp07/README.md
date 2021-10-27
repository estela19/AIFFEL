# Sentiment Classification  
네이버 영화 리뷰 데이터셋을 이용해 영화 리뷰를 보고 긍정적인 리뷰인지, 부정적인 리뷰인지 Binary Classification하는 모델을 만들어 보자!  


# Workflow  
### 1. DataLoader 
### 2. Add padding
### 3. Modeling  
### 4. Model Train  
### 5. Model Test  

# Embedding  
일반적으로 pretrain 된 embedding보다 원핫 embedding이 주어진 task에 맞게 더욱 잘 embedding 되기 때문에 일반적으로 pretrain 임베딩을 잘 사용하지 않는다.  
다만, 학습데이터가 작을 경우 미리 많은 데이터로 학습한 pretrain embedding이 더 효과적이다.  
본 모델에서 사용한 embedding은 keras embedding layer와 word2vec이다.  

## keras embedding layer  
원 핫 임베딩  

## word2vec 
cbow와 skip gram 형식으로 학습한 인베딩으로 오토인코더와 비슷한 성격을 가졌다.  

### 장점  
  * 빠르다
  * 비교적 정확한 벡터
  
### 단점  
  * 출현빈도가 적은 단어일 경우 벡터가 정확하지 않다.

## fasttext  
빈도가 적은 단어에 대한 학습과 OoV(out of vocabulary)에 대한 대처가 어려운 word2vec를 개선하기 위해 만들어졌다.  
이를 위해 subword로 분절한 후 원래 단어의 embedding을 구하는 방식을 택했다.  

### 학습방법  
1. 단어를 subword로 나눈다.
2. skip-gram을 활용, 각 subword에 대한 embedding vector에 주변단어의 context vector를 곱한다 .  
3. 위 값이 최대가 되도록 학습한다.  
   

# Model  
## LSTM  
하나의 lstm layer 와 dense layer로 구성되어 있다.  
```python
vocab_size = 10000    
word_vector_dim = 25

lstm_model = keras.Sequential()
lstm_model.add(keras.layers.Embedding(vocab_size, word_vector_dim))
lstm_model.add(keras.layers.LSTM(256, activation = 'relu'))
lstm_model.add(keras.layers.Dense(256, activation='relu'))
lstm_model.add(keras.layers.Dense(128, activation='relu'))
lstm_model.add(keras.layers.Dense(1, activation='sigmoid'))
```  

## Attention before LSTM  
attention layer를 통과한 후 lstm 연산을 시행한다.  
attention 방식에는 다양한 방법이 있지만 `embedding 된 input tensor`와 이 `input을 dense layer에 통과시킨 tensor`의 `dot product`를 통해서 구현하였다.  

```python
vocab_size = 10000   
word_vector_dim = 25
time_steps = 41

# input layer
inputs = keras.layers.Input(shape=(time_steps,))

embed = keras.layers.Embedding(vocab_size, word_vector_dim)(inputs)

# attention layer
a = keras.layers.Permute((2, 1))(embed) 

a = keras.layers.Dense(time_steps, activation='softmax')(a)
a_probs = keras.layers.Permute((2, 1), name='attention_vec')(a)
output_attention_mul  = keras.layers.multiply([embed, a_probs])


# lstm layer
lstm_out = keras.layers.LSTM(256, return_sequences=True, activation='relu')(output_attention_mul)

# fc layer
fc = keras.layers.Dense(64)(lstm_out)
output = keras.layers.Dense(1, activation='sigmoid')(fc)

attention_3d_model = keras.Model(inputs=[inputs], outputs=output)
```  


## Attention after LSTM  
lstm layer를 통과한 후 attention 연산을 수행한다.  
attention은 lstm의 보조적인 연산이라고 생각해 lstm layer를 거친 후 attention을 하는 것이 큰 의미가 없을 것으로 생각했으나  의외로 lstm layer를 거치기 전 attention을 하는 것과 비슷한 성능을 보였다.  

```python
vocab_size = 10000   
word_vector_dim = 25
time_steps = 41

# input layer
inputs = keras.layers.Input(shape=(time_steps,))

embed = keras.layers.Embedding(vocab_size, word_vector_dim)(inputs)

# lstm layer
lstm_out = keras.layers.LSTM(256, return_sequences=True, activation='relu')(embed)

# attention layer
a = keras.layers.Permute((2, 1))(lstm_out)
a = keras.layers.Dense(time_steps, activation='softmax')(a)
a_probs = keras.layers.Permute((2, 1), name='attention_vec')(a)
attention_mul = keras.layers.multiply([lstm_out, a_probs])

# fc layer
fc = keras.layers.Dense(64)(attention_mul)
output = keras.layers.Dense(1, activation='sigmoid')(fc)

Att_after_LSTM = keras.Model(inputs=[inputs], outputs=output)

Att_after_LSTM.summary()
```

## Transformer
transformer는 attention만으로 인코더와 디코더를 구현한 모델이다. 
본 모델에 대한 이해가 아직은 완벽하지 않아 이미 구현되어 있는 코드를 이용해 모델링했다.  
```python
embedding_dim = 32  # 각 단어의 임베딩 벡터의 차원
num_heads = 2  # 어텐션 헤드의 수
dff = 32  # 포지션 와이즈 피드 포워드 신경망의 은닉층의 크기
vocab_size = 10000
max_len = 41

inputs = keras.layers.Input(shape=(max_len,))
embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embedding_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embedding_dim, num_heads, dff)
x = transformer_block(x)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.1)(x)
x = keras.layers.Dense(20, activation="relu")(x)
x = keras.layers.Dropout(0.1)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)

transformer = keras.Model(inputs=inputs, outputs=outputs)
```  
Transformer의 성능은 이미 여러차례 검증이 되어있기에 가시적인 성능 향상을 기대했으나 의외로 단일 attention모델과 accuracy가 크게 다르지 않았다.  또한 valid accuracy가 다른 모델과 달리 1 epoch만에 0.84를 기록했지만 학습을 진행해도 크게 수렴하지 않는 모습을 보였으며 오히려 accuracy가 감소하는 것을 볼 수 있었다.  transformer를 적절히 학습시키기에 본 데이터셋이 작기 때문이라고 유추했다.  

## Self Attention  
Attention의 세가지 요소인 key, query, value가 모두 같은 attention을 Self attention 이라고 한다.  
self attention은 `keras_self_attention`을 통해서 쉽게 사용할 수 있다.  
```python
SelfAttention = keras.Sequential()
SelfAttention.add(keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len))
SelfAttention.add(keras.layers.LSTM(256, return_sequences=True, activation='relu'))
SelfAttention.add(SeqSelfAttention(attention_activation='sigmoid'))
SelfAttention.add(keras.layers.Dense(1, activation='sigmoid'))
```


# Result  
## Default Embedding  

| model                 | hidden  size | embedding  size | accuracy |
|-----------------------|--------------|-----------------|----------|
| LSTM                  | 256          | 25              | 0.804    |
| Attention before LSTM | 256          | 25              | 0.8399   |
| Attention after LSTM  | 256          | 25              | 0.8377   |
| Transformer           | -            | 32              | 0.8439   |
| Self Attention        | 256          | 32              | 0.8504   |  

## Word2Vec Embedding
| model                 | hidden  size | embedding  size | accuracy |
|-----------------------|--------------|-----------------|----------|
| LSTM                  | 128          | 200             | 0.8174   |
| Attention before LSTM | 256          | 200             | 0.7828   |
| Attention after LSTM  | 256          | 200             | 0.6993   |
| Self Attention        | 256          | 200             | 0.8193   | 

# Appendix  - Pororo
카카오브레인에서 Bert, Transformer등을 이용해 만든 자연어처리 라이브러리 [pororo](github.com/kakaobrain/pororo)를 이용하여 테스트해 보았다.   
pororo 라이브러리의 사용방법은 다음과 같다.  
```
pip install pororo
```
```python
from pororo import Pororo

movie = Pororo(task='sentiment', model='brainbert.base.ko.nsmc', lang='ko')

movie("비주얼로 설득시키는 대서사시.")
```  
위와 같이 pororo 라이브러리를 사용했을때 `Negative` 또는 `Positive`를 리턴해준다.  
해당 라이브러리로 네이버 영화리뷰 test dataset을 사용했을때 **0.9103**으로 매우 높은 정확도를 보여주었다.   

# Think about..
* 직접 구현한 attention에서 padding에 집중하는 등 잘못된 attention vector 존재했는데 어떻게 개선할까?  
* attention할때 BOS, EOS, UNK, PAD 등의 토큰은 어떻게 처리해주어야 할까?

# Todo
* 데이터 자체를 정제 후 정확도 비교
* CNN을 이용한 text classification
* train data를 길이로 sorting 한 후 batch마다 가장 긴 문장 기준으로 padding을 채우기 (padding이 상대적으로 적게 들어가서 학습의 효율을 높일 수 있음)  

# Reference
[밑바닥부터 이해하는 어텐션 매터니즘](https://glee1228.tistory.com/3)  
[케라스 어텐션 매커니즘](https://yjam.tistory.com/73?category=1080798)  
[어텐션과 트랜스포머](https://huidea.tistory.com/150)  
[딥러닝을 이용한 자연어 처리 입문](https://wikidocs.net/103802)