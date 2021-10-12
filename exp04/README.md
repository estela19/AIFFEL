# NLG (Natural Language Generation)  
첫 단어가 주어졌을 때 완결된 문장을 생성하는 자연어 생성 모델을 만들었다.  
  
# NLP/NLG Task Pipeline  
## 1. Preproccesing Pipeline  
## 1-1. Preproccesing Sentence  
* 특수문자 및 구두점 제거
* 띄어쓰기 정규화
* 모두 소문자로 치환
* 문장의 처음과 끝에 `<start>` `<end>` token 추가
* 빈 문자열 제거
* 특정 길이 이상의 문장 제거
  * 특정 길이 이상의 문장을 제거 하지 않으면 특별히 긴 몇개의 문장 때문에 대부분의 나머지 문장에 과도한 padding이 들어가게 되어 학습을 방해한다. 
  * 일반적으로는 특정길이 이상을 제거하기 보다 문장 길이로 sorting 한 다음 batch마다 sentence length를 다르게 줌으로써 해당 문제를 해결한다.  

## 1-2. Tokenize  
자연어의 단어들을 컴퓨터가 이해할 수 있도록 인덱싱 해주는 과정.  
크게 one-hot encoding과 int encoding이 있다.  


### one-hot encoding
전체 단어 크기의 array를 가지고 해당하는 단어의 위치에면 1로 마킹한다.  
array가 sparse 해지기 때문에 현재는 잘 쓰이지 않는다.  
``` 
# one hot encoding
apple  : [1, 0, 0, 0, 0]
window : [0, 0, 0, 1, 0]
      ...
```

### int encoding  
단어를 순차적으로 int값을 mapping한다.  
현재 가장 많이 쓰이는 방식이다.  
```angular2html
# int encoding  
apple  : 1
window : 2
        ...
```  

### Auto encoder  
인코딩과 디코딩 하는 과정을 딥러닝으로 처리하는 auto encoding도 존재한다.  
input을 차원축소하는 encoder를 지나 다시 decoder를 통과시켜 처음 input과 비교하는 방식으로 학습한다.  

## 1-3. Embedding  
encodding된 word vector들은 단지 숫자로 mapping 되어있을 뿐이다.  
단어를 embedding해 단어사이의 관계를 표현해준다.  
Word2Vec, GloVe, FastText 등이 있다  

## 2. Sequence Model  
이미지 처리와 다르게 문장은 각 단어의 순서가 있는 sequential data이다.  
이를 위해 sequential model에서는 전 단계의 output을 후 단계의 hidden state로 함께 연산한다.  
### RNN  
가장 초기 모델로 이전 layer의 state를 다음 layer의 input으로 준다.  
하지만 문장이 길어질 수록 앞 전의 state가 소실되는 문제점이 생긴다.  
![rnn](https://github.com/estela19/AIFFEL/blob/master/exp04/utils/rnn.JPG)

### LSTM  
RNN의 단점을 해결하고자 핵심정보를 다음 step에 잘 전달하고자 했다.  
4개의 가중치를 가지고 연산하는 4개의 레이어를 가지고 있다.  
![lstm](https://github.com/estela19/AIFFEL/blob/master/exp04/utils/lstm.JPG)

## 3. Detokenize
Tokenizer와 반대로 숫자 token을 읽어 자연어로 치환해주는 과정이다.  
[Tokenize & Detokenize](https://github.com/estela19/AIFFEL/blob/master/exp04/utils/detokenize.JPG)
