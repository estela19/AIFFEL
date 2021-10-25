# Sentiment Classification  
네이버 영화 리뷰 데이터셋을 이용해 영화 리뷰를 보고 긍정적인 리뷰인지, 부정적인 리뷰인지 Binary Classification하는 모델을 만들어 보자!  

# Workflow  
### 1. DataLoader 구성  
### 2. 데이터 분석 및 가공  
### 3. Modeling  
### 4. Model Train  
### 5. Model Test  

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
| LSTM                  | 256          | 25              | 0.804    |
| Attention before LSTM | 256          | 25              | 0.8399   |
| Attention after LSTM  | 256          | 25              | 0.8377   |
| Transformer           | -            | 32              | 0.8439   |
| Self Attention        | 256          | 32              | 0.8504   |  

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
해당 라이브러리로 네이버 영화리뷰 test dataset을 사용했을때 **`0.9103`**으로 매우 높은 정확도를 보여주었다.   

