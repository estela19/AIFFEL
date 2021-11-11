# Session based Recomendation  
user id와 movie id, Time (영화를 본 시점)이 주어질 때 어떤 유저가 다음에 볼 영화를 예측해보자  

[😎Go To Code😎](https://github.com/estela19/AIFFEL/blob/master/exp12/baseline.ipynb)

# Model
# GRU4REC  
## Layer  
* one-hot encodding
* GRU layer

![model](https://github.com/estela19/AIFFEL/blob/master/exp12/utility/model.png)

## Session-Parallel Mini-Batches
Session이 끝날 때까지 기다리지 않고 병렬적으로 계산  

### Problem
긴 세션과 짧은 세션이 섞여 있다.  
이 세션들을 하나의 샘플로 mini-batch를 구성하면 제일 긴 세션의 연산이 끝날 때가지 짧은 세션들이 기다려야 해 자원이 낭비된다. 

![session](https://github.com/estela19/AIFFEL/blob/master/exp12/utility/session.png)

위 그림에서 세선 1, 2, 3, 을 하나의 미니배치로 만든다면 세션 3이 끝날때 까지 기다린다. 
이렇게 구성할 경우 미니배치의 shape은 (3,1,1)이고 RNN cell state가 1개로만 이루어진다.  
또한 tensorflow기준으로 rnn의 stateful=True 옵션을 사용하고 세션이 끝나면 state를 0으로 만들어준다. 

![ssesion-parallel](https://github.com/estela19/AIFFEL/blob/master/exp12/utility/session-parallel.png)

### Salpling on the eoutput
Negative Sampling과 같은 개념.
Item 수가 많기에 Loss를 계산할때 모든 아이템과 비교하지 않고 인기도를 고려하여 샘플링.  
(본 코드에서는 구현하지 않음)

### Ranking Loss  
본 task를 여러 아이템 중 다음 아이템이 무엇인지 classification하는 task로 생각할 수 있으나  
여러 아이템을 관련도 순으로 랭킹을 매겨 높은 랭킹의 아이템을 추천하는 task로 생각할 수도 있다.  
따라서 Ranking을 맞추는 objective function에 대한 연구가 있었고 이를 고려한 Loss를 사용하였다.  
(하지만 본 코드에서는 간소화를 위해 Cross-Entropy Loss를 사용했다)

# Reference  
[Session-based recommendation with recurrentt neural networks](https://arxiv.org/pdf/1511.06939v4.pdf)  