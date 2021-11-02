# Movie Recommendation  
영화 데이터와 유저 정보가 주어질 때 해당 정보를 가지고 다음에 볼 영화를 추천해주는 추천시스템을 만들어보자!  
[👉👉Go To Code👈👈](https://github.com/estela19/AIFFEL/blob/master/exp09/baseline.ipynb)  

# Recommend System Workflow  
## 1. 데이터 전처리  
유저 id와 무비 id 를 정수 인덱싱  
이미 movie id와 user id가 중복 없는 정수로 mapping 되어 있다.  

이번 프로젝트에서는 별점을 해당 영화를 본 횟수로 처리하였다.  

본 프로젝트에서는 내가 다음 볼 영화를 추천하므로 movie list에서 내가 관심있게 본 영화 5개를 matrix에 추가해 주었다.  

## 2. CSR matrix 만들기  
전체 movie id와 user id를 matrix의 row와 col으로 넣어준다.  (matrix의 shape는 num of user * num of movie)  
MF의 개선방안으로 sparse한 데이터를 표현한 csr matrix를 통해 효율적으로 연산 할 수 있게 해준다.  

## 3. 모델링
ALS (Alternating Least Squares) 모델을 사용하였다.  
ALS는 사용자와 아이템의 Latent Factor를 한번 씩 번갈아가면서 학습시킨다.  
아이템 행렬을 상수로 놓은 후 사용자 행렬을 학습시키고  
사용자 행렬을 상수로 놓고 아이템 행렬을 학습시키는 방식이다.  
이 방식을 반복하며 학습하며 최적의 사용자와 아이템 latent factor를 학습시킨다.  

## 4. 모델 학습 
 
 
## 5. 테스트  
als model 자체에서 제공하는 reccomend 메소드를 통해 영화를 추천해 줄 수 있다.  
해당 메소드는 다음과 같이 사용할 수 있다.  
```python
movie_recommended = als_model.recommend(user_id, csr_data, N=20, filter_already_liked_items=True)
```  

# Result
모델이 추천해준 영화는 다음과 같다.  
```
score: 0.4335787892341614, Toy Story 2 (1999)
score: 0.293207585811615, Bug's Life, A (1998)
score: 0.22639738023281097, Mask, The (1994)
score: 0.20469099283218384, Aladdin (1992)
score: 0.16231714189052582, Hook (1991)
score: 0.1596899926662445, Twelve Monkeys (1995)
score: 0.15715646743774414, Wrong Trousers, The (1993)
score: 0.15678977966308594, Close Shave, A (1995)
score: 0.15655234456062317, Babe (1995)
score: 0.15463928878307343, Santa Clause, The (1994)
score: 0.14413881301879883, Indian in the Cupboard, The (1995)
```   
toy story와 쥬만지 등의 movie를 추가해주었는데, toy story2를 추천해 준 것으로 보아 꽤나 성공적으로 추천했음을 알 수 있다.  