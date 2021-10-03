# Machine Learning example  
keras의 toy dataset인 digits, wine, breast cancer를 이용해 machine learning을 수행해보았다.  
  

# Model  
다양한 모델 중 어떤 모델이 가장 좋을지 고민하던 중, 여러 모델을 합쳐서 결과를 내는 ensemble 기법을 사용하면 보다 정확도를 높일 수 있지 않을까 생각했다.  
ensemble 모델은 아래 그림처럼 개별 모델들의 예측값을 하나의 모델의 입력값으로 받아 최종 예측값을 구한다.  
![ensemble](https://github.com/estela19/AIFFEL/blob/master/exp02/ensemble.JPG)  
개별모델은 svm, random forest, logistic regression 을 사용하였고, 최종모델은 LightGBM을 사용하였다.  
다만 지금 사용한 기본적인 stacking ensemble은 과적합등의 문제가 발생할 수 있다고 한다.  
따라서 보통 CV기반(KFold etc)의 stacking ensemble을 사용한다고 한다.   
  
# Result  
실험 결과 breast cancer에서는 예측대로 ensemble모델의 성능이 가장 좋았지만 digits와 wine에는 몇가지 의문점이 생겼다.  
### digits  
digits의 경우 random forest의 accuracy, svm의 accuracy가 0.99인 반면 ensemble model의 경우 accuracy가 0.97이었다.  
처음에는 overfitting인지 고민했지만 다른 두 데이터셋에 비해 데이터가 1797개로 많아 overfitting이 아닌 다른 문제인 것 같다.  
  
### wine  
wine의 경우 svm, random forest의 accuracy가 0.54로 낮은 반면 ensemble model은 accuracyrk **1.00** 이었다 (!)  
처음에는 train data로 test를 하였나 싶었지만 test data로 정상적으로 test했다.  
몇차례 반복해서 시도해보아도 accuracy가 1.00 이었다. 
본 결과로 ensemble모델은 각 모델의 accuracy가 높을 때 보다 낮을 때 보다 효과적인 동작을 하는 것 같다고 추론했다.  

  
   
     
     
## 참고자료 및 사진출처  
https://lsjsj92.tistory.com/558  
https://lsjsj92.tistory.com/548  
