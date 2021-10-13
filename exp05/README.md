# Face similarity  
사람 얼굴을 인식에 나와 닮은 사람을 찾아보자!  
[Go to Code!](https://github.com/estela19/AIFFEL/blob/master/exp05/%5BE-05%5Dface_similarity.ipynb)  

## Face Embedding Module
### 1. Face Detection
사람의 얼굴을 인식한 후 얼굴 영역만 가져온다.  

### 2. Face Embedding  
face_recoginition 모듈에서 제공하는 face_encodings 함수로 해당 얼굴의 임베딩 벡터를 구한다. 
```angular2html
import face_recognition

face_recoginition.face_encodings(image)
```

### 3. Calculate Distance
각각의 얼굴 임베딩 벡터사이의 거리를 구해 유사도를 측정한다.  
본 프로젝트에서는 `get_distance` 함수로 구현되어 있으며 세부 구현체는 다음과 같다.  
```angular2html
def get_distance(name1, name2):
    return np.linalg.norm(embedding_dict[name1]-embedding_dict[name2], ord=1)
```

### 4. Get nearest Face  
계산한 거리를 sorting 하여 가장 유사한 얼굴을 구한다.   

## Visualization  
타겟사진을 초록색, 가장가까운 상위 5개사진을 빨간색, 나머지 사진을 갈색으로 표현하였다.  
원래는 해당 사진의 이름을 띄우려고 했으나, 한국어 encoding이 깨지는 문제가 발생했다.  

## Another Way
### 1. Average Hash  
이미지를 해시로 변환한 후 각 해시사이의 유사도를 구하는 [Average Hash](https://codegongbang.tistory.com/15) 알고리즘  

### 2. Auto Encoder  
인코더와 디코더를 딥러닝으로 학습시켜 Face Embedding vector를 구할 수 있다.
