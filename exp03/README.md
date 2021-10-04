# Image composition  
요즘 카메라 앱들에서는 사람의 얼굴을 인식해 다양한 필터 및 스티커를 붙여주는 기능을 제공한다.  
exp03에서는 그런 카메라 스티커기능을 구현한다. 사진에서 인물을 인식하여 적당한 위치에 스티커를 붙여주는 프로젝트를 진행하였다.  
  
## Face Detection  
스티커를 붙이기 위해서는 붙일 대상 즉, 사람의 얼굴을 인식해야한다.  
본 프로젝트에서는 opencv와 dlib의 `get_frontal_face_detector()`를 이용하여 사람의 얼굴을 인식하였다.  
![face_detect](https://github.com/estela19/AIFFEL/blob/master/exp03/face_detect.JPG)  
  
스티커를 붙이기 위해서는 단순히 얼굴만 detection하는 것이 아니라 특정한 위치를 잡아낼 수 있어야 했다.  
본 프로젝트에서 붙이는 고양이 콧수염의 경우 코의 위치를 인식해 내야 했는데, 
이는 HOG방식으로 이미 학습되어있든 face_landmarks 모델을 가져와 사용했다.  
![face_feature](https://github.com/estela19/AIFFEL/blob/master/exp03/face_feature.JPG)


## How to use face_landmarks
first, download pre trained model
```
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2  
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2  
```  
second, get predictor  
```
model_path = "/content/shape_predictor_68_face_landmarks.dat"
landmark_predictor = dlib.shape_predictor(model_path)
```
