# Human Segmentation  
human segmentation 기술을 활용하여 인물에만 초점이 맞고 배경은 outfocus된 인물사진을 만들어보자

# Shallow focus workflow  
### 1. Human Detection  
pretrained 된 DeepLabModel을 이용하여 human detection 한다.  
DeepLabModel에서 검출할 수 있는 대상은 다음과 같다.  
  
`'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'`  
  
  총 21개의 대상을 검출 할 수 있다.  
    
대상을 검출한 뒤 사람에 대한 위치정보를 마킹한 segmentation map을 가지고 있는다.  

### 2. Background Blur  
배경을 흐리게 만들기 위해 사진을 흐리게 처리한다. 
사진을 흐리게 만드는 방법은 cv2의 blur함수를 사용해서 쉽게 처리해 줄 수 있다.  
```python
import cv2 

cv2.blur(img, (13, 13))
```
이때 (13, 13)은 blurring kernel size를 의미한다.  
  
### 3. Image Segmentation  
`Human detection` 단계에서 만들어 놓은 segmentation map 을 이용해 흐리게 처리한 이미지에 원본 사람 이미지를 segmentation한다.  
이를 통해 사람을 제외한 배경은 흐리고, 사람은 선명한 shallow focus 사진을 만들 수 있다.  

# About DeepLabModel  
## What can we do?
detection 단계에서 대상을 다르게 함으로써 사람을 outfocus하는 이미지 뿐만 아니라 다양한 대상을 outfocus할 수 있다.  
검출한 대상을 원본 이미지에 segmentation하는 것이 아니라 다른 배경에 segmentation함으로써 chroma key (배경합성)을 할 수 있다.  

## Problem of DeepLabModel  
DeepLabModel에서는 객체의 경계를 완벽하게 detection 되지 않는 문제점이 있다. 예를 들면 손가락의 일부분 또는 경계가 깔끔하지 않게 잘리는 문제가 생긴다.  
![problem](https://github.com/estela19/AIFFEL/blob/master/exp08/utils/problem.JPG)
  
# Improvement about DeepLabModel  
## Watershed Algorithm  
이미지를 grayscale로 변환했을 때, pixel의 값을 통해서 이미지의 높고 낮음을 구별할 수 있다. (pixel 값이 급격하게 변하는 구간이 경계점임을 알 수 있다.)  이렇게 특정 지점부터 근처 픽셀값까지 확장하다 보면 두 경계가 만나는 부분이 생기게 되는데, 이 지점을 경계로 이미지 분할을 하게 된다.  
이렇게 분할을 하게 되면 `대상의 가장자리 부분이 잘려서 인식되는 것을 방지`할 수 있다.  

```python
import cv2

cv2.watershed(img, markers)
```
형태로 사용할 수 있다.  

![watershed](https://github.com/estela19/AIFFEL/blob/master/exp08/utils/watershed.JPG)

## Stereo Depth
같은 이미지를 왼쪽과 오른쪽 각각에서 촬영한 스테레오 이미지를 활용하여 깊이를 얻어 낼 수 있다. 이는 우리 눈에서 대상과의 거리와 원근감을 느끼게 되는 원리와 동일하다. 이미지의 깊이를 얻어 낼 수 있으므로 1차원 이미지에서 경계를 분리해 내는 것이 아닌 깊이를 기준으로 경계를 분리하여 보다 `정확하게 대상을 인식`할 수 있다.  
  
![streo](https://github.com/estela19/AIFFEL/blob/master/exp08/utils/stereo.JPG)

 `cv2` 모듈을 이용해 쉽게 stereo map을 가져올 수 있다.
```python
import cv2  

cvw.StereoBM_create()
```
위와 같은 형태로 사용한다.  

동일한 두 이미지에서 구해낸 stereomap의 예시이다.    
![stereomap](https://github.com/estela19/AIFFEL/blob/master/exp08/utils/stereomap.JPG)  

# Reference  
https://softwareeng.tistory.com/143  
http://www.gisdeveloper.co.kr/?p=6955
