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
~ 방식을 취하기 때문에 사람이 완벽하게 detection 되지 않는 문제점이 있다. 예를 들면 손가락의 일부분 또는 경계가 깔끔하지 않게 잘리는 문제가 생긴다.  
(사진)  
  
## Solution  

