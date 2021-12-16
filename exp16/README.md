# Super Resolution  
bicubic interpolation과 srgan을 활용하여 저해상도 이미지를 고해상도로 변경해보자  

[Go To Code]()

**super resolution**이란 저해상도 이미지를 고해상도로 변환하는 작업을 말한다.  
super resolution에는 두가지 어려움이 잇는데, 
* 변환 시 하나의 저해상도 이미지에 대해 여려개의 고해상도 이미지가 나올 수 있다.
* 2x2 크기의 이미지로 5x5의 이미지를 생성해야 할 경우 21개의 정보를 생성하는 등 제한된 정보로 많은 정보를 만들어 내야한다.   

## Interpolation  
보간을 이용한 방법으로 각 픽셀의 거리와 값에 따라 해당 값을 채우는 방식이다.  
bilinear과 bicubic등이 있으며 opencv 라이브러리의 `resize()`의 `interpolation` 설정에 따라 조절할 수 있다.

```python
bicubic_image = cv2.resize(
    lr_image, 
    dsize=(451, 300), # (가로 픽셀 수, 세로 픽셀 수)
    interpolation=cv2.INTER_CUBIC # bicubic interpolation 적용
)
```

## SRCNN (Super Resolution Convolutional Neural Networks)
저해상도 이미지를 bicubic interpolation하여 원하는 크기로 늘린 후 입력한다.
3개의 conv layer를 거쳐 고해상도 이미지를 생성하고, 실제 고해상도 이미지와의 차이를 역전파하여 가중치를 학습한다.  

![SRCNN]()

SRCNN은 다음 3가지 연산을 수행한다. 
* Patch extraction and representation : 저해상도 이미지에서 patch 추출  
* Non-linear mapping : patch를 다른 차원의 patch로 비선형 매핑
* Reconstrution : Patch로부터 고해상도 이미지를 생성  

SRCNN 이후 제안된 구조들은 다음과 같다.  

### VDSR (Very Deep Super Resolution)

![VDSR]()

* SRCNN과 동일하게 interpolation을 통해 저해상도 이미지의 크기를 늘려 입력  
* 20개의 convolutional layer 사용
* 최종 이미지 생성 전에 residual learning을 이용

### RDN (Residual Dense Network)

![RDN]()

* 저해상도 이미지가 입력되면 거치는 convolution layer의 출력을 최대한 활용

### RCAN (Residual Channel Attention Networks)

![RCAN]()

* convolution layer의 결과인 feature map에서 채널 간 모든 정보가 균일한 중요도를 갖는 것이 아닌 일부 중요한 채널에만 선택적으로 집중하도록 유도 (Attention)

## SRGAN (Super Resolution GAN)

![SRGAN]()

Generator가 저해상도 이미지를 입력받아 고해상도 이미지를 생성하면,  Discriminator는 생성된 이미지와 실제 이미지 중 진짜를 판별한다. 

### SRGAN Loss
![SRGAN Loss]()

content loss와 adversarial loss로 구성되어 있다. 
SRGAN은 content loss 와 adversarial loss를 합하여 perceptual loss로 정의해 학습에 사용
**adversirial loss**
일반적으로 알고 있는 GAN의 loss

**content loss**
![content loss]()

generator가 생성한 이미지와 실제 이미지를 비교하는 것이 아닌 **pretrained vgg** 모델에 입력하여 나오는 feature map의 차이를 계산  

## PNSR & SSIM 
Super Resolution 결과를 평가하는 정량적인 척도

* PNSR
	* 영상 내 신호가 가질 수 있는 최대 신호에 대한 잡음 비
	* PNSR 수치가 높을 수록 원본영상에 비해 손실이 적음

* SSIM
	* 영상의 구조 정보를 고려하여 얼마나 구조 정보를 변화시키지 않았는지  
	* SSIM값이 높을수록 원본 영상의 품질에 가깝다. 


# Additional Study
* RefSR (Reference-based Super Resolution)
* [EDSR](https://www.youtube.com/watch?v=OMIqkn2DCUk)
* [Super Resolution 어디까지 왔니](https://www.youtube.com/watch?v=nvsYKSHw0jo)

# Reference
* [bilinear interpolation](https://blog.naver.com/dic1224/220882679460)  
* [interpolation](https://bskyvision.com/789)  
* [SRCNN](https://d-tail.tistory.com/6)
* [PNSR](https://bskyvision.com/392)
* [SSIM](https://bskyvision.com/396)