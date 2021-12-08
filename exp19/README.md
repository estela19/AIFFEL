# Pix2Pix Road Generation. 
pix2pix U-Net Generator를 통해 segmentation map을 실제 도로 이미지를 생성해보자!  

[🚙.. Go To Code 🚙..](https://github.com/estela19/AIFFEL/blob/master/exp19/baseline.ipynb)
 

# GAN
* Generator
	* 노이즈 z(파란색)가 입력되고 특정
	* representation(검정색)으로 변환된 후 가짜 데이터 G(z)(빨간색)를 생성

* Discriminator
	* 실제 데이터 x와 Generator가 생성한 가짜 데이터
	* G(z)를 각각 입력받아 D(x) 및 D(G(z)) (보라색)를 계산하여 진짜와 가짜를 식별
  

# cGAN (Conditional Generative Model)
일반적인 GAN과 달리 특정 조건 정보를 함께 입력한다  

* Generator
	* 노이즈 z(파란색)와 추가 정보 y(녹색)을 함께 입력받아 generator 내부에서 결합되어 representation(검정색)으로 변환되며 가짜 데이터 G(z|y)를 생성
	* MNIST나 CIFAR-10 등의 데이터셋에 대해 학습시키는 경우 y는 레이블 정보, 일반적으로 one-hot 벡터를 입력.

* Discriminator
	* 실제 데이터 x와 Generator가 생성한 가짜 데이터 G(z|y)를 생성한 가짜 데이터 G(z|y)를 각각 입력받으며, y정보가 각각 함께 입력되어 진짜와 가짜를 식별
	* mnist 등의 데이터 셋에 대해 학습시키는 경우 실제 데이터 x와 y는 알맞은 한쌍("7"이라 쓰인 이미지의 경우 레이블도 7)을 이루고 generator에 입력된 y와 discriminator에 입력되는 y는 동일한 레이블을 나타내야 함

# Pix2Pix
기존노이즈 입력을 이미지로 변환하는 일반적인 GAN이 아닌, **이미지를 입력으로 하여 원하는 다른 형태의 이미지로 변환시킬 수 있는 GAN모델**로 image-to-image translation의 기초  
cGAN이 fully-connected 레이어를 연속적으로 쌓아 만드는 것에 비해, pix2pix는 convolution 레이어를 활용

## Pix2Pix Generator
### Encoder-Decoder generator
이미지를 입력받아 크기가 같은 변환된 이미지를 출력하기 위해 Encoder-Decoder 구조를 사용.
인코더의 최종 출력은 그림 중간에 위치한 가장 작은 사각형으로 `bottleneck`이라고 불리며 입력 이미지의 중요한 특징만을 담고있다.  
![generator](https://github.com/estela19/AIFFEL/blob/master/exp19/utils/generator.png)

### U-Net generator
각 레이어마다 encoder와 decoder가 skipconnection으로 연결됨
decoder가 변환된 이미지를 더 잘 생성하도록 encoder로부터 더 많은 추가 정보를 이용하는 방법  
![U-Net generator](https://github.com/estela19/AIFFEL/blob/master/exp19/utils/U-Net.png)


#### Encoder-Decoder generator와 U-Net Generator 비교
![result](https://github.com/estela19/AIFFEL/blob/master/exp19/utils/result.png)

## Pix2Pix Discriminator
DCGAN의 discriminator는 이미지 하나를 입력받아 convolution레이어를 이용해 크기를 줄여나가 하나의 이미지에 대해 하나의 확률값을 출력.  

### PatchGAN
pix2pix는 입력이미지의 일부, receptive filed영역에 대해서 진짜/가짜를 판별하는 확률값을 도출한다. 이러한 방식으로 서로 다른 영역에 대해 여러개의 확률값을 계산하여 이 값을 평균하여 최종 출력  
일반적으로 이미지에서 거리가 먼 두 픽셀은 서로 연관성이 거의 없기 때문에 일부 영역에서 세부적으로 진짜/가짜를 판별하는 것이 사실적인 이미지를 만드는데 도움이 된다.  
![PatchGAN](https://github.com/estela19/AIFFEL/blob/master/exp19/utils/patchGAN.png)

## Pix2Pix Loss
L1 loss나 L2 loss만 이용할 경우 Generator가 이미지의 평균적인 손실만 줄이고자하여 결과 이미지가 흐릿해지는 경향이 있다.  
L1 loss와 GAN loss를 같이 사용하면 더욱 좋은 결과를 얻을 수 있다. (discriminator를 잘 속여야 하기 때문)  
![loss](https://github.com/estela19/AIFFEL/blob/master/exp19/utils/loss.png)

# Reference
[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)  
[U-Net 논문 리뷰](https://medium.com/@msmapark2/u-net-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-u-net-convolutional-networks-for-biomedical-image-segmentation-456d6901b28a)  