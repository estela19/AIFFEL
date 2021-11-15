# Image Generation  
이미지생성 모델을 이용하여 CIFAR-10 이미지를 생성해보자
[✨Go To code✨]()

# GAN 
## Generator  
아무 의미없는 랜덤 노이즈로부터 신경망에서의 연산을 통해 이미지 형상의 벡터를 생성.  
새로운 이미지를 창조하는 역할  
판별자가 Fake Image에 대해 판별한 값, 즉 D(fake_image)값이 1에 가까워지는 것이 목표  

### Loss  
generator loss 는 fake output이 1에 가까워지기를 바라므로, tf.ones_lide와의 cross entropy값을 계산하면 된다.
즉, cross entropy(tf.ones_like(fake_output), fake_output) 값은 fake output이 (Real Image를 의미하는) 1에 가까울수록 작은 값을 가진다.  
```python
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
```

## Discriminator  
기존에 있던 진짜 이미지와 생성자가 만들어낸 이미지를 입력받아 각 이미지가 Real인지 Fake인지에 대한 판단 정도를 실수값으로 출력
Real Image 판별값인 D(real_image)는 1에, Fake Image 판별값인 D(fake_image)는 0에 가까워지는 것이 목표  

### Loss  
real output 값은 1에 가까워지기를, fake output 값은 0에 가까워지기를 바라기에 두가지 loss 값을 모두 계산한다. real output은 1로 채워진 벡터와 fake output은 0으로 채워진 벡터와 비교한다.
```python
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
```

### Accuracy  
real output과 fake output을 accuracy 1.0에 가깝게 정확하게 판단해 낸다면 생성자가 만드는 fake output과 real output이 차이가 많이 나서 판별자가 쉽게 판별해 낸다는 뜻이다. 따라서 real accuracy와 fake accuracy는 초반에 1.0에 가깝게 나오다가 서서히 낮아져 둘다 0.5에 가까워지는 것이 이상적이다. 

# Result
checkpoint를 통해서 이전 학습을 불러와 학습시켰다. 총 1523epoch를 학습시켰다.
![history]()  

![Image]()

# Reference  
[모델저장 및 복원](https://www.tensorflow.org/tutorials/keras/save_and_load)