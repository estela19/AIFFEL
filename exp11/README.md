# Pneumonia Classification  
폐렴 x-ray사진을 보고 폐렴인지 아닌지 진단을 내려보자!  
[👀See Code👀](https://github.com/estela19/AIFFEL/blob/master/exp11/baseline.ipynb)

# Model  
Convolution layer를 쌓은 CNN 모델을 이용하였다.  
모델의 구성은 다음과 같다.  

```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    
    tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
    tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPool2D(),
    
    conv_block(32),
    conv_block(64),
    
    conv_block(128),
    tf.keras.layers.Dropout(0.2),
    
    conv_block(256),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Flatten(),
    dense_block(512, 0.7),
    dense_block(128, 0.5),
    dense_block(64, 0.3),
    
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```
# Data Imbalance  
본 데이터에서는 정상인 라벨의 데이터 수가 폐렴환자의 데이터 수보다 훨씬 적었다.  
이러할 경우 모델의 정확도가 떨어질 수 있으므로 weight별로 다른 가중치를 적용하여 준다.  

```python
weight_for_0 = (1 / COUNT_NORMAL)*(TRAIN_IMG_COUNT)/2.0 
weight_for_1 = (1 / COUNT_PNEUMONIA)*(TRAIN_IMG_COUNT)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for NORMAL: {:.2f}'.format(weight_for_0))
print('Weight for PNEUMONIA: {:.2f}'.format(weight_for_1))
```

# Result  
9차 시도에서 accuracy 0.86을 달성하였다.  

| try | batch | epoch | dropout |    data augmentation    | accuracy |
|:---:|:-----:|:-----:|:-------:|:-----------------------:|:--------:|
|  1  |   16  |   25  |   0.2   |            X            |   0.839  |
|  2  |   64  |   10  |   0.4   | flip, change color(0.5) |   0.625  |
|  3  |   64  |   10  |   0.2   | flip, change color(0.5) |   0.625  |
|  4  |   32  |   10  |   0.2   | flip. change color(0.5) |   0.727  |
|  5  |   32  |   10  |   0.2   | flip, change color(0.3) |   0.75   |
|  6  |   16  |   10  |   0.2   | flip, change color(0.3) |   0.836  |
|  7  |   16  |   10  |   0.2   |            X            |   0.804  |
|  8  |   16  |   10  |   0.2   |            X            |   0.846  |
|  9  |   16  |   +1  |   0.2   |            X            | **0.86** |
|  10 |   16  |   15  |   0.2   |            X            |   0.66   |

data augmentation을 했을 경우 오히려 성능이 더 떨어짐을 알 수 있었다.  
