# Stock Prediction  
주가 데이터를 이용하여 내일의 주가를 예측해보자

[📈Go To Code📉]()

# Time-Series 
시계열이란 시간 순서대로 발생한 데이터의 수열로 **Stationary(안정적)** 데이터에 대해서만 예측이 가능하다.  

# Stationary  
안정적이라는 것은 시계열 데이터의 통계적 특성이 변하지 않는 것으로 시간의 변화에 무관하게 일정한 프로세스가 존재한다는 것이다.  

Stationary한 Time-Series data는 시간의 추이와 관계없이 다음 세가지 특성이 일정해야한다.  

* mean (평균)
* variance (분산)
* autocovariance (자기공분산) (X(t)와 X(t+h)와의 공분산)  

위 특성들이 t와 무관하게 h에 대해서만 달라지는 일정한 상관도를 가진다.  

# Statistic Method to Ceck Stationary
## Augmented Dickey-Fuller Test (ADF Test)
본 시계열 데이터가 안정적이지 않다고 가정(귀무가설)한 후 가설이 틀렸음을 증명해 안정적임(대립가설)을 보이는 귀납적 방법이다.   

*Cf. p-value (유의확률)*  
귀무가설을 가정했을 때 확률분포 상에서 현재의 관측보다 더 극단적인 관측이 나올 확률로 귀무가설의 가정이 틀렸다고 볼 수 있는 확률.  
이 값이 0.05 미만으로 매우 낮게 나온다면 p-value만큼의 오류 가능성하에 귀무가설을 기각하고 대립가설을 채택할 수 있는 근거가 된다.  

Statsmodels 패키지의 adfuller 메소드를 이용해 손쉽게 ADF Test를 시행할 수 있다. 
```python
from statsmodels.tsa.stattools import adfuller

def augmented_dickey_fuller_test(timeseries):
    # statsmodels 패키지에서 제공하는 adfuller 메소드를 호출합니다.
    dftest = adfuller(timeseries, autolag='AIC')  
    
    # adfuller 메소드가 리턴한 결과를 정리하여 출력합니다.
    print('Results of Dickey-Fuller Test:')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
```

# How to make Stationary Data
시계열을 안정적으로 만드는 방법은 크게 두가지가 있다. 

* 정성적인 분석을 통해 stationary한 특성을 가지도록 기존의 시계열 데이터를 가공/변형
* 시계열 분해

## 1. Stationary하게 가공  
### 로그함수 변환  (분산을 일정하게)
시간의 추이에 따라 분산이 점점 커지고 있을 경우 로그함수로 변환하는 것이 도움된다.

### 추세(Trend)상쇄  (moving average 제거)
추세(Trend)란 시간 추이에 따라 나타나는 평균값 변화.  
이러한 변화량을 제거해 주려면 Moving average (rolling mean)을 구해서 빼준다.  
```python
moving_avg = ts_log.rolling(window=12).mean() # moving average 구하기
ts_log_moving_avg = ts_log - moving_avg # 변화량 제거
```
- `window`는 평균을 구하는 구간의 크기.  
- window를 12로 했을경우 제일 처음 11개의 데이터는 moving average가 계산되지 않으므로 결측치가 발생하므로 제거해주어야 한다.  

### 계절성(Seasonality) 상쇄 (차분 Difference)
계절성이란 Trend에 잡히지 않지만 시계열 데이터 안에 포함된 **패턴이 파악되지 않은 주기적 변화**  
시계열을 한 스텝 앞으로 시프트한 시계열을 원래 시계열에 빼주는 Difference(차분)을 통해 계절성을 상쇄한다.
이렇게 될 경우 남은 값은 **이번 스텝에서 발생한 변화량**을 의미한다.  
데이터에 따라서 2차, 3차 Difference를 적용하기도 한다.  

## 2. Time series decomposition (시계열 분해)
statsmodels 라이브러리의 `seasonal_decompose`메소드를 통해 시계열 안에 존재하는 trend, seasonality를 직접 분리해 낼 수 있다. 
```python
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend # 추세(시간 추이에 따라 나타나는 평균값 변화 )
seasonal = decomposition.seasonal # 계절성(패턴이 파악되지 않은 주기적 변화)
residual = decomposition.resid # 원본(로그변환한) - 추세 - 계절성
```
**Trend + Seasonality + Residual = Original**

# ARIMA (Autoregressive Integrated Moving Average)
ARIMA = AR(Autoregressive) + I(Integrated) + MA(Moving Average)
ARIMA는 3가지 모델을 모두 한꺼번에 고려한다.  

예를 들어 주식값이 떨어졌을때 아래의 두가지 요소가 존재한다.  
* '오늘은 주식이 올라서 균형을 맞추겠지?'라는 AR 형태의 기대
* '어제 떨어졌으니 추세적으로 계속 떨어지지 않을까?'라는 MA 형태의 우려


## AR (자기회귀, Autoregressive)
![그래프](https://github.com/estela19/AIFFEL/blob/master/exp14/utils/AR_graph.png)
* 과거 값들에 대한 회귀로 미래 값을 예측하는 방법
* Y_t가 이전 p개의 데이터 Y_t-1, Y_t-2, ..., Y_t-p 의 가중합으로 수렴한다고 보는 모델
* AR은 **Residual**에 해당하는 부분을 모델링
* 가중치의 크기가 1보다 작은 Y_t-1, Y_t-2, ..., Y_t-p의 가중합으로 수렴하는 자기회귀 모델과 안정적 시계열은 통계적으로 동치
* 주식값이 항상 일정한 균형 수준을 유지할 것이라고 예측하는 관점이 주식 시계열을 AR로 모델링하는 관점
![수식](https://github.com/estela19/AIFFEL/blob/master/exp14/utils/AR_%EC%88%98%EC%8B%9D.png)

## MA (이동평균, Moving Average)
![그래프](https://github.com/estela19/AIFFEL/blob/master/exp14/utils/MA_graph.png)
* MA는 Y_t가 이전 q개의 예측오차값의 가중합으로 수렴한다고 보는 모델이다.
* MA는 **Trend**에 해당하는 부분을 모델링
* 예측오차값 e_t가 0보다 크면 모델 예측보다 관측값이 더 높다는 뜻이므로, 다음 Y_t 예측 시에는 예측치를 올려 잡는다.
* 주식값이 최근의 증감 패턴을 지속할 것이라고 보는 관점이 MA로 모델링하는 관점
![수식](https://github.com/estela19/AIFFEL/blob/master/exp14/utils/MA_%EC%88%98%EC%8B%9D.png)

## I (차분누적, Integration)
* I는 Y_t가 이전 데이터와 d차 차분의 누적합으로 보는 모델
* I는 시계열의 **Seasonality**에 해당하는 부분을 모델링
예를들어 d=1이라면 Y_t는 Y_t-1과 dY_t-1의 합으로 보는 것이다.

# ARIMA 모델의 parameter p, q, d
ARIMA의 파라미터는 3가지가 있다.
* p: 자기회귀 모형(AR)의 시차
* q: 이동평균 모형(MA)의 시차
* d: 차분누적(I)의 횟수

많은 시계열데이터가 AR이나 MA중 하나의 경향만 가지므로 `p + q < 2`, `p * q = 0` 인 값을 사용하는데 p와 q 중 하나는 0이라는 의미  

d는 d차 차분의 시계열이 안정된 상태일때의 d를 선택한다.  

ARIMA의 모수를 선택하는 방법은 대표적으로 **ACF**와 **PACF**가 있다.

**ACF를 통해 MA 모델의 시차 q를 결정하고, PACF를 통해 AR 모델의 시차 p를 결정할 수 있다**

![표](https://github.com/estela19/AIFFEL/blob/master/exp14/utils/ACF_PACF.png)

## ACF
* 시차(lag)에 따른 관측치들 사이의 관련성을 측정하는 함수
* 주어진 시계열의 현재 값이 과거 값과 어떻게 상관되는지 설명함.
* ACF plot에서 X 축은 상관 계수를 나타냄, y축은 시차 수를 나타냄

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(ts_log)   # ACF : Autocorrelation 그래프 그리기
```

## PACF
* 다른 관측치의 영향력을 배제하고 두 시차의 관측치 간 관련성을 측정하는 함수
* k 이외의 모든 시차를 갖는 관측치의 영향력을 배제한 가운데 특정 두 관측치 y_t와 y_t-1가 얼마나 관련이 있는지 나타내는 척도

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_pacf(ts_log)  # PACF : Partial Autocorrelation 그래프 그리기
```

*cf. 본 프로젝트에서의 p, q, d*
PACF 그래프에서 p >= 2 에서 PACF가 0에 가깝기 때문에 p = 1  
(PACF가 0이라는 의미는 현재 데이터와 p시점 떨어진 이전의 데이터는 상관도가 0으로 고려할 필요 X)  
ACF는 점차적으로 감소하여 AR(1)과 유사하고 q에 대해 적합한 값이 없음    
MA를 고려할 필요가 없으므로 q=0  
1차 차분의 p-value값이 낮으므로 d=1    


# Reference
[유의확률](https://ko.wikipedia.org/wiki/%EC%9C%A0%EC%9D%98_%ED%99%95%EB%A5%A0)  
[P-value](https://m.blog.naver.com/baedical/10109291879)  
[Unit Root Test로 ADF Test 알아보기](https://www.machinelearningplus.com/time-series/augmented-dickey-fuller-test/)  
[자기상관함수 및 편자기상관함수](http://kanggc.iptime.org/em/chap9/chap9.pdf)  