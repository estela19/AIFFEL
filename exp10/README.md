# News Summerization  
 뉴스 기사를 입력했을 때, 기사를 요약하여 헤드라인을 가져오자!  

 # Abstractive vs Extractive  
 ## Abstractive Summarization  
 추상적 요약으로 원문으로 부터 내용이 요약된 **새로운 문장**을 생성한다.  
 즉, 원문에 없던 문장일 수도 있으며, 이는 `NLG(자연어생성)`라고 볼 수 있다.  

[👉👉Go To Code👈👈](https://github.com/estela19/AIFFEL/blob/master/exp10/abstractive.ipynb)

 본 프로젝트에서 추상적 요약을 수행한 결과는 다음과 같다. 
 ```
 원문 : actress priyanka chopra production little visitors screened ongoing toronto international film festival earlier film first look unveiled cannes film festival actress also took part fest new segment share journey discussed career work outside entertainment industry 
실제 요약 : priyanka production screened at fest 
예측 요약 :  priyanka chopra to perform at cannes festival festival


원문 : sardar local bjp leader west bengal died friday night allegedly attacked goons sharp weapon kolkata mandir bazar returning home west bengal bjp accused trinamool congress incident tweeted continuation panchayat poll violence process panchayat board begun 
실제 요약 : bjp leader dies after allegedly being attacked by 
예측 요약 :  bjp leader dies in stone pelting near bengal


원문 : year old australian boy admitted hacking world valuable company apple servers internal files fan apple wanted work one day stored gb hacked data titled hack hack boy pleaded guilty return court sentencing september 
실제 요약 : year old apple fan admits to hacking the company 
예측 요약 :  yr old apple apple apple iphone lawsuit
```
대체적으로 실제요약과 비교했을 때 문맥상 비슷하였다.  
다만, 문법적으로 말이 안되는 경우가 상당히 많았다.  

 ## Extractive Summarization  
 추출적 요약으로 원문에서 **문장들을 추출** 해서 요약한다.  
 원문에서 중요한 단어들로 요약을 할 경우, 문장들 간의 호응이 자연스롭지 않을 수 있다.  
 이러한 방식은 딥러닝 보다 주로 전통적인 머신러닝 방식인 Text Rank 와 같은 알고리즘을 사용한다.  
원문으르 구성하는 문장 중 어느 것이 요약문에 들어갈 핵심문장인지 판별한다는 점에서 `Text Classification` 문제로 볼 수 있다.  

[👉👉Go To Code👈👈](https://github.com/estela19/AIFFEL/blob/master/exp10/extractive.ipynb)

본 프로젝트에서 추출적 요약을 수행한 결과는 다음과 같다.  
```
원문 :  Twenty-seven-year-old Mohammed Mahuwala was arrested in Indore on Wednesday for allegedly cheating e-commerce giant Amazon of nearly Ã¢ÂÂ¹30 lakh. Mahuwala was a member of a gang who ordered costly gadgets from Amazon. "[They] used to get refund of the amount paid...by saying the parcel...was empty. In reality, these devices were taken out from parcel and sold...to local shopkeepers," said police.
실제요약 :  Man arrested for cheating Amazon of Ã¢ÂÂ¹30 lakh by taking refunds
예측요악 :  Mahuwala was a member of a gang who ordered costly gadgets from Amazon.


원문 :  AgustaWestland chopper scam co-accused Rajiv Saxena was extradited to India from UAE on Wednesday. He had been evading the Enforcement Directorate's summons claiming he was suffering from leukaemia but had moved an anti-money laundering court for anticipatory bail in December, stating he had never been summoned at his Dubai address. Saxena's lawyers alleged he had been "illegally extradited". 
실제요약 :  AgustaWestland scam accused Rajiv Saxena extradited to India
예측요악 :  AgustaWestland chopper scam co-accused Rajiv Saxena was extradited to India from UAE on Wednesday.

원문 :  Gabrielle Reilly, a 22-year-old US woman, created earrings for her Apple AirPods and put it up for sale online for $20 (about Ã¢ÂÂ¹1,500). "I absolutely refuse to lose them (AirPods)...So I made earrings," she explained. The earrings, which took her over an hour to make, debuted in a video on Twitter that has since garnered over three million views.
실제요약 :  Woman turns Apple AirPods into earrings to avoid losing them
예측요악 :  "I absolutely refuse to lose them (AirPods)...So I made earrings," she explained.
``` 
대체적으로 문장이 문법적으로 맞는 편이지만 제대로 요약되지 않은 경우도 보인다. 

# Preprocessing  
다음과 같은 전처리를 해주었다.  
* 불용어(stop word)제거
	* 요약문(headline)의 경우 stop word가 의미있다고 생각하여 제거해 주지 않았다. 
* 축약 문장 정규화  
isn't -> is not, 'cause -> because, who's -> who is etc.  
* 텍스트 소문자화  
* html tag 제거  
* 괄호 문자열 제거  
my husband (and myslef) -> my husbasnd  
* 소유격 제거
* 영어 외 문자 공백으로 변환  
* m이 3개이상이면 2개로 변환  
ummmmmm yeah -> umm yeah  

```python
def preprocess_sentence(sentence, remove_stopwords=True):
    sentence = sentence.lower() # 텍스트 소문자화
    sentence = BeautifulSoup(sentence, "lxml").text # <br />, <a href = ...> 등의 html 태그 제거
    sentence = re.sub(r'\([^)]*\)', '', sentence) # 괄호로 닫힌 문자열 (...) 제거 Ex) my husband (and myself!) for => my husband for
    sentence = re.sub('"','', sentence) # 쌍따옴표 " 제거
    sentence = ' '.join([contractions[t] if t in contractions else t for t in sentence.split(" ")]) # 약어 정규화
    sentence = re.sub(r"'s\b","", sentence) # 소유격 제거. Ex) roland's -> roland
    sentence = re.sub("[^a-zA-Z]", " ", sentence) # 영어 외 문자(숫자, 특수문자 등) 공백으로 변환
    sentence = re.sub('[m]{2,}', 'mm', sentence) # m이 3개 이상이면 2개로 변경. Ex) ummmmmmm yeah -> umm yeah
    
    # 불용어 제거 (text)
    if remove_stopwords:
        tokens = ' '.join(word for word in sentence.split() if not word in stopwords.words('english') if len(word) > 1)
    # 불용어 미제거 (headlines)
    else:
        tokens = ' '.join(word for word in sentence.split() if len(word) > 1)
    return tokens
```

# Model  
본 프로젝트에서는 **Seq2Seq + Attention**을 사용하였다. 
seq2seq 만 사용할 경우 초기의 input이 반영되지 않는 문제점이 있기 때문에 Attention을 추가해주었다.   
## Seq2Seq  
seq2seq는 인코더에서 입력 시퀀스를 통해 context vector를 만들고 디코더에서 해당 context vector로 부터 출력 시퀀스를 생성해준다.  

### 배경  
RNN은 바로 이전 입력까지만 고려햐여 출력하기에 정확도가 떨어진다.  
그래서 전체 문장을 고려하기 위해 Seq2Seq모델이 등장하였다.  

### 장점  
* 단일 RNN과는 달리 Seq2Seq모델은 시퀀스 길이와 순서를 자유롭게 할 수 있다.  
* 전체 input을 고려해 context vector를 출력한다. (전체적인 맥락을 파악한다)

### 한계  
* 입력 데이터가 길어질 경우 gradient vanishing이 일어나 성능이 확연히 떨어진다.  
* context vector의 길이가 고정적이다.  
인코더는 3개의 LSTM layer로 구성되어 있으며 디코더는 1개의 LSTM layer로 구성되어있다.  

## Attention  
Attention을 통하여 디코더가 출력 시퀀스를 생성할때 input sequence의 어떤부분을 집중해야하는지 전달해준다.  

### 배경  
Seq2Seq에서 입력 문장이 길 경우 정확도가 떨어지는 문제가 생겼는데  
이러한 현상을 보정하기 위해 중요한 단어에 집중하여 Decoder에 전달하는 Attention 기법이 등장했다.  
Attention을 사용할 경우 기존 방식보다 훨씬 더 많은 데이터를 Decoder에 전달한다.  

### 적용방법  
1. Query를 key와 dot product하여 attention score를 구한다.  
2. attention score를 softmax해 집중해야할 위치를 구한다.  
3. 해당 score를 value에 concate 해준다.  

# Reference  
[seq2seq](https://blog.naver.com/sooftware/221784419691)  
[밑바닥부터 이해하는 어텐션 매커니즘](https://glee1228.tistory.com/3)  