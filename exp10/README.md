# News Summerization  
 뉴스 기사를 입력했을 때, 기사를 요약하여 헤드라인을 가져오자!  

 # Abstractive vs Extractive  
 ## Abstractive Summarization  
 추상적 요약으로 원문으로 부터 내용이 요약된 **새로운 문장**을 생성한다.  
 즉, 원문에 없던 문장일 수도 있으며, 이는 `NLG(자연어생성)`라고 볼 수 있다.  

 ## Extractive Summarization  
 추출적 요약으로 원문에서 **문장들을 추출** 해서 요약한다.  
 원문에서 중요한 단어들로 요약을 할 경우, 문장들 간의 호응이 자연스롭지 않을 수 있다.  
 이러한 방식은 딥러닝 보다 주로 전통적인 머신러닝 방식인 Text Rank 와 같은 알고리즘을 사용한다.  
원문으르 구성하는 문장 중 어느 것이 요약문에 들어갈 핵심문장인지 판별한다는 점에서 `Text Classification` 문제로 볼 수 있다.  