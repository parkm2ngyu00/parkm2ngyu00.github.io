---
title: "[NLP] 서브워드 토크나이저(Subword Tokenizer) - BPE"
category: NLP
tags:
    - NLP
    - BPE
toc: true
toc_sticky: true
---

## 서브워드 토크나이저
기계에게 아무리 많은 단어를 학습시켜도, 세상의 모든 단어를 알려줄 수는 없다. 만약, 기계가 모르는 단어가 등장하면 그 단어를 집합에 없는 단어란 의미에서 **OOV(Out-Of-Vocabulary)**라고 표현한다. 기계가 문제를 풀 때, 모르는 단어가 등장하면 문제를 푸는 것이 까다로워 진다. 이와 같이 모르는 단어로 인해 문제를 푸는 것이 까다로워지는 상황을 OOV 문제라고 한다.   
이러한 상황을 완화시키기 위해 하나의 단어를 더 작은 단위의 의미있는 여러 서브워드들 (Ex) birthday = birth + day)의 조합으로 구성된 경우가 많기 때문에, 하나의 단어를 여러 서브워드로 분리해서 단어를 인코딩 및 임베딩하겠다는 의도를 가진 전처리 작업이다. 이를 통해 OOV 문제나 희귀 단어, 신조어와 같은 문제를 완화시킬 수 있다.    

### 1. 바이트 페어 인코딩(Byte Pair Encoding, BPE)
***
아래와 같은 문자열이 주어졌을 때 BPE를 수행한다고 해보자.
```text
aaabdaaabac
```
BPE는 기본적으로 연속적으로 가장 많이 등장한 글자의 쌍을 찾아서 하나의 글자로 병합하는 방식을 수행한다. 예를 들어 위의 문자열 중 가장 자주 등장하고 있는 글자의 쌍은 'aa'이다. 이 'aa'를 'Z'로 치환해보자.
```text
ZabdZabac
Z=aa
```
위 문자열 중에서 가장 많이 등장하고 있는 바이트의 쌍은 'ab'이다. 이 'ab'를 'Y'로 치환해보자.
```
ZYdZYac
Y=ab
Z=aa
```
가장 많이 등장하고 있는 바이트의 쌍은 'ZY'이다. 이를 'X'로 치환해보자.
```
XdXac
X=ZY
Y=ab
Z=aa
```
더 이상 병합할 바이트의 쌍은 없으므로 BPE는 위의 결과를 최종 결과로 하여 종료된다.   

### 2. 자연어 처리에서의 BPE
***
자연어 처리에서의 BPE는 서브워드 분리(subword segmentation) 알고리즘이다. 기존에 있던 단어를 분리한다는 의미이다.
#### 1) 기존의 접근   
어떤 훈련 데이터로부터 각 단어들의 빈도수를 카운트했다고 해보자.
```python
# 훈련 데이터에 있는 단어와 등장 빈도수
low : 5, lower : 2, newest : 6, widest : 3
```
이 훈련 데이터에는 'low'란 단어가 5회 등장하였고, 'lower'란 단어는 2회 등장하였으며, 'newest'란 단어는 6회, 'widest'란 단어는 3회 등장하였다는 의미이다. 그렇다면 딕셔너리로부터 이 훈련 데이터의 단어 집합(vocabulary)을 얻는 것은 간단하다.
```python
# vocabulary
low, lower, newest, widest
```
그렇다면 테스트 과정에서 새로운 단어인 'lowest'란 단어가 등장한다면 기계는 이 단어를 학습한 적이 없으므로 OOV 문제가 발생한다.   
#### 2) BPE 알고리즘을 사용한 경우
***
위의 딕셔너리에 BPE를 적용해보자. 우선 딕셔너리의 모든 단어들을 글자(chracter) 단위로 분리한다. 이 경우 딕셔너리는 아래와 같다.
```python
# dictionary
l o w : 5,  l o w e r : 2,  n e w e s t : 6,  w i d e s t : 3
```
딕셔너리를 참고로 한 초기 단어 집합(vocabulary)을 아래와 같다. 간단히 말해 **초기 구성은 글자 단위로 분리된 상태이다**.
```python
# vocabulary
l, o, w, e, r, n, s, t, i, d
```
1회 - 위의 딕셔너리에서 빈도수가 가장 높은 (e, s)의 쌍을 es로 통합한다
```python
# dictionary update!
l o w : 5,
l o w e r : 2,
n e w es t : 6,
w i d es t : 3
```
```python
# vocabulary update!
l, o, w, e, r, n, s, t, i, d, es
```
2회 - 빈도수가 9로 가장 높은 (es, t)의 쌍을 est로 통합한다.
```python
# dictionary update!
l o w : 5,
l o w e r : 2,
n e w est : 6,
w i d est : 3
```
```python
# vocabulary update!
l, o, w, e, r, n, s, t, i, d, es, est
```
3회 - 빈도수가 7로 가장 높은 (l, o)를 lo로 통합한다.
```python
# dictionary update!
lo w : 5,
lo w e r : 2,
n e w est : 6,
w i d est : 3
```
```python
# vocabulary update!
l, o, w, e, r, n, s, t, i, d, es, est, lo
```
이와 같은 방식으로 반복하였을 때 얻은 딕셔너리와 단어 집합은 아래와 같다.
```python
# dictionary update!
low : 5,
low e r : 2,
newest : 6,
widest : 3
```
```python
# vocabulary update!
l, o, w, e, r, n, s, t, i, d, es, est, lo, low, ne, new, newest, wi, wid, widest
```
이 경우 테스트 과정에서 'lowest'란 단어가 등장한다면, 기존에는 OOV에 해당되는 단어가 되었겠지만 BPE 알고리즘을 사용한 위의 단어 집합에서는 더 이상 'lowest'는 OOV가 아니다. 기계는 우선 'lowest'를 전부 글자 단위로 분할한다. 즉, 'l, o, w, e, s, t'가 됩니다. 그리고 기계는 위의 단어 집합을 참고로 하여 'low'와 'est'를 찾아냅니다. 즉, 'lowest'를 기계는 'low'와 'est' 두 단어로 인코딩한다. 그리고 이 두 단어는 둘 다 단어 집합에 있는 단어이므로 OOV가 아니다.   

![image](https://github.com/parkm2ngyu00/BigleaderProject/assets/88785472/6272c5c7-f62a-4532-b7b7-3e93e0126074)

## 참고 서적
[딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/22592)  