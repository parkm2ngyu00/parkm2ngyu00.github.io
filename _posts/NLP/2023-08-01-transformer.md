---
title: "[NLP] 트랜스포머(Transformer) - 1"
category: NLP
tags:
    - NLP
    - Transformer
toc: true
toc_sticky: true
---

## 트랜스포머(Transformer)
트랜스포머(Transformer)는 2017년 구글이 발표한 논문인 "Attention is all you need"에서 나온 모델로 기존의 seq2seq의 구조인 인코더-디코더를 따르면서도, 논문의 이름처럼 어텐션(Attention)만으로 구현한 모델이다. 이 모델은 RNN을 사용하지 않고, 인코더-디코더 구조를 설계하였음에도 번역 성능에서도 RNN보다 우수한 성능을 보여주었다.

### 1. 기본의 seq2seq모델의 한계
***
트랜스포머에 대해서 배우기 전에 기존의 seq2seq를 상기해보자. 기존의 seq2seq 모델은 인코더-디코더 구조로 구성되어져 있다. 여기서 인코더는 입력 시퀀스를 하나의 벡터 표현으로 압축하고, 디코더는 이 벡터 표현을 통해서 출력 시퀀스를 만들어냈다. 하지만 이러한 구조는 인코더가 입력 시퀀스를 하나의 벡터로 압축하는 과정에서 입력 시퀀스의 정보가 일부 손실된다는 단점이 있었고, 이를 보정하기 위해 어텐션이 사용되었습니다. 그런데 어텐션을 RNN의 보정을 위한 용도로서 사용하는 것이 아니라 어텐션만으로 인코더와 디코더를 만들어보면 어떨까?   

### 2. 트랜스포머의 주요 하이퍼 파라미터
***
시작에 앞서 트랜스포머의 하이퍼파라미터를 정의한다. 각 하이퍼파라미터의 의미에 대해서는 뒤에서 설명하기로하고, 여기서는 트랜스포머에는 이러한 하이퍼파라미터가 존재한다는 정도로만 이해해보자. 아래에서 정의하는 수치는 트랜스포머를 제안한 논문에서 사용한 수치로 하이퍼파라미터는 사용자가 모델 설계시 임의로 변경할 수 있는 값들이다.   

$$d_{model} = 512$$   

트랜스포머의 인코더와 디코더에서의 정해진 입력과 출력의 크기를 의미한다. 임베딩 벡터의 차원 또한 $d_{model}$이며, 각 인코더와 디코더가 다음 층의 인코더와 디코더로 값을 보낼 때에도 이 차원을 유지한다.   

$$\text{num_layers} = 6$$   

트랜스포머에서 하나의 인코더와 디코더를 층으로 생각하였을 때, 트랜스포머 모델에서 인코더와 디코더가 총 몇 층으로 구성되었는지를 의미한다.   

$$\text{num_heads} = 8$$   

트랜스포머에서는 어텐션을 사용할 때, 한 번 하는 것 보다 여러 개로 분할해서 병렬로 어텐션을 수행하고 결과값을 다시 하나로 합치는 방식을 택했다. 이때 이 병렬의 개수를 의미한다.   

$$d_{ff} = 2048$$   

트랜스포머 내부에는 피드 포워드 신경망이 존재하며 해당 신경망의 은닉층의 크기를 의미한다. 피드 포워드 신경망의 입력층과 출력층의 크기는 $d_{model}$이다.   

### 3. 트랜스포머(Transformer)
***
<img src="https://github.com/parkm2ngyu00/AlgoPractice/assets/88785472/1297a4f5-b598-43bd-8457-014dfde954c4" width=300 height=200/>   

트랜스포머는 RNN을 사용하지 않지만 기존의 seq2seq처럼 인코더에서 입력 시퀀스를 입력받고, 디코더에서 출력 시퀀스를 출력하는 인코더-디코더 구조를 유지하고 있다. 이전 seq2seq 구조에서는 인코더와 디코더에서 각각 하나의 RNN이 t개의 시점(time step)을 가지는 구조였다면 이번에는 인코더와 디코더라는 단위가 N개로 구성되는 구조이다. 트랜스포머를 제안한 논문에서는 인코더와 디코더의 개수를 각각 6개 사용했다.   

<img src="https://github.com/parkm2ngyu00/AlgoPractice/assets/88785472/7d883421-6027-45aa-9507-883fd85b7c37" width=300 height=500/>   

위의 그림은 인코더와 디코더가 6개씩 존재하는 트랜스포머의 구조를 보여준다. 이 포스트에서는 인코더와 디코더가 각각 여러 개 쌓여있다는 의미를 사용할 때는 알파벳 s를 뒤에 붙여 encoders, decoders라고 표현한다.   

<img src="https://github.com/parkm2ngyu00/AlgoPractice/assets/88785472/339f9d74-034d-4c9f-9ba0-770a121158df" width=600 height=400/>   

위의 그림은 인코더로부터 정보를 전달받아 디코더가 출력 결과를 만들어내는 트랜스포머 구조를 보여준다. 디코더는 마치 기존의 seq2seq 구조처럼 시작 심볼 <sos>를 입력으로 받아 종료 심볼 <eos>가 나올 때까지 연산을 진행한다. 이는 RNN은 사용되지 않지만 여전히 인코더-디코더의 구조는 유지되고 있음을 보여준다.

트랜스포머의 내부 구조를 조금씩 확대해가는 방식으로 트랜스포머를 이해해보자. 우선 인코더와 디코더의 구조를 이해하기 전에 트랜스포머의 입력에 대해서 이해해야한다. 트랜스포머의 인코더와 디코더는 단순히 각 단어의 임베딩 벡터들을 입력받는 것이 아니라 임베딩 벡터에서 조정된 값을 입력받는데 이에 대해서 알아보기 위해 입력 부분을 확대해보자.   

### 4. 포지셔널 인코딩(Positional Encoding)
***
트랜스포머의 내부를 이해하기 전 우선 트랜스포머의 입력에 대해서 알아볼 필요가 있다. RNN이 자연어 처리에서 유용했던 이유는 단어의 위치에 따라 단어를 순차적으로 입력받아서 처리하는 RNN의 특성으로 인해 각 단어의 위치 정보(position information)를 가질 수 있다는 점에 있었다.   

하지만 트랜스포머는 단어 입력을 순차적으로 받는 방식이 아니므로 단어의 위치 정보를 다른 방식으로 알려줄 필요가 있다. 트랜스포머는 단어의 위치 정보를 얻기 위해서 각 단어의 임베딩 벡터에 위치 정보들을 더하여 모델의 입력으로 사용하는데, 이를 포지셔널 인코딩(positional encoding)이라고 한다.   

<img src="https://github.com/parkm2ngyu00/AlgoPractice/assets/88785472/8811de02-0f30-401f-bdd3-9ac3eb6b0cb3" width=600 height=400/>   

위의 그림은 입력으로 사용되는 임베딩 벡터들이 트랜스포머의 입력으로 사용되기 전에 포지셔널 인코딩의 값이 더해지는 것을 보여준다. 임베딩 벡터가 인코더의 입력으로 사용되기 전 포지셔널 인코딩값이 더해지는 과정을 시각화하면 아래와 같다.   

<img src="https://github.com/parkm2ngyu00/AlgoPractice/assets/88785472/80d96440-e792-42a8-a169-5cbc2e5659cc" width=500 height=300/>   

포지셔널 인코딩 값들은 어떤 값이기에 위치 정보를 반영해줄 수 있는 것일까? 트랜스포머는 위치 정보를 가진 값을 만들기 위해서 아래의 두 개의 함수를 사용한다.   

$$PE_{(pos,\ 2i)}=sin(pos/10000^{2i/d_{model}})$$   

$$PE_{(pos,\ 2i+1)}=cos(pos/10000^{2i/d_{model}})$$   

사인 함수와 코사인 함수의 그래프를 상기해보면 요동치는 값의 형태를 생각해볼 수 있는데, 트랜스포머는 사인 함수와 코사인 함수의 값을 임베딩 벡터에 더해주므로서 단어의 순서 정보를 더하여 준다. 그런데 위의 두 함수에는 $pos, i, d_{model}$등의 생소한 변수들이 있다. 위의 함수를 이해하기 위해서는 위에서 본 임베딩 벡터와 포지셔널 인코딩의 덧셈은 사실 임베딩 벡터가 모여 만들어진 문장 행렬과 포지셔널 인코딩 행렬의 덧셈 연산을 통해 이루어진다는 점을 이해해야 한다.   

<img src="https://github.com/parkm2ngyu00/AlgoPractice/assets/88785472/1fb4bdb0-910d-4975-8f2c-8f0f39ea3bf3" width=350 height=200/>   

$pos$는 입력 문장에서의 임베딩 벡터의 위치를 나타내며, $i$는 임베딩 벡터 내의 차원의 인덱스를 의미한다. 위의 식에 따르면 임베딩 벡터 내의 각 차원의 인덱스가 짝수인 경우에는 사인 함수의 값을 사용하고 홀수인 경우에는 코사인 함수의 값을 사용한다. 위의 수식에서 $(pos, 2i)$일 때는 사인 함수를 사용하고, $(pos, 2i+1)$일 때는 코사인 함수를 사용하고 있다.   

또한 위의 식에서 $d_{model}$은 트랜스포머의 모든 층의 출력 차원을 의미하는 트랜스포머의 하이퍼파라미터이다.    

위와 같은 포지셔널 인코딩 방법을 사용하면 순서 정보가 보존되는데, 예를 들어 각 임베딩 벡터에 포지셔널 인코딩의 값을 더하면 같은 단어라고 하더라도 문장 내의 위치에 따라서 트랜스포머의 입력으로 들어가는 임베딩 벡터의 값이 달라진다. 이에 따라 트랜스포머의 입력은 순서 정보가 고려된 임베딩 벡터가 된다.   

### 5. 어텐션(Attention)
***
트랜스포머에서 사용되는 세 가지의 어텐션에 대해서 간단히 정리해보자. 이 포스트에선 큰 그림을 이해하는 것에만 집중한다.   

<img src="https://github.com/parkm2ngyu00/AlgoPractice/assets/88785472/c55e2cc4-634d-4bc9-b6d6-45e8af5f4e87" width=230 height=380/>   

첫번째 그림인 셀프 어텐션은 인코더에서 이루어지지만, 두번째 그림인 셀프 어텐션과 세번째 그림인 인코더-디코더 어텐션은 디코더에서 이루어진다. 셀프 어텐션은 본질적으로 Query, Key, Value가 동일한 경우를 말한다. 반면, 세번째 그림 인코더-디코더 어텐션에서는 Query가 디코더의 벡터인 반면에 Key와 Value가 인코더의 벡터이므로 셀프 어텐션이라고 부르지 않는다.   
- 주의할 점은 여기서 Query, Key등이 같다는 것은 벡터의 값이 같다는 것이 아니라 벡터의 출처가 같다는 의미이다.   

정리하면 다음과 같다.
```text
인코더의 셀프 어텐션 : Query = Key = Value
디코더의 마스크드 셀프 어텐션 : Query = Key = Value
디코더의 인코더-디코더 어텐션 : Query : 디코더 벡터 / Key = Value : 인코더 벡터
```

<img src="https://github.com/parkm2ngyu00/AlgoPractice/assets/88785472/da60659a-602b-4758-87c7-59fda09e1c5c" width=500 height=450/>   

위 그림은 트랜스포머의 아키텍처에서 세 가지 어텐션이 각각 어디에서 이루어지는지를 보여준다. 세 개의 어텐션에 추가적으로 '멀티 헤드'라는 이름이 붙어있다. 트랜스포머가 어텐션을 병렬적으로 수행하는 방법을 의미한다.   

이번 포스트에서는 seq2seq의 단점을 극복한 트랜스포머 모델의 전체적인 큰 그림에 대해 알아보았다. 사실 트랜스포머의 구조는 세세히 들어가보면 매우 복잡한데, 지금 당장 공부하는 단계에서는 큰 그림만 알고 넘어가도 무리는 없을 것 같다. 혹시라도 트랜스포머 모델에 대해 자세히 공부하고자 하는 사람은 [이 링크](https://wikidocs.net/31379)의 내용을 참고하길 바란다.   

## 참고 서적
[딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/book/2155) 