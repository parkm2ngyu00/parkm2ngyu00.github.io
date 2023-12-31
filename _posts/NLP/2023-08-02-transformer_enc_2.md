---
title: "[NLP] 트랜스포머 인코더 'FFNN', '잔차 연결', '층 정규화'"
category: NLP
tags:
    - NLP
    - Transformer
toc: true
toc_sticky: true
---

## 시작하며
이번 포스팅에선 트랜스포머의 인코더의 포지션-와이드 피드 포워드 신경망(Position-wise FFNN), 잔차 연결(Residual connection)과 층 정규화(Layer Normalization)에 대해서 공부한다. 아래 포스팅의 내용을 알고 있다는 가정 하에 작성되는 글이기 때문에, 해당 내용의 개념이 부족하다면 먼저 공부하고 보는 것을 추천한다.
> [어텐션 메커니즘 공부하러 가기](https://parkm2ngyu00.github.io/nlp/attention/)   
> [트랜스포머 기초 개념 공부하러 가기](https://parkm2ngyu00.github.io/nlp/transformer/)    
> [트랜스포머 인코더 ‘셀프 어텐션’ & ‘멀티 헤드 어텐션’ 공부하러 가기](https://parkm2ngyu00.github.io/nlp/transformer_enc_1/)   

## 포지션-와이즈 피드 포워드 신경망(Position-wise FFNN)
지금은 인코더를 설명하고 있지만, 포지션 와이즈 FFNN은 인코더와 디코더에서 공통적으로 가지고 있는 서브층이다. 포지션-와이즈 FFNN는 쉽게 말하면 완전 연결 FFNN(Fully-connected FFNN)이라고 해석할 수 있다. 앞서 인공 신경망은 결국 벡터와 행렬 연산으로 표현될 수 있음을 배웠다. 아래는 포지션 와이즈 FFNN의 수식을 보여준다.   

$$FFNN(x) = MAX(0, x{W_{1}} + b_{1}){W_2} + b_2$$   

식을 그림으로 표현하면 아래와 같다.   

![image](https://github.com/parkm2ngyu00/AlgoPractice/assets/88785472/d9f8aed5-457d-4e36-8a49-d6d525866d88)   

여기서 $x$는 멀티 헤드 어텐션의 결과로 나온 $(\text{seq_len},\ d_{model})$의 크기를 가지는 행렬을 말한다. 가중치 행렬 $W_1$은 $(d_{model},\ d_{ff})$의 크기를 가지고, 가중치 행렬 $W_2$는 $(d_{ff},\ d_{model})$의 크기를 가진다. 논문에서 은닝층의 크기인 $d_{ff}$는 2048의 크기를 가진다.   

여기서 매개변수 $W_1, b_1, W_2, b_2$는 하나의 인코더 층 내에서는 다른 문장, 다른 단어들마다 정확하게 동일하게 사용된다. 하지만 인코더 층마다는 다른 값을 가진다.   

![image](https://github.com/parkm2ngyu00/AlgoPractice/assets/88785472/a13d634a-1dc3-46bb-a9f4-040d19c0b968)   

위의 그림에서 좌측은 인코더의 입력을 벡터 단위로 봤을 때, 각 벡터들이 멀티 헤드 어텐션 층이라는 인코더 내 첫번째 서브 층을 지나 FFNN을 통과하는 것을 보여준다. 이는 두번째 서브층인 Position-wise FFNN을 의미합니다. 물론, 실제로는 그림의 우측과 같이 행렬로 연산되는데, 두번째 서브층을 지난 인코더의 최종 출력은 여전히 인코더의 입력의 크기였던 $(\text{seq_len},\ d_{model})$의 크기가 보존되고 있다. 하나의 인코더 층을 지난 이 행렬은 다음 인코더 층으로 전달되고, 다음 층에서도 동일한 인코더 연산이 반복된다.   

## 잔차 연결(Residual connection)과 층 정규화(Layer Normalization)   

![image](https://github.com/parkm2ngyu00/AlgoPractice/assets/88785472/12c0997f-8a37-4aa5-9dce-3f85f838b267)   

인코더의 두 개의 서브층에 대해서 이해하였다면 인코더에 대한 설명은 거의 끝났다. 트랜스포머에서는 이러한 두 개의 서브층을 가진 인코더에 추가적으로 사용하는 기법이 있는데, 바로 Add & Norm이다. 더 정확히는 잔차 연결(residual connection)과 층 정규화(layer normalization)를 의미한다.   

위의 그림은 앞서 Position-wise FFNN를 설명할 때 사용한 앞선 그림에서 화살표와 Add & Norm(잔차 연결과 정규화 과정)을 추가한 그림입니다. 추가된 화살표들은 서브층 이전의 입력에서 시작되어 서브층의 출력 부분을 향하고 있는 것에 주목합시다. 추가된 화살표가 어떤 의미를 갖고 있는지는 잔차 연결과 층 정규화를 배우고 나면 이해할 수 있다.   

### 1) 잔차 연결(Residual connection)
***
잔차 연결(residual connection)의 의미를 이해하기 위해서 어떤 함수 $H(x)$에 대한 이야기를 해보자.   

![image](https://github.com/parkm2ngyu00/AlgoPractice/assets/88785472/28820c1d-6a8a-4009-86f5-9c15930763d6)   

위 그림은 입력 $x$와에 대한 어떤 함수 $F(x)$의 값을 더한 함수 $H(x)$의 구조를 보여준다. 어떤 함수 $F(x)$가 트랜스포머에서는 서브층에 해당된다. 다시 말해 잔차 연결은 서브층의 입력과 출력을 더하는 것을 말한다. 앞서 언급했듯이 트랜스포머에서 서브층의 입력과 출력은 동일한 차원을 갖고 있으므로, 서브층의 입력과 서브층의 출력은 덧셈 연산을 할 수 있다. 이것이 바로 위의 인코더 그림에서 각 화살표가 서브층의 입력에서 출력으로 향하도록 그려졌던 이유이다. 잔차 연결은 컴퓨터 비전 분야에서 주로 사용되는 모델의 학습을 돕는 기법이다.   

이를 식으로 표현하면 $x+Sublayer(x)$이다.   

가령, 서브층이 멀티 헤드 어텐션이었다면 잔차 연결 연산은 다음과 같다.   

$$H(x) = x+Multi-head\ Attention(x)$$   

![image](https://github.com/parkm2ngyu00/AlgoPractice/assets/88785472/a8051b8f-3229-4831-8b2e-46f8eec1f881)   

위 그림은 멀티 헤드 어텐션의 입력과 멀티 헤드 어텐션의 결과가 더해지는 과정을 보여준다.   

### 2) 층 정규화(Layer Normalization)
***
잔차 연결을 거친 결과는 이어서 층 정규화 과정을 거치게된다. 잔차 연결의 입력을 $x$, 잔차 연결과 층 정규화 두 가지 연산을 모두 수행한 후의 결과 행렬을 $LN$이라고 하였을 때, 잔차 연결 후 층 정규화 연산을 수식으로 표현하자면 다음과 같다.   

$$LN = LayerNorm(x+Sublayer(x))$$   

층 정규화를 하는 과정에 대해서 이해해보자. 층 정규화는 텐서의 마지막 차원에 대해서 평균과 분산을 구하고, 이를 가지고 어떤 수식을 통해 값을 정규화하여 학습을 돕는다. 여기서 텐서의 마지막 차원이란 것은 트랜스포머에서는 $d_{model}$차원을 의미한다. 아래 그림은 $d_{model}$차원의 방향을 화살표로 표현하였다.   

![image](https://github.com/parkm2ngyu00/AlgoPractice/assets/88785472/a0c08a73-ac0d-4224-ab2c-c4929c548260)   

층 정규화를 위해서 우선, 화살표 방향으로 각각 평균 $μ$과 분산 $σ^{2}$을 구한다. 각 화살표 방향의 벡터를 $x_i$라고 명명해보자.   

![image](https://github.com/parkm2ngyu00/AlgoPractice/assets/88785472/d425392e-c288-40a0-bb2d-6e4a68b99faa)   

층 정규화의 수식을 알아봅시다. 여기서는 층 정규화를 두 가지 과정으로 나누어서 설명한다. 첫번째는 평균과 분산을 통한 정규화, 두번째는 감마와 베타를 도입하는 것이다.   

평균과 분산을 이용한 정규화의 수식은 다음과 같다.   

$$\hat{x}_{i, k} = \frac{x_{i, k}-μ_{i}}{\sqrt{σ^{2}_{i}+\epsilon}}$$   

$ϵ$(입실론)은 분모가 0이 되는 것을 방지하는 값이다.   

감마와 베타를 이용한 정규화는 일단 감마와 벡터라는 벡터를 준비해야 한다. 이들의 초기값은 각각 1과 0이다.   

![image](https://github.com/parkm2ngyu00/AlgoPractice/assets/88785472/ef8baf8d-c705-4d47-8ab0-0d37088cc69e)   

감마와 베타를 도입한 층 정규화의 최종 수식은 다음과 같으며 감마와 베타는 학습 가능한 파라미터이다.   

$$ln_{i} = γ\hat{x}_{i}+β = LayerNorm(x_{i})$$   

케라스에서는 층 정규화를 위한 LayerNormalization()를 제공하고 있으므로 이를 가져와 사용한다.   

## 마치며
이번 포스트에선 인코더의 나머지 서브층인 FFNN서브층과 행렬연산을 더 효과적으로 하기 위한 잔차 연결과 층 정규화에 대해 알아봤다. 다음 포스트에서는 인코더에서 디코더를 연결하는 부분과 디코더에 대해서 알아보도록 하겠다.   

## 참고하면 좋은 글
>[https://nlpinkorean.github.io/illustrated-transformer/](https://nlpinkorean.github.io/illustrated-transformer/)

## 참고 서적
>[딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/book/2155) 