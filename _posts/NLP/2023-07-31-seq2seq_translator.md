---
title: "[NLP] 시퀀스-투-시퀀스(Sequence-to-Sequence)로 번역기 구현하기"
category: NLP
tags:
    - NLP
    - seq2seq
    - RNN
    - LSTM
toc: true
toc_sticky: true
---

## 문자 레벨 기계 번역기(Character-Level Neural Machine Translation) 구현하기

### 0. 데이터 준비하기
***
실제 성능이 좋은 기계 번역기를 구현하려면 정말 방대한 데이터가 필요하므로 여기서는 방금 배운 seq2seq를 실습해보는 수준에서 아주 간단한 기계 번역기를 구축해보자. 기계 번역기를 훈련시키기 위해서는 훈련 데이터로 병렬 코퍼스(parallel corpus)가 필요하다. 병렬 코퍼스란, 두 개 이상의 언어가 병렬적으로 구성된 코퍼스를 의미한다.   

다운로드 링크 : [http://www.manythings.org/anki](http://www.manythings.org/anki)   

이번 실습에서는 프랑스-영어 병렬 코퍼스인 fra-eng.zip 파일을 사용한다. 위의 링크에서 해당 파일을 다운받으시면 된다. 해당 파일의 압축을 풀면 fra.txt라는 파일이 있는데 이 파일이 이번 실습에서 사용할 파일이다.   

### 1. 병렬 코퍼스 데이터에 대한 이해와 전처리
***
여기서 사용할 fra.txt 데이터는 위와 같이 왼쪽의 영어 문장과 오른쪽의 프랑스어 문장 사이에 탭으로 구분되는 구조가 하나의 샘플이다. 그리고 이와 같은 형식의 약 16만개의 병렬 문장 샘플을 포함하고 있다. 해당 데이터를 읽고 전처리를 진행해보자. 앞으로의 코드에서 src는 source의 줄임말로 입력 문장을 나타내며, tar는 target의 줄임말로 번역하고자 하는 문장을 나타낸다.
```python
import os
import shutil
import zipfile

import pandas as pd
import tensorflow as tf
import urllib3
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
```
```python
http = urllib3.PoolManager()
url ='http://www.manythings.org/anki/fra-eng.zip'
filename = 'fra-eng.zip'
path = os.getcwd()
zipfilename = os.path.join(path, filename)
with http.request('GET', url, preload_content=False) as r, open(zipfilename, 'wb') as out_file:       
    shutil.copyfileobj(r, out_file)

with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
    zip_ref.extractall(path)
```
```python
lines = pd.read_csv('fra.txt', names=['src', 'tar', 'lic'], sep='\t')
del lines['lic']
print('전체 샘플의 개수 :',len(lines))
```
```text
전체 샘플의 개수 : 191954
```
전체 샘플의 개수는 총 약 19만 2천개이다.   
```python
lines = lines.loc[:, 'src':'tar']
lines = lines[0:60000] # 6만개만 저장
lines.sample(10)
```
해당 데이터는 약 19만 2천개의 병렬 문장 샘플로 구성되어있지만 여기서는 간단히 60,000개의 샘플만 가지고 기계 번역기를 구축해보도록 하자. 우선 전체 데이터 중 60,000개의 샘플만 저장하고 현재 데이터가 어떤 구성이 되었는지 확인해보자.   

![image](https://github.com/parkm2ngyu00/parkm2ngyu00.github.io/assets/88785472/78f95bf3-1289-4ee2-8cfb-a3b6d9f94be8)   

위의 테이블은 랜덤으로 선택된 10개의 샘플을 보여준다. 번역 문장에 해당되는 프랑스어 데이터는 앞서 배웠듯이 시작을 의미하는 심볼 <sos>과 종료를 의미하는 심볼 <eos>을 넣어주어야 한다. 여기서는 <sos>와 <eos> 대신 \t를 시작 심볼, \n을 종료 심볼로 간주하여 추가하고 다시 데이터를 출력해보자.
```python
lines.tar = lines.tar.apply(lambda x : '\t '+ x + ' \n')
lines.sample(10)
```   
   
![image](https://github.com/parkm2ngyu00/parkm2ngyu00.github.io/assets/88785472/2d835157-f3e1-4ad3-bc6b-dbcbf666f970)   

프랑스어 데이터에서 시작 심볼과 종료 심볼이 추가된 것을 볼 수 있다. 문자 집합을 생성해보자. 단어 집합이 아니라 문자 집합이라고 하는 이유는 토큰 단위가 단어가 아니라 문자이기 때문이다.   
```python
# 문자 집합 구축
src_vocab = set()
for line in lines.src: # 1줄씩 읽음
    for char in line: # 1개의 문자씩 읽음
        src_vocab.add(char)

tar_vocab = set()
for line in lines.tar:
    for char in line:
        tar_vocab.add(char)

```
```python
src_vocab_size = len(src_vocab)+1
tar_vocab_size = len(tar_vocab)+1
print('source 문장의 char 집합 :',src_vocab_size)
print('target 문장의 char 집합 :',tar_vocab_size)
```
```text
source 문장의 char 집합 : 79
target 문장의 char 집합 : 105
```
이후에 vocab을 아스키 코드 순서로 정렬해주고 정수 인코딩을 하기 위해 각 문자에 인덱스를 부여한다.
```python
src_vocab = sorted(list(src_vocab))
tar_vocab = sorted(list(tar_vocab))
src_to_index = dict([(word, i+1) for i, word in enumerate(src_vocab)])
tar_to_index = dict([(word, i+1) for i, word in enumerate(tar_vocab)])
print(src_to_index)
print(tar_to_index)
```
```text
{' ': 1, '!': 2, '"': 3, '$': 4, '%': 5, ... 중략 ... 'x': 73, 'y': 74, 'z': 75, 'é': 76, '’': 77, '€': 78}
{'\t': 1, '\n': 2, ' ': 3, '!': 4, '"': 5, ... 중략 ... 'û': 98, 'œ': 99, 'С': 100, '\u2009': 101, '‘': 102, '’': 103, '\u202f': 104}
```
인덱스가 부여된 문자 집합으로부터 갖고있는 훈련 데이터에 정수 인코딩을 수행한다. 우선 인코더의 입력이 될 영어 문장 샘플에 대해서 정수 인코딩을 수행해보고, 5개의 샘플을 출력해보자.
```python
encoder_input = []

# 1개의 문장
for line in lines.src:
  encoded_line = []
  # 각 줄에서 1개의 char
  for char in line:
    # 각 char을 정수로 변환
    encoded_line.append(src_to_index[char])
  encoder_input.append(encoded_line)
print('source 문장의 정수 인코딩 :',encoder_input[:5])
```
```text
source 문장의 정수 인코딩 : [[30, 64, 10], [30, 64, 10], [30, 64, 10], [31, 58, 10], [31, 58, 10]]
```
정수 인코딩이 수행된 것을 볼 수 있다. 다음은 디코더의 입력이 될 프랑스어 데이터에 대해서 정수 인코딩을 수행해보자.
```python
decoder_input = []
for line in lines.tar:
  encoded_line = []
  for char in line:
    encoded_line.append(tar_to_index[char])
  decoder_input.append(encoded_line)
print('target 문장의 정수 인코딩 :',decoder_input[:5])
```
```text
target 문장의 정수 인코딩 : [[1, 3, 48, 53, 3, 4, 3, 2], [1, 3, 39, 53, 70, 55, 60, 57, 14, 3, 2], [1, 3, 28, 67, 73, 59, 57, 3, 4, 3, 2], [1, 3, 45, 53, 64, 73, 72, 3, 4, 3, 2], [1, 3, 45, 53, 64, 73, 72, 14, 3, 2]]
```
정상적으로 정수 인코딩이 수행된 것을 볼 수 있다. 아직 정수 인코딩을 수행해야 할 데이터가 하나 더 남아있다. 디코더의 예측값과 비교하기 위한 실제값이 필요하다. 그런데 이 실제값에는 시작 심볼에 해당되는 <sos>가 있을 필요가 없다. 즉, 모든 프랑스어 문장의 맨 앞에 붙어있는 '\t'를 제거하도록 한다.
```python
decoder_target = []
for line in lines.tar:
  timestep = 0
  encoded_line = []
  for char in line:
    if timestep > 0:
      encoded_line.append(tar_to_index[char])
    timestep = timestep + 1
  decoder_target.append(encoded_line)
print('target 문장 레이블의 정수 인코딩 :',decoder_target[:5])
```
```text
target 문장 레이블의 정수 인코딩 : [[3, 48, 53, 3, 4, 3, 2], [3, 39, 53, 70, 55, 60, 57, 14, 3, 2], [3, 28, 67, 73, 59, 57, 3, 4, 3, 2], [3, 45, 53, 64, 73, 72, 3, 4, 3, 2], [3, 45, 53, 64, 73, 72, 14, 3, 2]]
```
앞서 먼저 만들었던 디코더의 입력값에 해당되는 decoder_input 데이터와 비교하면 decoder_input에서는 모든 문장의 앞에 붙어있던 숫자 1이 decoder_target에서는 제거된 것을 볼 수 있다. '\t'가 인덱스가 1이므로 정상적으로 제거된 것이다. 모든 데이터에 대해서 정수 인덱스로 변경하였으니 패딩 작업을 수행한다. 패딩을 위해서 영어 문장과 프랑스어 문장 각각에 대해서 가장 길이가 긴 샘플의 길이를 확인한다.
```python
max_src_len = max([len(line) for line in lines.src])
max_tar_len = max([len(line) for line in lines.tar])
print('source 문장의 최대 길이 :',max_src_len)
print('target 문장의 최대 길이 :',max_tar_len)
```
```text
source 문장의 최대 길이 : 23
target 문장의 최대 길이 : 76
```
각각 23와 76의 길이를 가진다. 이번 병렬 데이터는 영어와 프랑스어의 길이는 하나의 쌍이라고 하더라도 전부 다르므로 패딩을 할 때도 이 두 개의 데이터의 길이를 전부 동일하게 맞춰줄 필요는 없다. 영어 데이터는 영어 샘플들끼리, 프랑스어는 프랑스어 샘플들끼리 길이를 맞추어서 패딩하면 됩다. 여기서는 가장 긴 샘플의 길이에 맞춰서 영어 데이터의 샘플은 전부 길이가 23이 되도록 패딩하고, 프랑스어 데이터의 샘플은 전부 길이가 76이 되도록 패딩한다.
```python
encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
decoder_input = pad_sequences(decoder_input, maxlen=max_tar_len, padding='post')
decoder_target = pad_sequences(decoder_target, maxlen=max_tar_len, padding='post')
```
모든 값에 대해서 원-핫 인코딩을 수행한다. 문자 단위 번역기므로 워드 임베딩은 별도로 사용되지 않으며, 예측값과의 오차 측정에 사용되는 실제값뿐만 아니라 입력값도 원-핫 벡터를 사용한다.
```python
encoder_input = to_categorical(encoder_input)
decoder_input = to_categorical(decoder_input)
decoder_target = to_categorical(decoder_target)
```
### 2. 교사 강요(Teacher forcing)
***
코드를 따라가다 보면 한 가지 이상한 점을 발견할 수 있다. 현재 시점의 디코더 셀의 입력은 오직 이전 디코더 셀의 출력을 입력으로 받는다고 설명하였는데 decoder_input이 왜 필요할까?   

훈련 과정에서는 이전 시점의 디코더 셀의 출력을 현재 시점의 디코더 셀의 입력으로 넣어주지 않고, 이전 시점의 실제값을 현재 시점의 디코더 셀의 입력값으로 하는 방법을 사용한다. 그 이유는 이전 시점의 디코더 셀의 예측이 틀렸는데 이를 현재 시점의 디코더 셀의 입력으로 사용하면 현재 시점의 디코더 셀의 예측도 잘못될 가능성이 높고 이는 연쇄 작용으로 디코더 전체의 예측을 어렵게 한다. 이런 상황이 반복되면 훈련 시간이 느려진다. 만약 이 상황을 원하지 않는다면 이전 시점의 디코더 셀의 예측값 대신 실제값을 현재 시점의 디코더 셀의 입력으로 사용하는 방법을 사용할 수 있다. 이와 같이 RNN의 모든 시점에 대해서 이전 시점의 예측값 대신 실제값을 입력으로 주는 방법을 교사 강요라고 합니다.   

### 3. seq2seq 기계 번역기 훈련시키기
***
seq2seq 모델을 설계하고 교사 강요를 사용하여 훈련시켜보도록 하자.
```python
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
import numpy as np
```
```python
encoder_inputs = Input(shape=(None, src_vocab_size))
encoder_lstm = LSTM(units=256, return_state=True)

# encoder_outputs은 여기서는 불필요
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

# LSTM은 바닐라 RNN과는 달리 상태가 두 개. 은닉 상태와 셀 상태.
encoder_states = [state_h, state_c]
```
우선 LSTM의 은닉 상태 크기는 256으로 선택했다. 인코더의 내부 상태를 디코더로 넘겨주어야 하기 때문에 return_state=True로 설정한다. 인코더에 입력을 넣으면 내부 상태를 리턴한다.   

LSTM에서 state_h, state_c를 리턴받는데, 이는 각각 LSTM의 은닉 상태와 셀 상태에 해당됩니다. 은닉 상태만 전달하는 게 아니라 은닉 상태와 셀 상태 두 가지를 전달한다고 생각하면 된다. 이 두 가지 상태를 encoder_states에 저장한다. encoder_states를 디코더에 전달하므로서 이 두 가지 상태 모두를 디코더로 전달한다. 이것이 이전 포스트에서 배운 컨텍스트 벡터이다.
```python
decoder_inputs = Input(shape=(None, tar_vocab_size))
decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True)

# 디코더에게 인코더의 은닉 상태, 셀 상태를 전달.
decoder_outputs, _, _= decoder_lstm(decoder_inputs, initial_state=encoder_states)

decoder_softmax_layer = Dense(tar_vocab_size, activation='softmax')
decoder_outputs = decoder_softmax_layer(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
```
디코더는 인코더의 마지막 은닉 상태를 초기 은닉 상태로 사용한다. 위에서 initial_state의 인자값으로 encoder_states를 주는 코드가 이에 해당된다. 또한 동일하게 디코더의 은닉 상태 크기도 256으로 주었다. 디코더도 은닉 상태, 셀 상태를 리턴하기는 하지만 훈련 과정에서는 사용하지 않는다. 그 후 출력층에 프랑스어의 단어 집합의 크기만큼 뉴런을 배치한 후 소프트맥스 함수를 사용하여 실제값과의 오차를 구한다.
```python
model.fit(x=[encoder_input, decoder_input], y=decoder_target, batch_size=64, epochs=40, validation_split=0.2)
```
input으로 [encoder_input, decoder_input]을 입력하고, 타겟 데이터로 decoder_target을 입력해주며 모델을 학습시킨다   

### 4. seq2seq 기계 번역기 동작시키기
***
앞선 포스트에서 seq2seq는 훈련할 때와 동작할 때의 방식이 다르다고 언급한 바 있다. 이번에는 입력한 문장에 대해서 기계 번역을 하도록 모델을 조정하고 동작시켜보도록 한다

전체적인 번역 동작 단계를 정리하면 아래와 같다.
1. 번역하고자 하는 입력 문장이 인코더에 들어가서 은닉 상태와 셀 상태를 얻는다.
2. 상태와 <SOS>에 해당하는 \t를 디코더로 보낸다.
3. 디코더가 <EOS>에 해당하는 \n이 나올 때까지 다음 문자를 예측하는 행동을 반복한다.   

```python
encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)
```
우선 인코더를 정의한다. encoder_inputs와 encoder_states는 훈련 과정에서 이미 정의한 것들을 재사용하는 것이다. 이제 디코더를 설계해보자.
```python
# 이전 시점의 상태들을 저장하는 텐서
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# 문장의 다음 단어를 예측하기 위해서 초기 상태(initial_state)를 이전 시점의 상태로 사용.
# 뒤의 함수 decode_sequence()에 동작을 구현 예정
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

# 훈련 과정에서와 달리 LSTM의 리턴하는 은닉 상태와 셀 상태를 버리지 않음.
decoder_states = [state_h, state_c]
decoder_outputs = decoder_softmax_layer(decoder_outputs)
decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs] + decoder_states)
```
```python
index_to_src = dict((i, char) for char, i in src_to_index.items())
index_to_tar = dict((i, char) for char, i in tar_to_index.items())
```
단어로부터 인덱스를 얻는 것이 아니라 인덱스로부터 단어를 얻을 수 있는 index_to_src와 index_to_tar를 만들었다.
```python
def decode_sequence(input_seq):
  # 입력으로부터 인코더의 상태를 얻음
  states_value = encoder_model.predict(input_seq)

  # <SOS>에 해당하는 원-핫 벡터 생성
  target_seq = np.zeros((1, 1, tar_vocab_size))
  target_seq[0, 0, tar_to_index['\t']] = 1.

  stop_condition = False
  decoded_sentence = ""

  # stop_condition이 True가 될 때까지 루프 반복
  while not stop_condition:
    # 이점 시점의 상태 states_value를 현 시점의 초기 상태로 사용
    output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

    # 예측 결과를 문자로 변환
    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    sampled_char = index_to_tar[sampled_token_index]

    # 현재 시점의 예측 문자를 예측 문장에 추가
    decoded_sentence += sampled_char

    # <eos>에 도달하거나 최대 길이를 넘으면 중단.
    if (sampled_char == '\n' or
        len(decoded_sentence) > max_tar_len):
        stop_condition = True

    # 현재 시점의 예측 결과를 다음 시점의 입력으로 사용하기 위해 저장
    target_seq = np.zeros((1, 1, tar_vocab_size))
    target_seq[0, 0, sampled_token_index] = 1.

    # 현재 시점의 상태를 다음 시점의 상태로 사용하기 위해 저장
    states_value = [h, c]

  return decoded_sentence
```
```python
for seq_index in [3,50,100,300,1001]: # 입력 문장의 인덱스
  input_seq = encoder_input[seq_index:seq_index+1]
  decoded_sentence = decode_sequence(input_seq)
  print(35 * "-")
  print('입력 문장:', lines.src[seq_index])
  print('정답 문장:', lines.tar[seq_index][2:len(lines.tar[seq_index])-1]) # '\t'와 '\n'을 빼고 출력
  print('번역 문장:', decoded_sentence[1:len(decoded_sentence)-1]) # '\n'을 빼고 출력
```
```text
-----------------------------------
입력 문장: Hi.
정답 문장: Salut ! 
번역 문장: Salut. 
-----------------------------------
입력 문장: I see.
정답 문장: Aha. 
번역 문장: Je change. 
-----------------------------------
입력 문장: Hug me.
정답 문장: Serrez-moi dans vos bras ! 
번역 문장: Serre-moi dans vos patents ! 
-----------------------------------
입력 문장: Help me.
정답 문장: Aidez-moi. 
번역 문장: Aidez-moi. 
-----------------------------------
입력 문장: I beg you.
정답 문장: Je vous en prie. 
번역 문장: Je vous en prie. 
```

지금까지 문자 단위의 seq2seq를 구현하였다. 다음 포스트에서는 이번 실습에서 배운 내용을 바탕으로 문자 단위에서 단어 단위로 확장해서 기계 번역기를 구현해보겠다.   

## 참고 서적
>[딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/book/2155)  