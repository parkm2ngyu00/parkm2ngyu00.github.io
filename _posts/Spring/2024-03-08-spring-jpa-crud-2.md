---
title: "Spring JPA(EntityManager)를 이용한 REST CRUD 코드 작성(2)"
category: Spring
tag:
  - Spring
  - JPA
  - CRUD
toc: true
toc_sticky: true
---

## Spring JPA(EntityManager)를 이용한 REST CRUD 코드 작성(2)

이번 포스팅에선 저번 포스팅에 이어 나머지 Create, Read, Update, Delete에 대한 기능들을 완성해본다. 과정은 다음과 같다.

1. EmployeeDAO 인터페이스와 구현체 구현
2. EmployeeService 인터페이스와 구현체 구현
3. EmployeeController 구현

### 1. EmployeeDAO 인터페이스와 구현체 구현

저번 포스팅에선 Employee의 전체 리스트를 가져오는 기능만을 구현하였지만, 이번에는 id를 활용해 특정 Employee를 가져오는 기능과 삭제하는 기능, 또한 새로운 Employee를 추가하는 기능도 구현한다. 이를 위해 `EmployeeDAO`와 `EmployeeDAOImpl`을 다음과 같이 수정 및 추가하자.

<script src="https://gist.github.com/parkm2ngyu00/fda9564d2c878ee231a252b189954d0b.js"></script>

<script src="https://gist.github.com/parkm2ngyu00/9c646a015dd76446b9f909400a6a9a2d.js"></script>

### 2. EmployeeService 인터페이스와 구현체 구현

다음은 서비스 인터페이스와 구현체를 다음과 같이 리팩토링 해준다.

<script src="https://gist.github.com/parkm2ngyu00/f97b17e50790550b96db33bae960af3b.js"></script>

<script src="https://gist.github.com/parkm2ngyu00/39f167a1ad6433416d75c35415b4a1a3.js"></script>

### 3. EmployeeController 구현

마지막으로 컨트롤러의 코드를 리팩토링 해준다. request의 path variable을 가져오는 경우도 있고, body의 json을 가져오는 경우도 있기 때문에 이를 유의해 작성하여야 한다.

<script src="https://gist.github.com/parkm2ngyu00/804a4d5a4080dc8782fb2ed92468fcec.js"></script>

여러 HTTP method를 통해 구현이 된 것을 볼 수 있다.

## 마치며

사실 위의 코드는 문제가 있는 코드이다. 의존성 주입을 따로 설정 클래스에서 하는 것이 아닌 서비스와 컨트롤러 클래스에서 하고있는 것이다. 원래는 `Config.java`와 같은 의존성 주입을 위한 클래스를 따로 만들어야 하지만, 위의 예제의 본질은 그것이 아니고 여러 HTTP method를 사용해 아주 간단한 CRUD api를 설계하는 것이기 때문에, 어느정도 디자인 패턴이나 의존성 주입과 같은 문제들은 무시하고 진행하였다. 또한 요즘은 JPA를 한 단계 더 추상화시킨 Spring Data JPA를 많이 사용하는 추세이다. 그렇다고 해서 `EntityManager`애 대한 내용을 아예 모른다면 나중에 문제가 될 수 있기 때문에, 두 가지 방법 모두를 알고 있는것을 추천한다.
