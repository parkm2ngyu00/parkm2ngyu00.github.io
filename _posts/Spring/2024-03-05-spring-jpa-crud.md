---
title: "Spring JPA(EntityManager)를 이용한 REST CRUD 코드 작성(1)"
category: Spring
tag:
  - Spring
  - JPA
  - CRUD
toc: true
toc_sticky: true
---

## Spring JPA(EntityManager)를 이용한 REST CRUD 코드 작성

Spring의 JPA를 사용해 REST CRUD 코드를 작성하는 첫 번째 포스팅이다. 이번 포스팅에선 MySQL 연동과 각종 환경설정, 그리고 모든 결과 리스트를 조회하는 api를 설계하는 시간을 가져보도록 하겠다.

## 진행 순서

진행 순서는 다음과 같다.

1. Intellij와 MySQL DB 연동
2. Employee entity 코드 작성
3. DAO interface와 구현체 코드 작성
4. Service interface와 구현체 코드 작성
5. Controller 코드 작성
6. 테스트

### 1. Intellij와 MySQL DB 연동 및 환경설정

다음과 같이 SQL 스크립트를 작성하고 실행시킨다. 이 때 참고로 MySQL Connection은 username=springpractice, password=springpractice로 설정하였다.

```sql
CREATE DATABASE  IF NOT EXISTS `employee_directory`;
USE `employee_directory`;

--
-- Table structure for table `employee`
--

DROP TABLE IF EXISTS `employee`;

CREATE TABLE `employee` (
  `id` int NOT NULL AUTO_INCREMENT,
  `first_name` varchar(45) DEFAULT NULL,
  `last_name` varchar(45) DEFAULT NULL,
  `email` varchar(45) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=latin1;

--
-- Data for table `employee`
--

INSERT INTO `employee` VALUES
	(1,'Leslie','Andrews','leslie@luv2code.com'),
	(2,'Emma','Baumgarten','emma@luv2code.com'),
	(3,'Avani','Gupta','avani@luv2code.com'),
	(4,'Yuri','Petrov','yuri@luv2code.com'),
	(5,'Juan','Vega','juan@luv2code.com');
```

다음 스크립트를 실행시키면 `employee_directory` DB와 `employee` table이 생성되고 더미데이터 5가지가 테이블 안에 들어간다. 이 데이터는 나중에 테스트를 할 때 쓰일 예정이다.

```
#
# JDBC properties
#
spring.datasource.url=jdbc:mysql://localhost:3306/employee_directory
spring.datasource.username=springpractice
spring.datasource.password=springpractice
```

그리고 IntelliJ와의 연동을 위해 `application.properties`파일을 위와 같이 작성한다. 이러면 코드를 작성하기 위한 사전 DB 연동 작업은 모두 끝났다.

### 2. Employee entity 코드 작성

다음은 Employee Entity를 작성해보자. 아래와 같이 코드를 작성해야 하며, 특별히 주의해야 할 점은 없다.

> @Column(name="")에는 실제 DB table의 column 이름이 들어가면 되고, 필드 변수명은 언더바(\_)를 없애고 라마 표기법으로 바꿔주면 된다.(언더바 다음 글자를 대문자로 first_name => firstName)

<script src="https://gist.github.com/parkm2ngyu00/da2d6316d4c604cde5d1b548654740f3.js"></script>

### 3. DAO interface와 구현체 코드 작성

<script src="https://gist.github.com/parkm2ngyu00/a5b5bd55d0dbcb0376e43707a444bbe8.js"></script>

<script src="https://gist.github.com/parkm2ngyu00/90f3cbb84a6ba71261125bb3a25a76f7.js"></script>

이번 포스팅에선 findAll, 즉 모든 결과를 리스트로 반환하는 api를 작성할 것이기 때문에 다음과 같이 findAll method를 설계하면 된다.

> 참고로 인터페이스와 구현체를 나누어 코드를 작성하는 이유는 객체지향적으로 설계해서 유지보수를 유연하게 하기 위함이고, 이에 대해 더 자세히 알고 싶다면 여러 자료를 추가로 찾아보는 것을 권장한다.

### 4. Service interface와 구현체 코드 작성

<script src="https://gist.github.com/parkm2ngyu00/1f968dc95fdfe4c1f6786b4f203a100b.js"></script>

<script src="https://gist.github.com/parkm2ngyu00/31c7a94950fa4bfd67b8fa75c3c77da0.js"></script>

Service 코드를 작성한다. 마찬가지로 위와 같이 인터페이스와 구현체를 분리해 작성한다.

### 5. Controller 코드 작성

api endpoint를 만들어보자. 다음과 같이 코드를 작성하면 된다.

<script src="https://gist.github.com/parkm2ngyu00/f7be1c32d78307cfe1e3090174e10999.js"></script>

이제 필요한 모든 코드는 작성하였다. 이제 `/api/employees`로 get 요청을 보내 결과가 아까 DB에 저장한 데이터와 동일한지 검증해보자.

### 6. 테스트

![image](https://github.com/parkm2ngyu00/parkm2ngyu00.github.io/assets/88785472/4ccc0216-bc53-4748-b90e-c2c5a9b2dd94)

다음과 같이 올바른 결과가 나오는 것을 볼 수 있다.

## 전체 코드 구조

뭔가가 많이 코드로 작성되어서 구조가 조금 헷갈릴 수 있다. 이를 도식화해 보면 그리 어렵지 않으니 밑의 그림을 참고하자.

![image](https://github.com/parkm2ngyu00/parkm2ngyu00.github.io/assets/88785472/84433e1c-a704-4f3a-9906-6c0b9a7ff0a1)

Controller에서 HTTP Request를 받으면 Service class의 main logic이 동작하고, EmployeeDAO에서 실제 DB와 연동되어 (사실 DB까지 가는 길에 조금 더 추상화된 계층이 있다) 최종 코드가 완성되는 것이다.

## 마치며

이번 포스팅에선 MySQL과의 연동, Controller, Service, DAO의 코드를 작성하였고 전체 결과를 리스트로 조회하는 api를 작성하였다. 다음 포스팅엔 나머지 CRUD에 대한 기능 구현을 마치도록 하겠다.

## 참고 자료 및 강의

[Udemy Spring Boot 강의](https://www.udemy.com/course/spring-hibernate-tutorial)
