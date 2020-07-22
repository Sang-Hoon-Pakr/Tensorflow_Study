## lambda 유용하다고하는데 사용법 완벽하게 익혀보자
## 사용법 lambda 인자 : 표현식 으로 사용한다.

#예를들어 아래와 같은 함수가 존재한다고 하자 
def exlam(x, y):
    return x + y

print("함수 호출 결과: ", exlam(10,30)) #출력결과 : 40  

print("람다 사용: ", (lambda x, y : x + y)(10,30)) #출력결과 : 40

func = lambda x : x**2 # x 입력값을 제곱하는 람다식을 func로 정의
print("func 출력 결과: ", func(9)) #출력결과 : 81

def inc(n):
    return lambda x: x + n

f = inc(2) #inc 함수의 매개변수 n의 값을 2로 넣어준것. 
g = inc(4) #inc 함수의 매개변수 n의 값을 4로 넣어준것. 

print(f(12)) #람다 형식 f의 매개변수로 12를 넣어준것 즉, 위 식에서 x 값을 전달해준것. 출력결과: 14
print(g(12)) #람다 형식 g의 매개변수로 12를 넣어준것 즉, 위 식에서 x 값을 전달해준것. 출력결과: 16
print(inc(11)(12)) #inc 함수의 매개변수에 11을 주고, 람다 형식 f의 매개변수로 12를 넣어준것 즉, 위 식에서 x 값을 전달해준것. 출력결과: 23

## 그런데 lambda는 단독으로 쓰기보단 map(), reduce(), filter()와 함께 사용하는
## 경우가 많습니다.

## map() 함수의 사용법은 map(함수, 리스트) 입니다.
## 이 함수는 함수와 리스트를 인자로 받아서 리스트로 부터 원소를 하나씩 꺼내서 함수를 적용시킨 다음
## 그 결과를 새로운 리스트에 담아주는 기능을 합니다.
print(list(map(lambda x : x**2, range(5)))) #출력결과 : [0, 1, 4, 9, 16]


## reduce() 함수의 사용법은 reduce(함수, 순서형 자료) 입니다. #순서형자료 = 튜플, 리스트, 문자열
## 이 함수는 함수와 순서형 자료를 인자로 받아서 원소들을 누적으로 함수에 적용합니다.
from functools import reduce #reduce 함수 import 수행
print(reduce(lambda x,y: x+y, range(5))) # range(5) = [0, 1, 2, 3, 4] 출력결과 : 10
#더해지는 순서  = ((((0+1)+2)+3)+4)  입니다.
print(reduce(lambda x,y: x+y, 'ABCDEFG')) #출력결과 : ABCDEFG
print(reduce(lambda x,y: y+x, 'ABCDEFG')) #출력결과 : GFEDCBA
#더해지는 순서 = (G+(F+(E+(D+(C+(B+A)))))) -> GFEDCBA 로나오는것


## filter() 함수의 사용법은 filter(함수, 리스트) 입니다. 
## 이 함수는 함수와 리스트를 인자로 받아서 함수 조건에 맞는 값들만 필터링 해주는 기능을 합니다.
print(list(filter(lambda x: x < 5, range(10)))) #출력결과 : [0, 1, 2, 3, 4]
print(list(filter(lambda x: x % 2, range(10)))) #출력결과 : [1, 3, 5, 7, 9] #홀수는 나머지가 1이라 True 기 때문에.. filter 함수 통과
