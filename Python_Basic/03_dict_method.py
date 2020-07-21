# #딕셔너리 분기 태우는 방법
# def hello():
#     print("hello world!")
# def world():
#     print("world hello!")

# function = dict(h=hello, w=world)

# action = input("w or h 를 입력하세요: ")

# function[action]()

from itertools import permutations
def solution(numbers):
    answer = ''
    numbers = list(map(str, numbers))
    numbers.sort(key = lambda x: x*3, reverse=True)
    print(numbers)
    
    return answer

solution([1,2,3,5,4])