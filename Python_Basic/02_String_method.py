# 문자열 메서드 정리

## 1. join() 메서드 정리
## <A>.join(<B>) list 형태의 <B> 를 <A>를 구분자로 하여 연결하는 방법

method_join = ["Sang-Hoon, Park ", "Ha-Young, Kim", "player3"]
join_res = " ".join(method_join) # players 라는 변수에 " "(띄어쓰기)를 구분자로 하여 player 리스트를 연결한다.
print(join_res) #출력 결과: Sang-Hoon, Park  Ha-Young, Kim player3

reversed_join_res = " ".join(reversed(method_join)) # players 라는 변수에 " "(띄어쓰기)를 구분자로 하여 player 리스트를 역순으로 연결한다.
print(reversed_join_res) #출력 결과: player3 Ha-Young, Kim Sang-Hoon, Park

############################################################################

## 2. ljust(), rjust() 메서드 정리
## <A>.ljust(<width>, <fillchar>) <width> 길이를 전체로 보고, <A>를 먼저 채우고[왼쪽에..], 나머지 길이에 <fillchar>를 채우는 방법
## <A>.rjust(<width>, <fillchar>) <width> 길이에 <fillchar>를 먼저 채우고[왼쪽에..], 나머지 길이에 <A>를 채우는 방법 

method_lrjust = "Sang-Hoon"
ljust_res = method_lrjust.ljust(20, "_")
print(ljust_res) # 출력결과: Sang-Hoon___________

rjust_res = method_lrjust.rjust(20, "_")
print(rjust_res) # 출력결과: ___________Sang-Hoon

############################################################################

## 3. format() 메서드 정리
## <A>.format() 문자열 A 에 변수를 추가하거나 형식화 하기위해 사용된다.

print("{0} {1}".format("안녕", "상훈!")) # 출력결과: 안녕 상훈!
print("안녕 내 이름은 {who} 이고, 나이는 {age}살 이야" .format(who="상훈", age=31)) #출력결과: 안녕 내 이름은 상훈 이고, 나이는 31살 이야
print("안녕 내 이름은 {who} 이고, 나이는 {0}살 이야" .format(31, who="상훈")) #출력결과: 안녕 내 이름은 상훈 이고, 나이는 31살 이야
print("{} {} {}" .format("안녕", "즐거운", "파이썬!")) #인덱스 제거, 출력결과 : 안녕 즐거운 파이썬!

############################################################################

## 4. splitlines() 메서드 정리
## <A>.splitlines()는 문자열 <A>에 대해서 줄바꿈 문자를 기준으로 분리한 리스트를 return 한다. 

method_splitlines = "안녕하세요\n박상훈 입니다.\n잘 부탁드립니다."
splitlines_res = method_splitlines.splitlines()
print(splitlines_res) # 출력 결과: ['안녕하세요', '박상훈 입니다.', '잘 부탁드립니다.']

############################################################################

## 5. split() 메서드 정리
## <A>.split(<B>, <n>)는 문자열 <A>에 대해서 <B> 문자를 기준으로 왼쪽부터 <n>회 분리한 리스트를 return 한다. 
## <A>.rsplit(<B>, <n>)는 문자열 <A>에 대해서 <B> 문자를 기준으로 오른쪽부터 <n>회 분리한 리스트를 return 한다. 
## <n>을 안쓰는 경우에는 문자열 <A>에 대해서 <B>문자를 기준으로 모두 분리한 리스트를 return 한다.

method_split = "상훈#상혁#은#형제#입니다."
split_res = method_split.split("#") # "#"를 기준으로 나누겠다는 의미
print(split_res) #출력결과: ['상훈', '상혁', '은', '형제', '입니다.']
print(" ".join(split_res)) #출력결과: 상훈 상혁 은 형제 입니다.

split_res_left = method_split.split("#", 3)
split_res_right = method_split.rsplit("#", 3)
print(split_res_left) #출력결과: ['상훈', '상혁', '은', '형제#입니다.'] -> 왼쪽부터 "#"을 기준으로 세개를 나눔
print(split_res_right) #출력결과: ['상훈#상혁', '은', '형제', '입니다.'] -> 오른쪽부터 "#"을 기준으로 세개를 나눔

############################################################################

## 6. strip() 메서드 정리
## <A>.strip(<B>)는 문자열 <A>에 대해서 앞뒤의 문자열 <B> 를 제거하여 return 한다. 

method_strip = "aaaaa상훈 aa, 파이썬, hello World! aaaaaa"
strip_res = method_strip.strip("aa")
print(strip_res) #출력결과: 상훈 aa, 파이썬, hello World! -> 앞과 뒤에 있는것 모두지운다.. 그런데 앞에는 5개인데 모두지우는것으로봐서 글자단위 검사 하는듯..

