# 기본 코딩
## 중복되는 값들 카운트
# def solutions(n):
#     gro = list(set(n))
#     Result = dict()
#     for ip in gro:
#         Result[ip] = n.count(ip)
#     print(Result)

# if __name__ =="__main__":
#     solutions([1, 2, 2, 3, 4, 4, 4, 5])
#############################################################################################

## 최대 공약수
# def solutions(a, b):
#     i = min(a, b)
    
#     while True:
#         if a%i == 0 and b%i ==0:
#             return i
#         i -=1

# if __name__ =="__main__":
#     print(solutions(6 ,20))
#############################################################################################
    
## 하노이의 탑
# def solutions(n, from_pos, to_pos, middle_pos):
#     if n == 1:
#         print(from_pos, "->", to_pos)
#         return 0
    
#     solutions(n-1, from_pos, middle_pos, to_pos)
#     print(from_pos, "->", to_pos)
#     solutions(n-1, middle_pos, to_pos, from_pos)

# if __name__ =="__main__":
#     print(solutions(3, 1, 3, 2))

#############################################################################################
    
## 순차 탐색
# def search_list(n ,a):
#     length_n = len(n)

#     for i in range(length_n):
#         if n[i] == a:
#             return i

# if __name__ =="__main__":
#     print(search_list([1,2,3,4,5,6,7], 5))

#############################################################################################
    
## 버블 정렬
# def bubble_sort(n):
#     length_n = len(n)-1
#     for i in range(length_n): # 전체를 몇번 회전할 것인가 정하는 인덱스 j 값을 어디까지 돌릴까를 위한 값임 그냥
#         for j in range(length_n-i): #j 는 무조건 0부터 길이 지정까지 (한번 돌아가면 맨뒤에 고정되기 때문에 i만큼 깍아줌)
#             if n[j] > n[j+1] : # 비교하는 두 인덱스 중 앞에꺼가 더 크면 아래와 같이 값 순서를 바꿔줌
#                 n[j], n[j+1] = n[j+1], n[j]
#         print(n)
#     return n


# if __name__ =="__main__":
#     print(bubble_sort([2, 4, 5, 1, 3]))

#############################################################################################
    
## 선택 정렬 가장 값은 값을 맨 앞으로 변경 변경 해가는 아이디어
# def sel_sort(n):
#     length_n = len(n)
#     for i in range(length_n-1):
#         min_idx = i # 순서대로 가장 작은 수라고 가정 하고 진행 ( 작은것 부터 세우기 때문에 인덱스는 0부터 시작)
#         for j in range(i+1 , length_n): #첫 기준을 제외하고 뒤에중에서 가장 작은놈을 찾음
#             if n[min_idx] > n[j] : #내가 기준으로 세웠던(가장 작다고 생각했던..) 것보다 뒤에 있는 놈들중에 가장 작은게 있으면 
#                 min_idx = j #가장 작다고 생각하는 인덱스를 바꿔.. 
#         n[i], n[min_idx] = n[min_idx] , n[i] #어쨋든 포문을 돌면서 여러번 바뀔수도 있는데 최종적으로 가장 작은놈을 가장 앞인덱스에 넣고 어쨋든 바꿈
#         print(n)
#     return n


# if __name__ =="__main__":
#     print(sel_sort([2, 4, 5, 1, 3]))

#############################################################################################
    
# ## 삽입 정렬
# def ins_sort(n):
#     length_n = len(n)
#     for i in range(1, length_n):
#         key = n[i] #기준을 정함 
#         j = i - 1 #그 기준에 앞에 있는 값들을 돌릴거야 
#         while key < n[j] and j >=0: # 키값보다 앞에 있는놈들이 더크고, 비교하는 인덱스가 0보다 큰동안 돌리고 
#             n[j+1] = n[j] # 키값보다 앞에 있는 놈들이 더 크면 인덱스 하나씩 뒤로 쭉 ~말어 
#             j -= 1 #그리고 비교되는 값을 하나씩 인덱스 줄여주고
#         n[j+1] = key #다 되서 만족 안되는 순간 키 값 해당 인덱스에 넣어버려
#         print(n)
#     return n


# if __name__ =="__main__":
#     print(ins_sort([2, 4, 5, 1, 3]))

#############################################################################################
    
# ## 병합 정렬
# def merge_sort(n):
#     length_n = len(n)

#     if length_n ==1:
#         return n

#     mid = length_n // 2
#     left = n[:mid]
#     right = n[mid:]

#     left1 = merge_sort(left)
#     right1 = merge_sort(right)
#     print("left1: ", left1, "right1: ", right1)
#     merge_list = merge(left1, right1)
#     print(merge_list)
#     return merge_list

# def merge(left, right):
#     i = 0
#     j = 0
#     sorted_list = []
#     print("호출")

#     while i < len(left) and j <len(right):
#         if left[i] < right[j]:
#             sorted_list.append(left[i])
#             i += 1
#         else:
#             sorted_list.append(right[j])
#             j += 1

#     while i < len(left):
#         sorted_list.append(left[i])
#         i += 1
#     while j < len(right):
#         sorted_list.append(right[j])
#         j += 1
    
#     return sorted_list

# if __name__ =="__main__":
#     print(merge_sort([8,7,6,5,4,3,2,1]))

#############################################################################################
    
## 퀵 정렬 #피봇 기준으로 작은집단, 큰집단 비교하며 정렬하는 알고리즘
def quick_sort(n):
    length_n = len(n)
    if length_n <= 1: return n

    lesser_list, equal_list, greater_list = [], [], []

    pivot = n[length_n // 2] #
    
    for num in n:
        if num < pivot:
            lesser_list.append(num)
        elif num > pivot:
            greater_list.append(num)
        else:
            equal_list.append(num)

    return quick_sort(lesser_list) + equal_list + quick_sort(greater_list)


if __name__ =="__main__":
    print(quick_sort([8,7,6,5,4,3,2,1]))