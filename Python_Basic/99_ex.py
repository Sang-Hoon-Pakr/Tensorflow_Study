# # class Symbol(object):
# #     def __init__(self, value):
# #         self.value = value
# #         print(value) #출력 순서 xPy, yPy

# # if __name__ == "__main__":
# #     x = Symbol('xPy')
# #     y = Symbol('yPy')

# #     symbol = set()
# #     symbol.add(x)
# #     symbol.add(y)
# # def logger(func):
# #     print("logger start")
# #     def wrapper(*args, **kwargs):
# #         print("wrapper start")
# #         result = func(*args, **kwargs)
# #         print("func end")
# #         print('Result: ', result)
# #         return result
# #         print("logger end")
# #     return wrapper

# # @logger
# # def add(a, b):
# #     print("add start")
# #     return a+b

# # if __name__ == "__main__":
# #     reslt = add(20, 30)
# #     print(reslt)

# class aa:
#     def __init__(self):
#         self.data = 10
#         print("init success")

#     def call(self):
#         print("helloworld")

# if __name__=="__main__":
#     a= aa()


# def solutions(serialization):
#     root = serialization.pop(0)
#     print(root)
#     if root == -1:
#         if serialization: return False # -1이 있는데 더 남아있는 노드가 있다면 틀린거로 간주
#         else: return True # -1 만 있는 경우 True 하는거임 
#     else:
#         root_list = []
#         root_list.append(root)
#         print(root_list)
#         tree_dict = {}
#         cur_root = root_list[-1] #맨 뒤에 쌓인 것 
#         tree_dict[cur_root] =[]
#         print(tree_dict)

#         is_done = False #종료여부 판단
#         while serialization and root_list:
#             cur_value = serialization[0]
#             print("cur_value: ", cur_value)
#             cur_root = root_list[-1]
#             print("cur_root: ", cur_root)
#             print("tree_dict: ", tree_dict)
#             if cur_root not in tree_dict:
#                tree_dict[cur_root] =[] #키값 생성 value 값은 없는걸로 
#                #dict 의 key 값을 생성하는것 

#             while len(tree_dict[cur_root]) == 2: #현재 루트의 자식노드가 두개가 다 있는지 확인 하는 것 있으면 cur_root 변경을 통해 다시 검사 검사 하는거
#                 root_list.pop()
#                 print("root_leist22: ", root_list)
#                 print("len_tree_dict: ", len(tree_dict[cur_root]))
#                 if not root_list:
#                     is_done = True
#                     break
#                 cur_root = root_list[-1]
#             if is_done:
#                 break

#             tree_dict[cur_root].append(cur_value)
#             if len(tree_dict[cur_root]) == 2:
#                 root_list.pop()
#                 print("root_leist3: ", root_list)
            
#             if cur_value > 0:
#                 root_list.append(cur_value)

#             serialization.pop(0)
#             print("serialization: ",serialization)

#         if root_list: 
#             return False
#         if serialization: return False
#         else: return True




# if __name__ == "__main__":
#     print(solutions([3,5,6,8,-1,-1,-1,1,7,-1,-1,-1,4,-1,2,-1,-1]))


###############################################################################
# def solutions(s):
#         # substring길이: 길이 ~ 1
#     print("전체길이 :", len(s))
#     for i in range(len(s),0,-1):
#         print("i :", i)
#                 # 시작점
#         for j in range(len(s)-i+1):
#             print("공식: ", len(s)-i+1)
            
#             print("j :",j)
#             print("s[j:j+i]: ", s[j:j+i])
#             if s[j:j+i] == s[j:j+i][::-1]:
#                 print("앞부분: ", s[j:j+i])
#                 print("뒷부분: ", s[j:j+i][::-1])

#                 return i

# if __name__ == "__main__":
#     #s= "abaabaabaabadabaabaabaabaabbbaabbbaabbbaabbbaabbbaabbbaabbba"
#     s= 'adbaazzz'
#     print("결과: ", solutions(s))

############################################################
# from itertools import product

# def solutions(numbers):
#     answer = 0
#     pairs = [(n, -n) for n in numbers]
#     print(pairs)
#     prods = list(product(*pairs))
#     print(*pairs)
#     for prod in prods:
#         if sum(prod) == 0:
#             answer += 1
#     return answer%100000

# if __name__ == "__main__":

#     print(solutions([1,2,3,4]))
###############################################
def solutions(N):
    rank_list=[]
    for i in N:
        rank = 1
        for j in N:
            if j>i:
                rank+=1
        rank_list.append(rank)

    print(rank_list)
     


if __name__ == "__main__":
    N=range(1,3,1)
    print(list(N))
    solutions(N)