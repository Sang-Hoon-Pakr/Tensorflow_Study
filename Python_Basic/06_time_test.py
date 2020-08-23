# 시간 측정하기
import time

def test1():
    t0 = time.time()
    l=[]
    for i in range(100000):
        l= l+[i]
    t1 = time.time()
    return t1-t0

def test2():
    t0 = time.time()
    l=[]
    for i in range(100000):
        l.append(i)
    t1 = time.time()
    return t1-t0

def test3():
    t0 = time.time()
    l = [i for i in range(100000)]
    t1 = time.time()
    return t1-t0

def test4():
    t0 = time.time()
    l = list(range(10000))
    t1 = time.time()
    return t1-t0

if __name__=="__main__":
    print("Concat time: ", test1())
    print("Append time: ", test2())
    print("Comprehension time: ", test3())
    print("list range time: ", test4())


######## Results ########
# Concat time:  18.64295744895935
# Append time:  0.006012916564941406
# Comprehension time:  0.002001047134399414
# list range time:  0.0020012855529785156