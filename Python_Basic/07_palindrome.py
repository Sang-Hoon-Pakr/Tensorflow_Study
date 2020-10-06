def is_palindrome(s):
    l = s.split(" ")
    s2 = "".join(l)
    return s2 == s2[::-1]

def is_palindrome2(s):
    l = len(s)
    f, b = 0, l-1
    while f <l // 2:
        while s[f] ==" ":
            f += 1
        while s[b] ==" ":
            b -= 1
        if s[f] != s[b]:
            return False
        f += 1
        b -= 1
    return True

def is_palindrome3(s):
    s = s.split()
    if len(s) < 2:
        return True
    if s[0] == s[-1]:
        return is_palindrome(s[1:-1])
    else:
        return False

if __name__ == "__main__":
    str1 = "다시합창합시다가나"
    str2 = ""
    str3 = "hello"
    
    print(is_palindrome(str1))
    print(is_palindrome(str2))
    print(is_palindrome(str3))

    print(is_palindrome2(str1))
    print(is_palindrome2(str2))
    print(is_palindrome2(str3))

    print(is_palindrome3(str1))
    print(is_palindrome3(str2))
    print(is_palindrome3(str3))
