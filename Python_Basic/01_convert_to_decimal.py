#진법 변환하는 알고리즘
# base진수 number를 10진수 표기로 변경하는 것

def convert_to_decimal(number, base):
    multiplier, result = 1, 0
    while number > 0:
        result += number % 10 * multiplier
        multiplier *= base
        number = number //10
    return result

def test_convert_to_decimal():
    number, base = 1001, 2 # 10진수 1001을 2진수로 변경하는 것
    print(convert_to_decimal(number, base))
    assert(convert_to_decimal(number, base) == 9)
    print("Test Success !! ")

if __name__ == "__main__":
    test_convert_to_decimal()
