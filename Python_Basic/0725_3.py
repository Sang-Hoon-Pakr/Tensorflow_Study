from collections import defaultdict

def deleteBlock(board):
    n = len(board)
    count = 0
    
    for i in range(n): #행 
        count += len(board[i])
        for j in range(n):

            if i ==0 and i !=j and j > n-1:
                board[i][j] == board[i][j+1]:
        if i > 0:
            for j in range (n):
                if board[i][j] == board[i-1][j]:
                    count += 1
                    if j == 0:
                        board[i][j] == board[i][j+1]


                

                if i > 0 and i < n and j > 0 and j < n
                    if (board[i][j] == board[i-1][j]) or (board[i][j] == board[i+1][j]) or (board[i][j] == board[i][j-1]) or (board[i][j] == board[i][j+1]):
                        print("1")
                elif 

def solution(aaa):
    deleteBlock(aaa)

if __name__=="__main__":
    aaa=["ABBBBC", "AABAAC", "BCDDAC", "DCCDDE", "DCCEDE", "DDEEEB"]
    solution(aaa)