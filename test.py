def solution(A):
    found = False
    for i in range (1,len(A)+1):
        print(i)
        if i in A:
            found = False
        else:
            found = True
            return i


print(solution( [2, 3, 1, 5]))