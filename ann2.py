def and_not(x1,x2):
    
    
    y= []
    w1 = 1
    w2 = -1
    theta = 1

    #AND NOT Function

    for i in range(0,4):
        sum = x1[i]*w1 + x2[i]*w2
        if(sum>=theta):
            y.append(1)
        else:
            y.append(0)
    
    print(x1)
    print(x2)
    print(y)
    
x1 = [0,0,1,1]
x2 = [0,1,0,1]
and_not(x1,x2)
