from math import sqrt

def find_k_with_elbow(points,findRep,printer=0,**kwargs):
    N = len(points)

    rangeK = int(1*sqrt(N))

    errInK = [(None,None)] * rangeK
    for i in range(1,rangeK+1):
        errInK[i-1] = (i,findRep(points=points,K=i,**kwargs)[0])

    unzipped = list(zip(*errInK))

    if(printer):
        print(unzipped)

    x1, xn, y1, yn = errInK[0][0], errInK[-1][0], errInK[0][1], errInK[-1][1]

    def pointToLine(x1, y1, xn, yn, x0, y0) :
        return abs ((xn-x1)*(y1-y0)-(x1-x0)*(yn-y1)) / (sqrt((xn-x1)**2 + (yn-y1)**2))
    
    disToLine = [None] * rangeK
    for i in range(rangeK):
        disToLine[i] = pointToLine(x1, y1, xn, yn, errInK[i][0], errInK[i][1])
    
    if(printer):
        from matplotlib import pyplot as plt
        plt.plot(unzipped[0],disToLine)
        plt.plot([x1,xn],[y1,yn],'-ro')
        plt.show()
    
    from numpy.core.fromnumeric import argmax
    K = argmax(disToLine)+1

    return K

