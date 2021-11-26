from math import sqrt

def find_k_with_elbow(points, findRep, p=2, q=2, plotter=0):
    N = len(points)

    rangeK = int(1.5*sqrt(N)+1)

    errInK = [(None,None)] * rangeK
    for i in range(1,rangeK+1):
        errInK[i-1] = (i,findRep(points=points, K=i, p=p, q=q)[0])

    def dis_point_from_line(x1, y1, xn, yn, x0, y0) :
        return abs ((xn-x1)*(y1-y0)-(x1-x0)*(yn-y1)) / (sqrt((xn-x1)**2 + (yn-y1)**2))
    
    x1, xn, y1, yn = errInK[0][0], errInK[-1][0], errInK[0][1], errInK[-1][1]
    
    dis_from_line = [None] * rangeK
    for i in range(rangeK):
        dis_from_line[i] = dis_point_from_line(x1, y1, xn, yn, errInK[i][0], errInK[i][1])
    
    if(plotter):
        from matplotlib import pyplot as plt
        zipped = list(zip(*errInK))
        plt.plot(zipped[0],zipped[1])
        plt.plot([x1,xn],[y1,yn],'-r')
        plt.show()
        plt.plot(list(zip(*errInK))[0],dis_from_line)
        plt.show()
    
    from numpy.core.fromnumeric import argmax
    K = argmax(dis_from_line)+1

    return K

