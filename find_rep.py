import numpy as np
from count_element import count_element, is_sorted

from norm import update_norm, deviation_vector_form_rep, find_rep_of_vector, norm_of_vector, update_norm_from_rep

def update_rep_in_cluster(points, newPoint, preRep, preErr, norm = 2):
    """
    It gets points, new point, pre-representor, pre-error and norm
    Return new-representor and new-error
    """
    
    points = np.concatenate((points,[newPoint]))
    
    if(norm < 2 and is_sorted(points) == 0):
        points.sort()

    count = len(points)
    newRep, newErr = preRep, preErr
    if(norm == 0):
        if(newPoint == preRep):
            return preRep, preErr

        cntNew = count_element(points,newPoint,0,len(points)) + 1
        cntPre = count - preErr - 1

        if(cntPre < cntNew):
            newRep = cntNew
            newErr = count - cntNew

    elif(norm == 1):
        newRep = points[count//2]
        '''
        Remember new value should be more than median
        '''
        if(count % 2):
            newErr = preErr + newPoint - newRep
        else:
            newErr = preErr + abs(newRep-preRep) + abs(newPoint - newRep) 

    elif(norm == 2):
        newRep = preRep + (newPoint-preRep)/count
        newErr = np.sqrt(preErr**2 + (newPoint-preRep)*(newPoint-newRep))
    
    else:
        newRep = (points.min() + newPoint) / 2
        newErr = min(newRep-points[0], newPoint - newRep)

    return newRep, newErr

def update_dev_in_cluster(points, newPoint, rep, preErr, norm = 2):
    """
    It gets points, new point, representor, pre-error and norm
    Return new-error
    """
    
    points = np.concatenate((points,[newPoint]))
    
    if(norm < 2 and is_sorted(points) == 0):
        points.sort()

    count = len(points)
    newErr = preErr
    if(norm == 0):
        if(newPoint == rep):
            return rep, preErr

        cntNew = count_element(points,newPoint,0,len(points)) + 1
        cntPre = count - preErr - 1

        if(cntPre < cntNew):
            newRep = cntNew
            newErr = count - cntNew

    elif(norm == 1):
        newRep = points[count//2]
        '''
        Remember new value should be more than median
        '''
        if(count % 2):
            newErr = preErr + newPoint - newRep
        else:
            newErr = preErr + abs(newRep-rep) + abs(newPoint - newRep) 

    elif(norm == 2):
        newRep = rep + (newPoint-rep)/count
        newErr = np.sqrt(preErr**2 + (newPoint-rep)*(newPoint-newRep))
    
    else:
        newRep = (points.min() + newPoint) / 2
        newErr = min(newRep-points[0], newPoint - newRep)

    return newRep, newErr

def find_rep_with_DP(points, K, p = 2, q = 2, printer = 0):
    N = len(points)
    if(q == 0):
        from statistics import mode
        cntMode = 1
    k = 0

    dp = np.full((K,N),-1.)
    cnt = np.full((K,N),-1)

    repP = 0
    repN = points[0]
    errP = 0
    errN = 0
    dp[0][0] = norm_of_vector(errN,p)
    count = 1
    cnt[0][0] = count
    for j in range (1,N):
        nw = points[j]
        repP = repN
        errP = errN
        count += 1
        cnt[0][j] = count
        if(q == 0):
            cntNw = count_element(points,nw,0,j+1)
            if(cntNw > cntMode):
                cntMode = cntNw
                repN = nw
            errN = count - cntMode
        elif(q == 1):
            repN = points[count//2]
            if(count % 2):
                errN += nw - repN
            else:
                errN += (repN-repP) + (nw - repN) 
        elif(q == 2):
            repN = repP + (nw-repP)/count
            errN = np.sqrt(errP**2 + (nw-repP)*(nw-repN))
        else:
            repN = (points[0] + nw) / 2
            errN = min(repN-points[0], nw - repN)
        dp[0][j] = norm_of_vector(errN,p)
        cnt[0][j] = count

    for k in range(1,K):
        dp[k][k] = norm_of_vector(0,p)
        cnt[k][k] = 1
        for i in range(k+1,N):
            dp[k][i] = dp[k-1][i-1]
            cnt[k][i] = 1
            repP = 0
            repN = points[i]
            cntMode = 1
            errP = 0
            errN = 0
            count = 1
            for j in range (1,i-k+1):
                nw = points[i-j]
                fr = i - j
                repP = repN
                errP = errN
                count += 1
                cnt[0][j] = count
                if(q == 0):
                    cntNw = count_element(points,nw,fr,i+1)
                    if(cntNw > cntMode):
                        cntMode = cntNw
                        repN = nw
                    errN = count - cntMode
                elif(q == 1):
                    repN = points[i-count//2]
                    if(count % 2):
                        errN += repP - nw
                    else:
                        errN += (repP-repN) + (repN - nw)
                elif(q == 2):
                    repN = repP + (nw-repP)/count
                    errN = np.sqrt(errP**2 + (nw-repP)*(nw-repN))
                else:
                    repN = (points[i] + nw) / 2
                    errN = min(points[i]-repN, repN - nw)
                
                valN = update_norm(dp[k-1][i-1-j], errN, p)
                if (valN < dp[k][i]):
                    dp[k][i] = valN
                    cnt[k][i] = j+1
    
    if(printer):
        print(f'DP is:\n{dp}\n')
        print(f'cnt is:\n{cnt}\n')

    def pr(k,n):
        ret = np.array([cnt[k][n]])
        if(k > 0):
            tRet = pr(k-1,int(n-ret[0]))
            ret = np.concatenate((tRet,ret))
        return ret

    cntRep = pr(K-1,N-1)

    ind = int(0)
    rep = np.zeros((K))
    for i in range(K):
        if(cntRep[i] == 0):
            continue
        indN = ind + int(cntRep[i])
        if(q == 0):
            rep[i] = mode(points[ind:indN])
        elif(q == 1):
            rep[i] = np.median(points[ind:indN])
        elif(q == 2):
            rep[i] = np.mean(points[ind:indN])
        else:
            rep[i] = (points[ind]+points[indN-1])/2

        ind = indN

    return dp[K-1][N-1], rep, cntRep

def norm_deviation_after_assign(newPoint, dev, rep, j, p, q):
    """
    get newPoint, dev vector, value of jth representor, p and q
    Compute deviation after assign newPoint to represetor j
    Return norm p of all deviation
    """
    tmpDev = dev.copy()
    tmpDev[j] = update_norm_from_rep(dev[j],newPoint,rep,q)
    return norm_of_vector(tmpDev, p)

def assign_and_find_error_and_new_rep(points, p, q, rep):
    K = len(rep)
    assignRep = {i:[] for i in range(K)}
    newRep = np.zeros(K)

    curDev = np.zeros(K)
    shuffle_i = np.arange(len(points))
    np.random.shuffle(shuffle_i)
    for i in shuffle_i:
        norm_after_assign_each_rep = list(map(lambda x: norm_deviation_after_assign(points[i],curDev,rep[x],x,p,q),range(K)))
        z = np.argmin(norm_after_assign_each_rep)
        curDev[z] = update_norm_from_rep(curDev[z],points[i],rep[z],q)
        assignRep[z].append(points[i])

    # for i in range(len(points)):
    #     z = (np.abs(points[i] - rep)).argmin()
    #     assignRep[z].append(points[i])

    newDev = np.zeros(K)

    for i in range(K):
        if(len(assignRep[i])):  
            newRep[i] = find_rep_of_vector(assignRep[i], q)
            newDev[i] = deviation_vector_form_rep(assignRep[i],newRep[i],q)

    cntRep = np.array(list(map(lambda x: len(x[1]),assignRep.items())))

    return norm_of_vector(newDev, p), newRep, cntRep

def find_rep_with_AS(points, K, p = 2, q = 2, plotter = 0):
    sRenge = points[0]
    eRange = points[-1] 
    rep = np.random.uniform(sRenge, eRange, K) 
    rep.sort()
    # rep = np.array([4.6,0,5.05])   

    if(plotter): 
        from matplotlib import pyplot as plt
        y = np.zeros_like(rep) + 0
        plt.plot(rep, y, 'x')
        y = np.zeros_like(points) + 0
        plt.plot(points, y, '.')
        plt.show()

    # err, nRep, cntRep = assign_and_find_error_and_new_rep(points,p,q,rep)
    # while(any(rep != nRep)):
        # if(q == 2):
        #     print(f'err{err}\n{rep}')
        # rep = nRep
        # err, nRep, cntRep = assign_and_find_error_and_new_rep(points,p,q,rep)
    noImp = 0
    N = len(points)
    best = assign_and_find_error_and_new_rep(points,p,q,rep)
    while(noImp < np.sqrt(N) * K**2):
        if(plotter): 
            from matplotlib import pyplot
            y = np.zeros_like(rep) + 0
            pyplot.plot(rep, y, 'x')
            y = np.zeros_like(points) + 0
            pyplot.plot(points, y, '.')
            pyplot.show()
        tmp = assign_and_find_error_and_new_rep(points,p,q,rep)
        if(tmp[0] < best[0]):
            best = tmp
            noImp = 0
        else:
            noImp = noImp + 1
        rep = tmp[1]
    # return err, rep, cntRep
    return best

def find_rep_with_AS_best(points, K, p = 2, q = 2):
    N = len(points)
    # best = (N*(points[-1]-points[0])**2,None,None)
    best = (norm_of_vector([norm_of_vector(points[-1]-points[0],q)] * K,p),None,None)
    noImp = 0
    while(noImp < np.sqrt(N) * K**2):
        # print(noImp)
        if(q == 2):
            print(f'err{noImp}')
        tmp = find_rep_with_AS(points,K,p,q,0)
        if(tmp[0] < best[0]):
            best = tmp
            noImp = 0
        else:
            noImp = noImp + 1

    return best

def find_parts(points,cntRep):
    K = len(points)
    P = [(None,None)] * K
    cntS = 0
    tPoint = np.concatenate(([-np.inf],points,[np.inf,np.inf]))

    for i in range(0,K):
        if(cntRep[i] == 0):
            P[i] = tPoint[cntS],tPoint[cntS]
        else:
            P[i] = tPoint[cntS],tPoint[cntS+cntRep[i]+(i==0)]
        cntS += cntRep[i]+(i==0)

    return P