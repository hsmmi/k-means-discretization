import numpy as np

from norm import update_norm, deviation_vector_form_rep, find_rep_of_vector, norm_of_vector

def find_rep_with_DP(points, K, p = 2, q = 2, printer = 0):
    N = len(points)
    if(q == 0):
        from statistics import mode

        cntMode = 1

        def countEl(arr, x, st, en):
            i = first(arr, st, en-1, x, st, en)
            if i == -1:
                return i
            j = last(arr, i, en-1, x, st, en);    
            return j-i+1
        
        def first(arr, low, high, x, st, en):
            if high >= low:
                mid = (low + high)//2     
                if (mid == st or x > arr[mid-1]) and arr[mid] == x:
                    return mid
                elif x > arr[mid]:
                    return first(arr, (mid + 1), high, x, st, en)
                else:
                    return first(arr, low, (mid -1), x, st, en)
            return -1
        
        def last(arr, low, high, x, st, en):
            if high >= low:
                mid = (low + high)//2
                if(mid == en-1 or x < arr[mid+1]) and arr[mid] == x :
                    return mid
                elif x < arr[mid]:
                    return last(arr, low, (mid -1), x, st, en)
                else:
                    return last(arr, (mid + 1), high, x, st, en)    
            return -1

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
            cntNw = countEl(points,nw,0,j+1)
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
                    cntNw = countEl(points,nw,fr,i+1)
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

def assign_and_find_error_and_new_rep(points, p, q, rep):
    K = len(rep)
    assignRep = {i:[] for i in range(K)}
    dev = np.zeros(K)

    for i in range(len(points)):
        z = (np.abs(points[i] - rep)).argmin()
        assignRep[z].append(points[i])
    
    nRep = np.array(rep)
    for i in range(K):
        if(len(assignRep[i])):  
            dev[i] = deviation_vector_form_rep(assignRep[i],rep[i],q)

            nRep[i] = find_rep_of_vector(assignRep[i], q)
    cntRep = list(map(lambda x: len(x[1]),assignRep.items()))
    return norm_of_vector(dev, p), nRep, cntRep

def find_rep_with_AS(points, K, p = 2, q = 2, printer = 0):
    rep = np.zeros(K)
    sRenge = points[0]
    eRange = points[-1] 
    for i in range(K):
        rep[i] = np.random.uniform(sRenge, eRange)
    rep.sort()

    if(printer): 
        from matplotlib import pyplot as plt
        y = np.zeros_like(rep) + 0
        plt.plot(rep, y, 'x')
        y = np.zeros_like(points) + 0
        plt.plot(points, y, '.')
        plt.show()

    err, nRep, cntRep = assign_and_find_error_and_new_rep(points,p,q,rep)
    while(any(rep != nRep)):
        rep = nRep
        if(printer): 
            from matplotlib import pyplot
            y = np.zeros_like(rep) + 0
            pyplot.plot(rep, y, 'x')
            y = np.zeros_like(points) + 0
            pyplot.plot(points, y, '.')
            pyplot.show()
        
        err, nRep, cntRep = assign_and_find_error_and_new_rep(points,p,q,rep)
    return err, rep, cntRep

def find_rep_with_AS_best(points, K, p = 2, q = 2):
    N = len(points)
    best = (N*(points[-1]-points[0])**p,None,None)
    noImp = 0
    while(noImp < np.sqrt(N) * K):
        tmp = find_rep_with_AS(points,K,p,q,printer=0)
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