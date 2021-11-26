import numpy as np
from count_element import count_element, is_sorted

from norm import update_norm, deviation_vector_form_rep, find_rep_of_vector, norm_of_vector, update_norm_from_rep

def update_rep_and_err_in_cluster(points, new_point, pre_rep, pre_err, norm = 2):
    """
    It gets points, new point, pre-representor, pre-error and norm
    Return new-representor and new-error
    """
    
    points = np.concatenate((points,[new_point]))
    
    if(norm < 2 and is_sorted(points) == 0):
        points.sort()

    count = len(points)

    if(norm == 0):
        if(new_point == pre_rep):
            return pre_rep, pre_err

        new_cnt = count_element(points,new_point,0,len(points))
        pre_cnt = count - pre_err - 1

        if(pre_cnt < new_cnt):
            new_rep = new_point
            new_err = count - new_cnt
        else:
            new_rep = pre_rep
            new_err = count - pre_cnt

    elif(norm == 1):
        new_rep = points[count//2]
        '''
        Remember new value should be more than median
        '''
        if(count % 2):
            new_err = pre_err + abs(new_point - new_rep)
        else:
            new_err = pre_err + abs(new_rep-pre_rep) + abs(new_point - new_rep) 

    elif(norm == 2):
        new_rep = pre_rep + (new_point-pre_rep)/count
        new_err = np.sqrt(pre_err**2 + (new_point-pre_rep)*(new_point-new_rep))
    
    else:
        new_rep = (points.min() + points.max()) / 2
        new_err = min(new_rep-points.min(), points.max() - new_rep)

    return new_rep, new_err

def update_dev_in_cluster(points, new_point, rep, pre_err, norm = 2):
    """
    It gets points, new point, representor, pre-error and norm
    Return new-error
    """
    
    points = np.concatenate((points,[new_point]))
    
    if(norm < 2 and is_sorted(points) == 0):
        points.sort()

    count = len(points)
    new_err = pre_err
    if(norm == 0):
        if(new_point == rep):
            return rep, pre_err

        new_cnt = count_element(points,new_point,0,len(points)) + 1
        pre_cnt = count - pre_err - 1

        if(pre_cnt < new_cnt):
            new_rep = new_cnt
            new_err = count - new_cnt

    elif(norm == 1):
        new_rep = points[count//2]
        '''
        Remember new value should be more than median
        '''
        if(count % 2):
            new_err = pre_err + new_point - new_rep
        else:
            new_err = pre_err + abs(new_rep-rep) + abs(new_point - new_rep) 

    elif(norm == 2):
        new_rep = rep + (new_point-rep)/count
        new_err = np.sqrt(pre_err**2 + (new_point-rep)*(new_point-new_rep))
    
    else:
        new_rep = (points.min() + new_point) / 2
        new_err = min(new_rep-points[0], new_point - new_rep)

    return new_rep, new_err

def find_rep_with_DP(points, K, p = 2, q = 2, printer = 0):
    N = len(points)
    if(q == 0):
        from statistics import mode
    k = 0

    dp = np.full((K,N),-1.)
    cnt = np.full((K,N),-1)

    pre_rep = 0
    new_rep = points[0]
    pre_err = 0
    new_err = 0
    dp[0][0] = norm_of_vector(new_err,p)
    count = 1
    cnt[0][0] = count
    for j in range (1,N):
        new_point = points[j]
        count += 1
        cnt[0][j] = count
        pre_rep = new_rep
        pre_err = new_err
        new_rep,new_err = update_rep_and_err_in_cluster(points[0:j],new_point,pre_rep,pre_err,q)
        dp[0][j] = norm_of_vector(new_err,p)
        cnt[0][j] = count

    for k in range(1,K):
        dp[k][k] = norm_of_vector(0,p)
        cnt[k][k] = 1
        for i in range(k+1,N):
            dp[k][i] = dp[k-1][i-1]
            cnt[k][i] = 1
            pre_rep = 0
            new_rep = points[i]

            pre_err = 0
            new_err = 0
            count = 1
            for j in range (1,i-k+1):
                new_point = points[i-j]
                pre_rep = new_rep
                pre_err = new_err
                count += 1
                cnt[0][j] = count

                new_rep,new_err = update_rep_and_err_in_cluster(points[i-j+1:i+1],new_point,pre_rep,pre_err,q)
                
                valN = update_norm(dp[k-1][i-1-j], new_err, p)
                if (valN < dp[k][i]):
                    dp[k][i] = valN
                    cnt[k][i] = j+1
    
    if(printer):
        print(f'DP is:\n{dp}\n')
        print(f'cnt is:\n{cnt}\n')

    def pr(k,n):
        ret = np.array([cnt[k][n]])
        if(k > 0):
            tmp_ret = pr(k-1,int(n-ret[0]))
            ret = np.concatenate((tmp_ret,ret))
        return ret

    size_rep = pr(K-1,N-1)

    ind = int(0)
    rep = np.zeros((K))
    for i in range(K):
        if(size_rep[i] == 0):
            continue
        indN = ind + int(size_rep[i])
        if(q == 0):
            rep[i] = mode(points[ind:indN])
        elif(q == 1):
            rep[i] = np.median(points[ind:indN])
        elif(q == 2):
            rep[i] = np.mean(points[ind:indN])
        else:
            rep[i] = (points[ind]+points[indN-1])/2

        ind = indN

    return dp[K-1][N-1], rep, size_rep

def norm_deviation_after_assign(new_point, dev, rep, j, p, q):
    """
    get new_point, dev vector, value of jth representor, p and q
    Compute deviation after assign new_point to represetor j
    Return norm p of all deviation
    """
    tmp_dev = dev.copy()
    tmp_dev[j] = update_norm_from_rep(dev[j],new_point,rep,q)
    return norm_of_vector(tmp_dev, p)

def assign_and_find_error_and_new_rep(points, p, q, rep):
    K = len(rep)
    assign_to_rep = {i:[] for i in range(K)}
    new_rep = np.zeros(K)
    rep.sort()
    cur_dev = np.zeros(K)
    shuffle_i = np.arange(len(points))
    np.random.shuffle(shuffle_i)
    for i in shuffle_i:
        norm_after_assign_each_rep = list(map(lambda x: norm_deviation_after_assign(points[i],cur_dev,rep[x],x,p,q),range(K)))
        z = np.argmin(norm_after_assign_each_rep)
        cur_dev[z] = update_norm_from_rep(cur_dev[z],points[i],rep[z],q)
        assign_to_rep[z].append(points[i])

    # for i in range(len(points)):
    #     z = (np.abs(points[i] - rep)).argmin()
    #     assign_to_rep[z].append(points[i])

    new_dev = np.zeros(K)

    for i in range(K):
        if(len(assign_to_rep[i])):  
            new_rep[i] = find_rep_of_vector(assign_to_rep[i], q)
            new_dev[i] = deviation_vector_form_rep(assign_to_rep[i],new_rep[i],q)

    size_rep = np.array(list(map(lambda x: len(x[1]),assign_to_rep.items())))

    return norm_of_vector(new_dev, p), new_rep, size_rep

def find_rep_with_AS(points, K, p = 2, q = 2, plotter = 0):
    st_renge = points[0]
    en_range = points[-1] 
    rep = np.random.uniform(st_renge, en_range, K) 
    rep.sort()
    # rep = np.array([4.6,0,5.05])   

    if(plotter): 
        from matplotlib import pyplot as plt
        y = np.zeros_like(rep) + 0
        plt.plot(rep, y, 'x')
        y = np.zeros_like(points) + 0
        plt.plot(points, y, '.')
        plt.show()

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
    return best

def find_rep_with_AS_best(points, K, p = 2, q = 2):
    N = len(points)

    best = find_rep_with_AS(points,K,p,q,0)
    noImp = 0

    while(noImp < np.sqrt(N) * K**2):

        tmp = find_rep_with_AS(points,K,p,q,0)
        if(tmp[0] < best[0]):
            best = tmp
            noImp = 0
        else:
            noImp = noImp + 1

    return best

def find_parts(points,size_rep):
    K = len(points)
    P = [(None,None)] * K
    cnt_points = 0
    tmp_point = np.concatenate(([-np.inf],points,[np.inf,np.inf]))

    for i in range(0,K):
        if(size_rep[i] == 0):
            P[i] = tmp_point[cnt_points],tmp_point[cnt_points]
        else:
            P[i] = tmp_point[cnt_points],tmp_point[cnt_points+size_rep[i]+(i==0)]
        cnt_points += size_rep[i]+(i==0)

    return P