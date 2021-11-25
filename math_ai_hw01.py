import numpy as np

from my_io import build_testcase, dataframe_to_docx_table, read_dataset_with_pandas, read_testcase, read_dataset, string_to_dataframe
from find_rep import find_rep_with_AS, find_rep_with_AS_best, find_rep_with_DP
# build_testcase(N = 5, p=0, q=1, Range= 10)
# p, q, points = read_testcase("dataset/HW1DS.txt",1)
# points.sort()
# points = np.array(points)
# K = 2
# resultDP = find_rep_with_DP(points,K,p,q)
# resultAS = find_rep_with_AS_best(points,K,p,q)
# print(resultDP)
# print(resultAS)
# print(type(resultDP[2]))
# print(type(resultAS[2]))
# print(f'{round(resultDP[0],3)}/{round(resultAS[0],3)}')
# exit()

from find_k import find_k_with_elbow

# K = find_k_with_elbow(points,find_rep_with_DP,p=p,q=q)

# print(K)

# exit()

import docx

doc = docx.Document()
from docx.enum.table import WD_TABLE_ALIGNMENT

for atr in range(4):
    data = ""
    colName, points = read_dataset_with_pandas("dataset/iris.csv",0)
    points = points.to_numpy()
    points = points.ravel()
    points = np.sort(points, kind='mergesort')

    # points = read_dataset("dataset/iris.data",atr)
    # points.sort()
    # points = np.array(points)

    ind = (0,1,2,np.inf)
    for j in range(len(ind)):
        data += f'"{ind[j]}",'
    data = data[:-1] + '\n'

    for i, p in enumerate(ind):
        for j, q in enumerate(ind):
            K = 3
            # K = find_k_with_elbow(points,find_rep_with_DP,p=p,q=q)
            print(f'K is {K} in atr={atr} and p={p} and q={q}')
            resultDP = find_rep_with_DP(points,K,p,q)
            if(p == 2 and q == 0):
                print('hi')
            if(p == np.inf and q == 0):
                print('hi')
            # resultAS = find_rep_with_AS_best(points,K,p,q)
            resultAS = find_rep_with_AS(points,K,p,q)
            print(f'DP:{round(resultDP[0],3)}\tAS:{round(resultAS[0],3)}')
            if(round(resultDP[0],6) > round(resultAS[0],6)):
                print(f'Shit16 :) dp{round(resultDP[0],6)} / as{round(resultAS[0],6)}')
            print('\n')
            data += f'DP:{round(resultDP[0],3)}/AS:{round(resultAS[0],3)},'
            # data += f'K:{K}/DP:{round(resultDP[0],3)}/AS:{round(resultAS[0],3)}'
        data = data[:-1] + '\n'

    data = string_to_dataframe(data)
    data.index = ind

    header = f'Result on attribute {atr+1} of iris dataset'
    doc = dataframe_to_docx_table(header,data,'docs/hw01.docx',doc,save=(atr==3))




    

