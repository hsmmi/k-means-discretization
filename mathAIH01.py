import numpy as np

from myIO import buildTestcase, dataframe_to_docx_table, read_dataset_with_pandas, readTestcase, readDataset, string_to_dataframe
from findRep import find_rep_with_AS, find_rep_with_AS_best, find_rep_with_DP
# buildTestcase(N = 5, p=0, q=1, Range= 10)
# p, q, points = readTestcase("HW1DS.txt",1)
# points.sort()
# points = np.array(points)
# K = 2
# resultDP = find_rep_with_DP(points,K,p,q)
# resultAS = find_rep_with_AS_best(points,K,p,q)
# print(f'{round(resultDP[0],3)}/{round(resultAS[0],3)}')

from findK import find_k_with_elbow

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

    # points = readDataset("dataset/iris.data",atr)
    # points.sort()
    # points = np.array(points)

    ind = (0,1,2,np.inf)
    for j in range(len(ind)):
        data += f'"{ind[j]}",'
    data = data[:-1] + '\n'

    for i, p in enumerate(ind):
        for j, q in enumerate(ind):
            K = find_k_with_elbow(points,find_rep_with_DP,p=p,q=q)
            # K = 3
            print(f'K is {K} in atr={atr} and p={p} and q={q}')
            resultDP = find_rep_with_DP(points,K,p,q)
            resultAS = find_rep_with_AS_best(points,K,p,q)
            data += f'K:{K}/DP:{round(resultDP[0],3)}/AS:{round(resultAS[0],3)},'
        data = data[:-1] + '\n'

    data = string_to_dataframe(data)
    data.index = ind

    header = f'Result on attribute {atr+1} of iris dataset'
    doc = dataframe_to_docx_table(header,data,'docs/hw01.docx',doc,save=(atr==3))




    

