import numpy as np
from find_k import find_k_with_elbow
import docx
from my_io import build_testcase, dataframe_to_docx_table, read_dataset_with_pandas, read_testcase, string_to_dataframe
from find_rep import find_rep_with_AS, find_rep_with_DP

doc = docx.Document()

while(1):
    sum = 0

    for atr in range(4):
        data = ""
        colName, points = read_dataset_with_pandas("dataset/iris.csv",atr)
        points = points.to_numpy()
        points = points.ravel()
        points = np.sort(points, kind='mergesort')

        ind = (0,1,2,np.inf)
        for j in range(len(ind)):
            data += f'"{ind[j]}",'
        data = data[:-1] + '\n'

        for i, p in enumerate(ind):
            for j, q in enumerate(ind):
                K = 3
                # K = find_k_with_elbow(points,find_rep_with_DP,p,q)
                print(f'K is {K} in atr={atr} and p={p} and q={q}')
                resultDP = find_rep_with_DP(points,K,p,q)
                resultAS = find_rep_with_AS(points,K,p,q)
                print(f'DP:{round(resultDP[0],3)}\tAS:{round(resultAS[0],3)}')
                print('\n')
                sum += resultAS[0] / resultDP[0]
                data += f'DP:{round(resultDP[0],3)}/AS:{round(resultAS[0],3)},'
                # data += f'K:{K}/DP:{round(resultDP[0],3)}/AS:{round(resultAS[0],3)}'
            data = data[:-1] + '\n'

        data = string_to_dataframe(data)
        data.index = ind

        header = f'Result on attribute {atr+1} of iris dataset'
        if(atr == 3):
            doc = dataframe_to_docx_table(header,data,f'report/hw01_{sum-64}.docx',doc,save=1)
            doc = docx.Document()
        else:
            doc = dataframe_to_docx_table(header,data,f'report/hw01_{sum-64}.docx',doc,save=0)


    

