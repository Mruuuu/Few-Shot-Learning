'''
Author: Yen-Ju Chen  mru.11@nycu.edu.tw
Date: 2023-03-14 20:39:05
LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
LastEditTime: 2023-04-11 11:40:04
FilePath: /mru/Few-Shot-Learning/helper/view_performance.py
Description: 

'''
import os
path = "./logs/tune_classical_fsl_seed"


acc=list()

for f in os.listdir(path):
    
    lines = [line.strip() for line in open(os.path.join(path, f, "train_record.txt"))]
    flag = False
    for line in lines[::-1]:
        
        if "saving model..." in line:
            flag = True
            continue
        if flag:
            # print(f, line)
            if float(line[-12:-6]) < 1:
                acc.append([f, float(line[-12:-6])])
            break

for i in sorted(acc, key = lambda student : student[1], reverse=True):
    print(i)
