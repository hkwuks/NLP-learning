import loggging
import random

import numpy as np
import torch

loggging.basicConfig(level=loggging.INFO,format='%(asctime)-15s %(levelname)s: %(message)s')

#set seed
seed=666
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

#split data to 10 fold
fold_num=10
data_file='../datalab/train_set.csv'
import pandas as pd

def all_data2fold(fold_num,num=10000):
    fold_data=[]
    f=pd.read_csv(data_file,sep='\t',encoding='utf-8')
    texts=f['text'].tolist()[:num]
    labels=f['label'].tolist()[:num]
    
    total=len(labels)
    
    index=list(range((total)))
    np.random.shuffle(index)#打乱

    all_texts=[]
    all_labels=[]
    for i in index:
        all_texts.append(texts[i])#append() 方法用于在列表末尾添加新的对象
        all_labels.append(label[i])
    
    label2id={}#{'label':[x,y,z,...]}
    for i in range(total):
        label=str(all_labels[i])
        if label not in label2id:
            label2id[label]=[i]
        else:
            label2id[label].append(i)
        
        all_index=[[] for _ in range(fold_num)]