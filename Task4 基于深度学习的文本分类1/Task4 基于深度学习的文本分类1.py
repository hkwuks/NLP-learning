import pandas as pd
from sklearn.metrics import f1_score

#转换为fasttext需要的格式
train_df=pd.read_csv('./datalab/train_set.csv',sep='\t',nrows=15000)
train_df['label_ft']='__label__'+train_df['label'].astype(str)
train_df[['text','label_ft']].iloc[:-5000].to_csv('train.csv',index=None,header=None,sep='\t')

import fasttext
model=fasttext.train_supervised('train.csv',lr=1.0,wordNgrams=2,verbose=2,minCount=1,epoch=25,loss="hs")

val_pred=[model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[-5000:]['text']]
print(f1_score(train_df['label'].values[-5000:].astype(str),val_pred,average='macro'))