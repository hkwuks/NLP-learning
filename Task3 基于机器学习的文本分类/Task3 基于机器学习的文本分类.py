#Bag of Words(Count Vectors)
from sklearn.feature_extraction.text import CountVectorizer
corpus=[
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?'
]
vectorizer=CountVectorizer()
print(vectorizer.fit_transform(corpus).toarray())

#基于机器学习的文本分类

#Count Vectors + RidgeClassifier
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

train_df=pd.read_csv('./datalab/train_set.csv',sep='\t',nrows=15000)

vectorizer=CountVectorizer(max_features=3000)
train_test=vectorizer.fit_transform(train_df['text'])

clf=RidgeClassifier()
clf.fit(train_test[:10000],train_df['label'].values[:10000])

val_pred=clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:],val_pred,average='macro'))

#TF-IDF + RidgeClassifier
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

train_df=pd.read_csv('./datalab/train_set.csv',sep='\t',nrows=15000)

tfidf=TfidfVectorizer(ngram_range=(1,3),max_features=3000)
train_test=tfidf.fit_transform(train_df['text'])

clf=RidgeClassifier()
clf.fit(train_test[:10000],train_df['label'].values[:10000])

val_pred=clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:],val_pred,average='macro'))