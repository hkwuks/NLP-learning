#数据读取
import pandas as pd
train_df=pd.read_csv('./datalab/train_set.csv',sep='\t')
print(train_df.head())

#句子长度分析
train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))
print(train_df['text_len'].describe())

#下图将句子长度绘制了直方图，可见大部分句子的长度都几种在2000以内。
import matplotlib.pyplot as plt
_ = plt.hist(train_df['text_len'], bins=200)
plt.xlabel('Text char count')
plt.title("Histogram of char count")
plt.show()
#新闻类别分布
train_df['label'].value_counts().plot(kind='bar')
plt.title('News class count')
plt.xlabel("category")
plt.show()

#字符分布统计
#统计每个字符出现的次数
from collections import Counter
all_lines = ''.join(list(train_df['text']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:d[1], reverse = True)

print(len(word_count))
# 6869

print(word_count[0])
# ('3750', 7482224)

print(word_count[-1])
# ('3133', 1)

#下面代码统计了不同字符在句子中出现的次数，其中字符3750，字符900和字符648在20w新闻的覆盖率接近99%，很有可能是标点符号。
train_df['text_unique'] = train_df['text'].apply(lambda x: ' '.join(list(set(x.split(' ')))))
all_lines = ' '.join(list(train_df['text_unique']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:int(d[1]), reverse = True)

print(word_count[0])
# ('3750', 197997)

print(word_count[1])
# ('900', 197653)

print(word_count[2])
# ('648', 191975)

#本章作业

#1.假设字符3750，字符900和字符648是句子的标点符号，请分析赛题每篇新闻平均由多少个句子构成？
import re
train_df['sentence_num']=train_df['text'].apply(lambda x:len(re.split('3750|900|648',x)))
print('每篇新闻平均包含句子数量：',sum(train_df['sentence_num'])/len(train_df['sentence_num']))

#2.统计每类新闻中出现次数最多的字符
all_lines_in_a_class=[]
for i in range(14):
    line=' '.join(train_df[train_df['label']==i]['text'])
    all_lines_in_a_class.append(re.sub('3750|900|648','',line))#将标点替换成''

for i,line in enumerate(all_lines_in_a_class):
    word_count=Counter(line.split(' '))
    word_count=sorted(word_count.items(),key=lambda d:int(d[1]),reverse=True)
    print('新闻种类',i,':',word_count[1])