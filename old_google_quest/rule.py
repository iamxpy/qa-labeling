import numpy as np
import pandas as pd
import nltk, re
from nltk.stem.snowball import EnglishStemmer

df=pd.read_csv('test.csv')


# stemming
df['title_split'] = df['question_title'].apply(nltk.word_tokenize)
df['body_split'] = df['question_body'].apply(nltk.word_tokenize)
df['answer_split'] = df['answer'].apply(nltk.word_tokenize)

unique_words_title = {word for sentence in df.title_split.values for word in sentence}
unique_words_body = {word for sentence in df.body_split.values for word in sentence}
unique_words_answer = {word for sentence in df.answer_split.values for word in sentence}

unique_words=unique_words_title|unique_words_body|unique_words_answer

stopwords = nltk.corpus.stopwords.words('english') + ['']

stemmer = EnglishStemmer()

stemmer_dict = {u: stemmer.stem(u) for u in unique_words}

df['title_stemmed'] = df['title_split'].apply(lambda x: [stemmer_dict[y] for y in x if re.sub('[^a-z]+','',y.lower()) not in stopwords])


df['body_stemmed'] = df['body_split'].apply(lambda x: [stemmer_dict[y] for y in x if re.sub('[^a-z]+','',y.lower()) not in stopwords])


df['answer_stemmed'] = df['answer_split'].apply(lambda x: [stemmer_dict[y] for y in x if re.sub('[^a-z]+','',y.lower()) not in stopwords])

# 统计
def gen_word_features(df,word):
    df[f'title_n_{word}']=df.title_stemmed.apply(lambda x: x.count(word))
    df[f'body_n_{word}']=df.body_stemmed.apply(lambda x: x.count(word))
    df[f'answer_n_{word}']=df.answer_stemmed.apply(lambda x: x.count(word))
    df[f'n_{word}']=df[f'title_n_{word}']+df[f'body_n_{word}']+df[f'answer_n_{word}']


words=['mean','definit','syllabl','pronunci','pronounc','sound','grammar','grammat','noun','pronoun','verb','adject','adverb','preposit','conjunct']

for word in words:
    gen_word_features(df, word)

# sub是前面得到用于提交的dataframe
df=pd.concat([df,sub[['question_type_spelling']]],axis=1)

# 应用规则

df.loc[:,'question_type_spelling']=0.0

cond1=(df.category=='CULTURE')&((df.host=='english.stackexchange.com')|(df.host=='ell.stackexchange.com'))

cond2=(df['n_syllabl']>0) | (df['n_pronounc']>0) | (df['n_pronunci']>0) | (df['n_sound']>0)

# cond3=(df['n_grammar']>0) | (df['n_grammat']>0) | (df['n_noun']>0) | (df['n_pronoun']>0) | (df['n_verb']>0) | (df['n_adject']>0) | (df['n_adverb']>0) | (df['n_preposit']>0) | (df['n_conjunct']>0) | (df['n_mean']>0) | (df['n_definit']>0)

df.loc[cond1 & cond2, 'question_type_spelling']=1.0

# df.loc[cond1 & cond3, 'question_type_spelling'] = -1

sub.loc[:,'question_type_spelling'] += df.question_type_spelling

sub.loc[:,'question_type_spelling'] /= 2

print("Rules applied!")

