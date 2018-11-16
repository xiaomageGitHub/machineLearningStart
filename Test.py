#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import numpy as np

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
#
# # 特征提取
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(twenty_train.data)
# print(X_train_counts.shape)
# print(count_vect.vocabulary_.get(u'algorithm'))
#
# # tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
# # X_train_tf = tf_transformer.transform(X_train_counts)
# # X_train_tf.shape
#
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# print(X_train_tfidf.shape)
#
# # naive bayes分类器
# clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
#
# # 测试集分类
# docs_new = ['God is love', 'OpenGL on the GPU is fast']
#
# # 测试集特征抽取
# X_new_counts = count_vect.transform(docs_new)
# X_new_tfidf = tfidf_transformer.transform(X_new_counts)
#
# # 分类
# predicted = clf.predict(X_new_tfidf)
#
# for doc, category in zip(docs_new, predicted):
#     print('%r => %s' % (doc, twenty_train.target_names[category]))

print("-------------------------------------------------------------")
twenty_test = fetch_20newsgroups(subset='test',
                                 categories=categories, shuffle=True, random_state=42)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])
text_clf.fit(twenty_train.data, twenty_train.target)

# 测试集分类
# docs_new = ['God is love', 'OpenGL on the GPU is fast']
#
# predicted = text_clf.predict(docs_new)
#
# for doc, category in zip(docs_new, predicted):
#     print('%r => %s' % (doc, twenty_train.target_names[category]))


predicted = text_clf.predict(twenty_test.data)
m = np.mean(predicted == twenty_test.target)
print(m)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None)),
                     ])
text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(twenty_test.data)
x = np.mean(predicted == twenty_test.target)
print(x)
