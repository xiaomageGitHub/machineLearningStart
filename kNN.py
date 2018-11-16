#!/usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn import datasets
from sklearn import svm
# import pickle
from sklearn.externals import joblib


iris = datasets.load_iris()


digits = datasets.load_digits()
clf = svm.SVC(gamma=0.001, C=100.)

clf.fit(digits.data[:-1], digits.target[:-1])

joblib.dump(clf, 'clf.joblib')
clf2 = joblib.load('clf.joblib')
x = clf2.predict(digits.data[0:])
print(x)

