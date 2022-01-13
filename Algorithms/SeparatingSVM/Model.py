from typing import List

from anytree import NodeMixin, RenderTree
from sklearn.pipeline import Pipeline

from Resources.Objects.Points.AccessPoint import AccessPoint
from Resources.Objects.Zone import Zone
from Algorithms.svm.svm import svm_model
import numpy as np
import pandas as pd
from keras.models import Sequential
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


class MyModel(NodeMixin):

    def __init__(self, name: str, zones: List[Zone], access_points: List[AccessPoint], clf: Pipeline = None,
                 svm: svm_model = None,
                 parent=None, children=None):
        self.clf = clf
        self.name = name
        self.zones = zones
        self.access_points = access_points
        if svm:
            self.svm = svm
        if parent:
            self.parent = parent
        if children:
            self.children = children

    @property
    def svm_model(self):
        return self.svm

    @property
    def id(self):
        return int(self.name)


class Partition:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data
        self.model = None
        self.center = None
        # Insert Node

    @property
    def children(self):
        return [self.left, self.right]

    def insert(self, partition):
        if self.data:
            if self.left is None:
                self.left = partition
            else:
                self.right = partition

    def other_name(self, level=0):
        print('--' * level + repr(self.data))
        for child in self.children:
            if child is not None:
                child.other_name(level + 1)


class MLModel:

    def __init__(self, rf: RandomForestClassifier, knn: KNeighborsClassifier, dt: DecisionTreeClassifier,
                 ann: Sequential, svm: SVC, svm1: SVC, gnb: GaussianNB):
        self.rf = rf
        self.knn = knn
        self.dt = dt
        self.ann = ann
        self.svm = svm
        self.gnb = gnb
        self.svm1 = svm1

    def train_one(self, mode, X_train, y_train):
        if mode == "ann":
            self.ann.fit(X_train, y_train, epochs=10)
        if mode == "rf":
            self.rf.fit(X_train, y_train)
        if mode == "dt":
            self.dt.fit(X_train, y_train)
        if mode == "knn":
            self.knn.fit(X_train, y_train)
        if mode == "svm":
            self.svm.fit(X_train, y_train)
        if mode == "svm1":
            self.svm1.fit(X_train, y_train)
        if mode == "gnb":
            self.gnb.fit(X_train, y_train)

    def train(self, X_train, y_train):
        self.ann.fit(X_train, y_train, epochs=10)
        self.rf.fit(X_train, y_train)
        self.dt.fit(X_train, y_train)
        self.knn.fit(X_train, y_train)
        self.svm.fit(X_train, y_train)
        self.svm1.fit(X_train, y_train)
        self.gnb.fit(X_train, y_train)

    def test_one(self, mode, X_test):
        if mode == "rf":
            return self.rf.predict(X_test)
        if mode == "dt":
            return self.dt.predict(X_test)
        if mode == "knn":
            return self.knn.predict(X_test)
        if mode == "ann":
            return np.argmax(self.ann.predict(X_test), axis=-1)
        if mode == "svm":
            return self.svm.predict(X_test)
        if mode == "svm1":
            return self.svm1.predict(X_test)
        if mode == "gnb":
            return self.gnb.predict(X_test)

    def get_model(self, mode):
        if mode == "rf":
            return self.rf
        if mode == "dt":
            return self.dt
        if mode == "knn":
            return self.knn
        if mode == "ann":
            return self.ann
        if mode == "svm":
            return self.svm
        if mode == "svm1":
            return self.svm1
        if mode == "gnb":
            return self.gnb

    def test(self, X_test, y_test):
        rf_pred = self.rf.predict(X_test)
        dt_pred = self.dt.predict(X_test)
        knn_pred = self.knn.predict(X_test)
        ann_pred = np.argmax(self.ann.predict(X_test), axis=-1)
        svm_pred = self.svm.predict(X_test)
        svm1_pred = self.svm1.predict(X_test)
        gnb_pred = self.gnb.predict(X_test)

        rf_acc = metrics.accuracy_score(y_test, rf_pred)
        dt_acc = metrics.accuracy_score(y_test, dt_pred)
        knn_acc = metrics.accuracy_score(y_test, knn_pred)
        ann_acc = metrics.accuracy_score(y_test, ann_pred)
        svm_acc = metrics.accuracy_score(y_test, svm_pred)
        svm1_acc = metrics.accuracy_score(y_test, svm1_pred)
        gnb_acc = metrics.accuracy_score(y_test, gnb_pred)

        return rf_acc, dt_acc, knn_acc, ann_acc, svm_acc, gnb_acc, svm1_acc

    def predict(self, X):
        rf_result = self.rf.predict(X)
        dt_result = self.dt.predict(X)
        knn_result = self.knn.predict(X)
        ann_result = np.argmax(self.ann.predict(X), axis=-1)
        svm_result = self.svm.predict(X)
        svm1_result = self.svm1.predict(X)
        gnb_result = self.gnb.predict(X)

        result_list = [rf_result[0], dt_result[0], knn_result[0], ann_result[0], svm_result[0], svm1_result[0], gnb_result[0]]

        return result_list
