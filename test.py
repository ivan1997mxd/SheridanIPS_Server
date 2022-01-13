from __future__ import division, print_function

import os
from math import log

from cv2 import cv2

"""
========================
Fuzzy c-means clustering
========================

Fuzzy logic principles can be used to cluster multidimensional data, assigning
each point a *membership* in each cluster center from 0 to 100 percent. This
can be very powerful compared to traditional hard-thresholded clustering where
every point is assigned a crisp, exact label.

Fuzzy c-means clustering is accomplished via ``skfuzzy.cmeans``, and the
output from this function can be repurposed to classify new data according to
the calculated clusters (also known as *prediction*) via
``skfuzzy.cmeans_predict``

Data generation and setup
-------------------------

In this example we will first undertake necessary imports, then define some
test data to work with.

"""
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

# Define three cluster centers
centers = [[4, 2],
           [1, 7],
           [5, 6]]

# Define three cluster sigmas in x and y, respectively
sigmas = [[0.8, 0.3],
          [0.3, 0.5],
          [1.1, 0.7]]

# Generate test data
np.random.seed(42)  # Set seed for reproducibility
xpts = np.zeros(1)
ypts = np.zeros(1)
labels = np.zeros(1)
for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
    xpts = np.hstack((xpts, np.random.standard_normal(200) * xsigma + xmu))
    ypts = np.hstack((ypts, np.random.standard_normal(200) * ysigma + ymu))
    labels = np.hstack((labels, np.ones(200) * i))

# Visualize the test data
fig0, ax0 = plt.subplots()
for label in range(3):
    ax0.plot(xpts[labels == label], ypts[labels == label], '.',
             color=colors[label])
ax0.set_title('Test data: 200 points x3 clusters.')

"""
.. image:: PLOT2RST.current_figure

Clustering
----------

Above is our test data. We see three distinct blobs. However, what would happen
if we didn't know how many clusters we should expect? Perhaps if the data were
not so clearly clustered?

Let's try clustering our data several times, with between 2 and 9 clusters.

"""
# Set up the loop and plot
fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
alldata = np.vstack((xpts, ypts))
fpcs = []

for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)

    # Store fpc values for later
    fpcs.append(fpc)

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)
    for j in range(ncenters):
        ax.plot(xpts[cluster_membership == j],
                ypts[cluster_membership == j], '.', color=colors[j])

    # Mark the center of each fuzzy cluster
    for pt in cntr:
        ax.plot(pt[0], pt[1], 'rs')

    ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    ax.axis('off')

fig1.tight_layout()

"""
.. image:: PLOT2RST.current_figure

The fuzzy partition coefficient (FPC)
-------------------------------------

The FPC is defined on the range from 0 to 1, with 1 being best. It is a metric
which tells us how cleanly our data is described by a certain model. Next we
will cluster our set of data - which we know has three clusters - several
times, with between 2 and 9 clusters. We will then show the results of the
clustering, and plot the fuzzy partition coefficient. When the FPC is
maximized, our data is described best.

"""

fig2, ax2 = plt.subplots()
ax2.plot(np.r_[2:11], fpcs)
ax2.set_xlabel("Number of centers")
ax2.set_ylabel("Fuzzy partition coefficient")

"""
.. image:: PLOT2RST.current_figure

As we can see, the ideal number of centers is 3. This isn't news for our
contrived example, but having the FPC available can be very useful when the
structure of your data is unclear.

Note that we started with *two* centers, not one; clustering a dataset with
only one cluster center is the trivial solution and will by definition return
FPC == 1.


====================
Classifying New Data
====================

Now that we can cluster data, the next step is often fitting new points into
an existing model. This is known as prediction. It requires both an existing
model and new data to be classified.

Building the model
------------------

We know our best model has three cluster centers. We'll rebuild a 3-cluster
model for use in prediction, generate new uniform data, and predict which
cluster to which each new data point belongs.

"""
# Regenerate fuzzy model with 3 cluster centers - note that center ordering
# is random in this clustering algorithm, so the centers may change places
cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
    alldata, 3, 2, error=0.005, maxiter=1000)

# Show 3-cluster model
fig2, ax2 = plt.subplots()
ax2.set_title('Trained model')
for j in range(3):
    ax2.plot(alldata[0, u_orig.argmax(axis=0) == j],
             alldata[1, u_orig.argmax(axis=0) == j], 'o',
             label='series ' + str(j))
ax2.legend()

"""
.. image:: PLOT2RST.current_figure

Prediction
----------

Finally, we generate uniformly sampled data over this field and classify it
via ``cmeans_predict``, incorporating it into the pre-existing model.

"""

# Generate uniformly sampled data spread across the range [0, 10] in x and y
newdata = np.random.uniform(0, 1, (1100, 2)) * 10

# Predict new cluster membership with `cmeans_predict` as well as
# `cntr` from the 3-cluster model
u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    newdata.T, cntr, 2, error=0.005, maxiter=1000)

# Plot the classified uniform data. Note for visualization the maximum
# membership value has been taken at each point (i.e. these are hardened,
# not fuzzy results visualized) but the full fuzzy result is the output
# from cmeans_predict.
cluster_membership = np.argmax(u, axis=0)  # Hardening for visualization

fig3, ax3 = plt.subplots()
ax3.set_title('Random points classifed according to known centers')
for j in range(3):
    ax3.plot(newdata[cluster_membership == j, 0],
             newdata[cluster_membership == j, 1], 'o',
             label='series ' + str(j))
ax3.legend()

plt.show()

"""
.. image:: PLOT2RST.current_figure

"""

"""
函数说明:创建测试数据集
"""
proposed_pattern = {
    (1, 6, 7, 11, 12, 17, 16, 2): {
        (1, 2, 7, 6): {
            (1, 2): None,
            (7, 6): None
        },
        (17, 12, 11, 16): {
            (17, 12): None,
            (11, 16): None
        }
    },
    (20, 15, 19, 14, 13, 9, 10, 4, 8, 3, 18, 5): {
        (20, 19, 14, 13, 15, 18): {
            (20, 18): None,
            (13, 14, 15, 19): {
                (13, 14, 19): {
                    (13, 14): None,
                    (19,): None
                },
                (15,): None
            }
        },
        (3, 4, 5, 9, 10, 8): {
            (3, 5): None,
            (10, 9, 8, 4): {
                (10, 9, 4): {
                    (10, 9): None,
                    (4,): None
                },
                (8,): None
            }
        }
    }
}


def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 分类属性
    return dataSet, labels  # 返回数据集和分类属性


"""
函数说明:计算给定数据集的经验熵(香农熵)
Parameters:
    dataSet - 数据集
Returns:
    shannonEnt - 经验熵(香农熵)
"""


def calcShannonEnt(dataSet):
    numEntires = len(dataSet)  # 返回数据集的行数
    labelCounts = {}  # 保存每个标签(Label)出现次数的字典
    for featVec in dataSet:  # 对每组特征向量进行统计
        currentLabel = featVec[-1]  # 提取标签(Label)信息
        if currentLabel not in labelCounts.keys():  # 如果标签(Label)没有放入统计次数的字典,添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # Label计数
    shannonEnt = 0.0  # 经验熵(香农熵)
    for key in labelCounts:  # 计算香农熵
        prob = float(labelCounts[key]) / numEntires  # 选择该标签(Label)的概率
        shannonEnt -= prob * log(prob, 2)  # 利用公式计算
    return shannonEnt  # 返回经验熵(香农熵)


"""
函数说明:按照给定特征划分数据集
Parameters:
    dataSet - 待划分的数据集
    axis - 划分数据集的特征
    value - 需要返回的特征的值
"""


def splitDataSet(dataSet, axis, value):
    retDataSet = []  # 创建返回的数据集列表
    for featVec in dataSet:  # 遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 去掉axis特征
            reducedFeatVec.extend(featVec[axis + 1:])  # 将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
    return retDataSet  # 返回划分后的数据集


"""
函数说明:选择最优特征
Parameters:
    dataSet - 数据集
Returns:
    bestFeature - 信息增益最大的(最优)特征的索引值
"""

#
# def chooseBestFeatureToSplit(dataSet):
#     numFeatures = len(dataSet[0]) - 1  # 特征数量
#     baseEntropy = calcShannonEnt(dataSet)  # 计算数据集的香农熵
#     bestInfoGain = 0.0  # 信息增益
#     bestFeature = -1  # 最优特征的索引值
#     for i in range(numFeatures):  # 遍历所有特征
#         # 获取dataSet的第i个所有特征
#         featList = [example[i] for example in dataSet]
#         uniqueVals = set(featList)  # 创建set集合{},元素不可重复
#         newEntropy = 0.0  # 经验条件熵
#         for value in uniqueVals:  # 计算信息增益
#             subDataSet = splitDataSet(dataSet, i, value)  # subDataSet划分后的子集
#             prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
#             newEntropy += prob * calcShannonEnt(subDataSet)  # 根据公式计算经验条件熵
#         infoGain = baseEntropy - newEntropy  # 信息增益
#         print("第%d个特征的增益为%.3f" % (i, infoGain))  # 打印每个特征的信息增益
#         if (infoGain > bestInfoGain):  # 计算信息增益
#             bestInfoGain = infoGain  # 更新信息增益，找到最大的信息增益
#             bestFeature = i  # 记录信息增益最大的特征的索引值
#     return bestFeature  # 返回信息增益最大的特征的索引值
#
#
# if __name__ == '__main__':
#     dataSet, features = createDataSet()
#     entropy = calcShannonEnt(dataSet)
#     bestfeature = chooseBestFeatureToSplit(dataSet)
#     print("训练集的熵为:%f" % (entropy))
#     print("最优特征索引值:" + str(bestfeature))


# def makeDict(devices, ip_addresses):
#     new_dict = dict()
#     for i in range(len(devices)):
#         device = devices[i]
#         ip_address = ip_addresses[i]
#         new_dict[device] = ip_address
#     while True:
#         key = input("Please enter the key (enter q to quit):")
#         if key == "q":
#             break
#         if key in new_dict.keys():
#             print("key:{}, value:{}".format(key, new_dict[key]))
#         else:
#             print("Key not found")
#
#
# tuple1 = ("PC1", "PC2", "PC3")
# tuple2 = ("192.168.1.1", "192.168.1.2", "192.168.1.3")
# makeDict(tuple1, tuple2)
# zone_list = [(73, 418), (153, 418), (233, 418),
#              (73, 338), (153, 338), (233, 338),
#              (73, 258), (153, 258), (233, 258),
#              (73, 178), (153, 178), (233, 178),
#              (73, 98), (153, 98), (233, 98)]
# zone_list = [(73, 418), (73, 338), (73, 258), (73, 178), (73, 98),
#              (153, 418), (153, 338), (153, 258), (153, 178), (153, 98),
#              (233, 418), (233, 338), (233, 258), (233, 178), (233, 98)]
#
# image = cv2.imread("static/img/map1.jpg")
# overlay = image.copy()
# output = image.copy()
#
# for index, zone in enumerate(zone_list):
#     x = zone[0]
#     y = zone[1]
#     # cv2.rectangle(image, (x - 40, y - 40), (x + 40, y + 40), (255, 0, 0), 2)
#     # cv2.circle(image, (x, y), 5, (0, 0, 255), thickness=-1)
#     if index >= 9:
#         cv2.putText(overlay, str(index + 1), (x - 45, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (107, 107, 107), 3)
#     else:
#         cv2.putText(overlay, str(index + 1), (x - 20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (107, 107, 107), 3)
#
# cv2.addWeighted(overlay, 0.5, output, 0.5,
#                 0, output)
# basePath = os.path.dirname(__file__)
# cv2.imwrite(os.path.join(basePath, 'static/img', 'map2.png'), output)
# print("Images changed")
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from sklearn import svm
#
# # we create 40 separable points
# np.random.seed(0)
# X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
# Y = [0] * 20 + [1] * 20
#
# # figure number
# fignum = 1
#
# # fit the model
# for name, penalty in (('unreg', 1), ('reg', 0.05)):
#
#     clf = svm.SVC(kernel='linear', C=penalty)
#     clf.fit(X, Y)
#
#     # get the separating hyperplane
#     w = clf.coef_[0]
#     a = -w[0] / w[1]
#     xx = np.linspace(-5, 5)
#     yy = a * xx - (clf.intercept_[0]) / w[1]
#
#     # plot the parallels to the separating hyperplane that pass through the
#     # support vectors (margin away from hyperplane in direction
#     # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
#     # 2-d.
#     margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
#     print(margin)
#     yy_down = yy - np.sqrt(1 + a ** 2) * margin
#     yy_up = yy + np.sqrt(1 + a ** 2) * margin
#
#     # plot the line, the points, and the nearest vectors to the plane
#     plt.figure(fignum, figsize=(4, 3))
#     plt.clf()
#     plt.plot(xx, yy, 'k-')
#     plt.plot(xx, yy_down, 'k--')
#     plt.plot(xx, yy_up, 'k--')
#
#     plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
#                 facecolors='none', zorder=10, edgecolors='k',
#                 cmap=cm.get_cmap('RdBu'))
#     plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=cm.get_cmap('RdBu'),
#                 edgecolors='k')
#
#     plt.axis('tight')
#     x_min = -4.8
#     x_max = 4.2
#     y_min = -6
#     y_max = 6
#
#     YY, XX = np.meshgrid(yy, xx)
#     xy = np.vstack([XX.ravel(), YY.ravel()]).T
#     Z = clf.decision_function(xy).reshape(XX.shape)
#
#     # Put the result into a contour plot
#     plt.contourf(XX, YY, Z, cmap=cm.get_cmap('RdBu'),
#                  alpha=0.5, linestyles=['-'])
#
#     plt.xlim(x_min, x_max)
#     plt.ylim(y_min, y_max)
#
#     plt.xticks(())
#     plt.yticks(())
#     fignum = fignum + 1
#
# plt.show()

# from itertools import combinations
# from random import randint
#
# A = [[randint(-5, 5) for coord in range(2)] for point in range(500)]
#
#
# def square_distance(x, y): return sum([(xi - yi) ** 2 for xi, yi in zip(x, y)])
#
#
# max_square_distance = 0
# for pair in combinations(A, 2):
#     if square_distance(*pair) > max_square_distance:
#         max_square_distance = square_distance(*pair)
#         max_pair = pair
#
# print(max_pair)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# we create 40 separable points
X, y = make_blobs(n_samples=40, centers=2, random_state=6)

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()
