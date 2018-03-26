#coding:utf-8
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn
import numpy as np

def data_type(s):
    it = {'0': 0, '1': 1}
    return it[s]
path = u'/Users/tal/PycharmProjects/svmproject/data.text'
data = np.loadtxt(path, dtype=float, delimiter=',', converters={0: data_type})
# print data

#数据，数据分成几份？/分割位置？,次数就是x为0到19，y为20最后一个，轴，为1水平分割，为0垂直分割
x, y = np.split(data, (19,), axis=1)
print x
print y

#只取前两列，这个地方应该需要优化，原例子分类在最后一列
# x = x[:, :2]


#参数为 索要划分的样本特征集，所要划分的样本结果，测试赝本占比（这个示例上没有，不加会报错），训练样本占比，如果为整数就是样本数量，随机数的种子

# 随机数种子：
# 其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，
# 其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。随机数的产生取决于种子，
# 随机数和种子之间的关系遵从以下两个规则：种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。

#下面这段代码是迭代赋值，函数有4个返回值，但是y_train是array([], shape=(7, 0), dtype=float64)
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.4, train_size=0.6, random_state=1)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.4, train_size=0.6, random_state=1)


# kernel 为liner时为线性核 C越大分类效果越好，但有可能会过拟合（defaul C=1。
# rbf为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
#decision_function_shape='ovr'时，为one v rest，即一个类别与其他类别进行划分，
#decision_function_shape='ovo'时，为one v one，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果


print x_train
# # print len(x_train)
# print y_train
# # print y_train.ravel()



# # clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')

# ravle函数的作用是数组将维[[1,2],[3,4]]----[1,2,3,4]
clf.fit(x_train, y_train.ravel())


print clf.score(x_train, y_train)  # 精度
y_hat = clf.predict(x_train)
print y_hat
#show_accuracy(y_hat, y_train, '训练集')
print clf.score(x_test, y_test)
y_hat = clf.predict(x_test)
#show_accuracy(y_hat, y_test, '测试集')