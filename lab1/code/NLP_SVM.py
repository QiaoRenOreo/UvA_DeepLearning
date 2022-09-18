# Eg 1 https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
print (X.shape,y.shape)
# clf = make_pipeline ( StandardScaler(), LinearSVC(random_state=0, tol=1e-5) )
# pipeline= clf.fit(X, y)
# print (pipeline)
#
# print(clf.named_steps['linearsvc'].coef_)
# print(clf.named_steps['linearsvc'].intercept_)
# print(clf.predict([[0, 0, 0, 0]]))

### eg2
# import numpy as np
# from sklearn import preprocessing, cross_validation, neighbors
# import pandas as pd
# import sklearn
#
# df = pd.read_csv('breast-cancer-wisconsin.data.txt')
# df.replace('?',-99999, inplace=True)
# df.drop(['id'], 1, inplace=True)
#
# X = np.array(df.drop(['class'], 1))
# y = np.array(df['class'])
#
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
#
# clf = neighbors.KNeighborsClassifier()
#
#
# clf.fit(X_train, y_train)
# confidence = clf.score(X_test, y_test)
# print(confidence)
#
# example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
# example_measures = example_measures.reshape(len(example_measures), -1)
# prediction = clf.predict(example_measures)
# print(prediction)

# eg3 https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm
# X_x = [1, 5, 1.5, 8, 1, 9]
# X_y = [2, 8, 1.8, 8, 0.6, 11]
# plt.scatter(X_x,X_y)
# plt.show()

X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])
label = np.array([[0,1,0,1,0,1]])
print (X.shape, label.shape)

clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X,label)
print(clf.predict([[0.58,0.76]]))
print(clf.predict([[10.58,10.76]]))
#
# # visualize data
# w = clf.coef_[0]
# print(w)
# a = -w[0] / w[1]
# xx = np.linspace(0,12)
# yy = a * xx - clf.intercept_[0] / w[1]
# h0 = plt.plot(xx, yy, 'k-', label="non weighted div")
# plt.scatter(X[:, 0], X[:, 1], c = y)
# plt.legend()
# plt.show()
