# install graphviz and put <installation path>\bin into path
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X = iris.data[:, 1:] # petal length and width
y = iris.target
clf = LogisticRegression(C=100.0, random_state=1)
clf.fit(X, y)


print(clf.predict_proba(X[:3, :]))
print(clf.predict(X[:3, :]))

labels=['Setosa', 'Versicolour', 'Virginica']
print(labels[clf.predict(X[:3, :])[0]])

print('-------------------------------------')
print(clf.predict_proba(X[50:53, :]))
print(clf.predict(X[50:53, :]))

labels=['Setosa', 'Versicolour', 'Virginica']
print(labels[clf.predict(X[50:53, :])[0]])

