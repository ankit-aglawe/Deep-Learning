

from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier

iris = load_iris()

X = iris.data
y = iris.target

mlp =MLPClassifier(activation='tanh')

mlp.fit(X,y)

y_pred = mlp.predict(X)

#print(y_pred)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y,y_pred))


#--------------------------------------------

mlp =MLPClassifier(activation='relu')

mlp.fit(X,y)

y_pred = mlp.predict(X)

#print(y_pred)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y,y_pred))
