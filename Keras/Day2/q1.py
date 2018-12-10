from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

iris = load_iris()
onehot = OneHotEncoder()

X = iris.data
y = iris.target

y = y.reshape(-1,1)

y = onehot.fit_transform(y)


def Kmodel(a,b):
    global model 
    model = Sequential()
    model.add(Dense(12,input_dim=4,activation='relu'))
    model.add(Dense(12,activation='relu'))
    model.add(Dense(3,activation='softmax'))

    optimizer = Adam(lr=0.001)
    print(model.summary())

    model.compile(optimizer,loss='categorical_crossentropy',metrics=['accuracy'] )

    model.fit(a,b,verbose=2,epochs=101)

    return model

Kmodel(X,y)

results = model.evaluate(X,y)

print(results[0])
print(results[1])
    
    


