from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pandas as pd

from sklearn.preprocessing import OneHotEncoder,LabelEncoder


#data
data = pd.read_csv('sonar.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


#encoder
onehot = OneHotEncoder()
le=LabelEncoder()
y = le.fit_transform(y)

#standard Scalar
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#model 
def Kmodel(a,b):
    global model 
    model = Sequential()
    model.add(Dense(10,input_shape=(60,),activation='relu'))
    model.add(Dense(12,activation='relu'))
    model.add(Dense(1))

    optimizer = Adam(lr=0.001)
    print(model.summary())

    model.compile(optimizer,loss='binary_crossentropy',metrics=['accuracy'] )

    model.fit(a,b,verbose=2,epochs=40)

    return model

Kmodel(X,y)

results = model.evaluate(X,y)

print(results[0])
print('Accuracy is {0} %'.format(results[1]*100))
