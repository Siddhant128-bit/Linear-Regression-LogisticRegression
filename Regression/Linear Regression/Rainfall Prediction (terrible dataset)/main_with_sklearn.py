from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rainfall_data=pd.read_csv('Rainfallmm.csv')
X=[]
Y=[]
for i in range(2004,2016):
    X.append(i)

for i in X:
    rainfall_data[str(i)]=rainfall_data[str(i)].fillna(0)

print(rainfall_data)

print('\n\n\t\t Enter the number of the city to train the model on\n\n' )
print(rainfall_data['Station'])
x=int(input('\n\nEnter the number of the city: '))
print('Chosen City is: '+rainfall_data['Station'][x])
for i in X:
    rainfall_data[str(i)].fillna(0)
    Y.append(rainfall_data[str(i)][x])


X=np.array(X)
Y=np.array(Y)

X=X.reshape(-1,1)
linearmodel=LinearRegression()
linearmodel.fit(X,Y)
Y_pred=linearmodel.predict(X)
mse=mean_squared_error(Y,Y_pred)
print(mse)
print(linearmodel.score(X,Y))
