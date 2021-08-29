import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('Headbrain.csv')
print(data)
X=data['Head Size(cm^3)'].values
Y=data['Brain Weight(grams)'].values
n=len(data)
x_mean=np.mean(X)
y_mean=np.mean(Y)
num_r=0
den_r=0
for i in range(n):
    num_r=num_r+((X[i]-x_mean)*(Y[i]-y_mean))
    den_r=den_r+((X[i]-x_mean)**2)
m=num_r/den_r
c=y_mean-m*x_mean
#upto here is calculation of y=mx+c now plotting it all
max_x=np.max(X)+100
min_x=np.min(X)-100
x=np.linspace(min_x,max_x,100)
y=m*x+c
plt.plot(x,y,color="#58b970",label='Regression Line')
plt.scatter(X,Y,c="#ef5423",label="Scatter plot")
plt.xlabel('head size (cm^3)')
plt.ylabel('brain weights (grams)')
plt.legend()
plt.show()
# now calculation of R squared to get the error
num_r=0
den_r=0
for i in range(n):
    num_r=num_r+((m*X[i]+c)-Y[i])**2
    den_r=den_r+((Y[i]-y_mean)**2)

R_squared=1-(num_r/den_r)
print('Rsquared value: ',R_squared)
print(m,c)
'''
x1=input('Enter the value of Headsize')
x1=int(x1)
print(m*x1+c)'''
