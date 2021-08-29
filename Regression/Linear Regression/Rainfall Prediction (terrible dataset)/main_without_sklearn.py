import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
X=[]
Y=[]
for i in range(2004,2016):
    X.append(i)

rainfall_data=pd.read_csv('Rainfallmm.csv')
print('\n\n\t\t Enter the number of the city to train the model on\n\n' )
print(rainfall_data['Station'])
x=int(input('\n\nEnter the number of the city: '))
print('Chosen City is: '+rainfall_data['Station'][x])
for i in X:
    rainfall_data[str(i)]=rainfall_data[str(i)].fillna(0)
    
for i in X:
    Y.append(rainfall_data[str(i)][x])

x_mean=np.mean(X)
y_mean=np.mean(Y)


n=len(Y)

num_r=0
den_r=0
for i in range(n):
    num_r=num_r+((X[i]-x_mean)*(Y[i]-y_mean))
    den_r=den_r+((X[i]-x_mean)**2)
m=num_r/den_r
c=y_mean-m*x_mean

print('Year: ',X,'\nRanfall: ',Y)

#testing with R squared method
num_r=0
den_r=0
for i in range(len(Y)):
    num_r=num_r+((m*X[i]+c)-Y[i])**2
    den_r=den_r+((Y[i]-y_mean)**2)

R_squared=1-(num_r/den_r)
print('Rsquared value: ',R_squared)
Z=int(input("Enter the year: "))
print(m*Z+c)

max_x=np.max(X)+100
min_x=np.min(X)-100
x=np.linspace(min_x,max_x,100)
y=m*x+c
plt.plot(x,y,color="#58b970",label='Regression Line')
plt.scatter(X,Y,c="#ef5423",label="Scatter plot")

plt.legend()
plt.show()
