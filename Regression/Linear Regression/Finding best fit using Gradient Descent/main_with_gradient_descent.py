import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def R_squaredcheck(X,Y,m,c):
    num_r=0
    den_r=0
    x_mean=np.mean(X)
    y_mean=np.mean(Y)
    for i in range(len(Y)):
        num_r=num_r+((m*X[i]+c)-Y[i])**2
        den_r=den_r+((Y[i]-y_mean)**2)
    R_squared=1-(num_r/den_r)
    print('Rsquared value: ',R_squared)


def ResidualCalculations(X,Y):
    learning_rate_for_c=0.00000000001
    learning_rate_for_m=0.00000000001
    m=30
    c=0
    step=0
    while True:
        sum_of_residuals_dif_c=0
        sum_of_residuals_dif_m=0
        for i in range(len(Y)):
            sum_of_residuals_dif_c=sum_of_residuals_dif_c+(-2)*(Y[i]-(m*X[i]+c))
            sum_of_residuals_dif_m=sum_of_residuals_dif_m+(-2)*(X[i])*(Y[i]-(m*X[i]+c))
        m=m-(sum_of_residuals_dif_m*learning_rate_for_m)
        c=c-(sum_of_residuals_dif_c*learning_rate_for_c)
        print(m,c)
        step+=1
        if(step==1000):
            break
    m=float(f'{m: .6f}'[:-1])
    c=float(f'{c: .6f}'[:-1])
    print(m,c)
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
    '''x1=input('Enter the value of Headsize')
    x1=int(x1)
    print(m*x1+c)'''
    R_squaredcheck(X,Y,m,c)

def getrequiredvalues():
    data=pd.read_csv('headbrain.csv')
    X=data['Head Size(cm^3)'].values
    Y=data['Brain Weight(grams)'].values
    ResidualCalculations(X,Y)


if __name__ == '__main__':
    getrequiredvalues()
