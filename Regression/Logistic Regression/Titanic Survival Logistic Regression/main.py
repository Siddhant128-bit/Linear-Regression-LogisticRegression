import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#importint data in csv format
titanic_data=pd.read_csv('titanic.csv')

#Data analysis carried out visually;
sns.countplot(x='Survived',data=titanic_data,label='Survived')
plt.figure(2)
sns.countplot(x='Survived',hue='Sex',data=titanic_data)
plt.figure(3)
sns.countplot(x='Survived',hue='Pclass',data=titanic_data)


#data cleaning
titanic_data.dropna(inplace=True)
print(titanic_data.isnull().sum())
sex_column=pd.get_dummies(titanic_data['Sex'],drop_first=True)
embark_column=pd.get_dummies(titanic_data['Embarked'],drop_first=True)
Pcl_column=pd.get_dummies(titanic_data['Pclass'],drop_first=True)
titanic_data=pd.concat([titanic_data,sex_column,embark_column,Pcl_column],axis=1)
titanic_data.drop(['Sex','Embarked','PassengerId','Name','Ticket','Pclass','Cabin','Fare'],axis=1,inplace=True)
print(titanic_data.head(10))

#training data
X=titanic_data.drop('Survived',axis=1) #except survived everything else is a feature which contributes to the label Independent variable
Y=titanic_data['Survived'].values #this acts as final label if 1 survived else perished Dependent variable final output Depedent variable
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=42)#divided as 70-30 70 train and 30 test data
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train) #fiting in this model logistic regression made on the train test columns
predictions=logmodel.predict(X_test) #prediction of all X_test features
#print(classification_report(y_test,predictions))#compare the two and get the results
print(accuracy_score(y_test,predictions))
plt.show()
