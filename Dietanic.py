# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 20:48:57 2018

@author: Dipankar Karmakar
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

train_data=pd.read_csv(r'C:\Users\Dipankar-PC\Downloads\train.csv',encoding='utf8')
Y=train_data['Survived']
print(train_data.head())
print(train_data.isnull().sum().sum())

le = preprocessing.LabelEncoder()
train_data['sex']=le.fit_transform(train_data['Sex'])
train_data['embarked']=le.fit_transform(train_data['Embarked'].astype(str))

train_data=train_data.drop(['PassengerId','Survived','Name','Ticket','Cabin','Sex','Embarked'],axis=1)
print(train_data.head())
X_train,X_test,Y_train,Y_test=train_test_split(train_data,Y,test_size=0.2)

print(X_train.head())
print(X_test.head())

conti_cols=['Age','Fare']
cater_cols=['Pclass','sex','SibSp','Parch','embarked']

#Dataframe=rows and columns ,i.e., 2d matrix, and only row or column is series)
conti=pd.DataFrame(X_train,columns=conti_cols)
cater=pd.DataFrame(X_train,columns=cater_cols)
conti_te=pd.DataFrame(X_test,columns=conti_cols)
cater_te=pd.DataFrame(X_test,columns=cater_cols)

#jaha jaha NaN dikhega waha waha mean lag jayega
imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
imp.fit(conti)
imputed_data=imp.transform(conti.values)
contitrain_dataa=pd.DataFrame(imputed_data,columns=conti_cols)
contitrain_dataa.isnull().sum().sum()

imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
imp.fit(conti_te)
imputed_data1=imp.transform(conti_te.values)
contitest_dataa=pd.DataFrame(imputed_data1,columns=conti_cols)
contitest_dataa.isnull().sum()

imp2=Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
imp2.fit(cater)
imputed_data2=imp2.transform(cater.values)
catertrain_dataa=pd.DataFrame(imputed_data2,columns=cater_cols)
catertrain_dataa.isnull().sum().sum()

imp2=Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
imp2.fit(cater_te)
imputed_data3=imp2.transform(cater_te.values)
catertest_dataa=pd.DataFrame(imputed_data3,columns=cater_cols)
catertest_dataa.isnull().sum()

data_f1=pd.concat([contitrain_dataa,catertrain_dataa],axis=1)
data_2=pd.concat([contitest_dataa,catertest_dataa],axis=1)

scl=StandardScaler()
data_scl=scl.fit_transform(data_f1)
data_pd=pd.DataFrame(data_scl,columns=data_f1.columns)

scl=StandardScaler()
data_scl1=scl.fit_transform(data_2)
data_pd1=pd.DataFrame(data_scl1,columns=data_2.columns)

import seaborn as sns
#by heatmap we see the level of covariance,then dropping the columns by the given threshold value
#-ve value means one increasing and other one decreasing
#-ve and +ve we dont see that in it we just see the hard value
sns.heatmap(data_pd.corr(), square=True, annot=True, cmap="RdBu")
data_pd=data_pd.drop(['Fare'],axis=1)
data_pd1=data_pd1.drop(['Fare'],axis=1)

print(data_pd1.isnull().sum().sum())
print(data_pd.isnull().sum().sum())

from sklearn.ensemble import RandomForestClassifier
rfe=RandomForestClassifier()
rfe.fit(data_pd,Y_train)
y_pred=rfe.predict(data_pd1)
from sklearn.metrics import accuracy_score
acc_sc=accuracy_score(Y_test,y_pred)
print(acc_sc)

'''pca eigen vectors eigen values'''

cor_matt=data_pd.corr()
eig_vals, eig_vecs = np.linalg.eig(cor_matt)
print(eig_vals)
print('sdaddddddddddddddd')
print(eig_vecs)


from sklearn.decomposition import PCA

pca=PCA(n_components=4)
train_features = pca.fit_transform(data_pd)

rfe=RandomForestClassifier()
rfe.fit(train_features,Y_train)

test_features = pca.transform(data_pd1)

y_pred=rfe.predict(test_features)
from sklearn.metrics import accuracy_score
acc_sc=accuracy_score(Y_test,y_pred)
print(acc_sc)
















random_imp=RandomForestClassifier()
random_imp.fit(data_pd,Y_train)
print(random_imp.feature_importances_)
print(data_pd.columns)
a=data_pd.columns
b=random_imp.feature_importances_
c={}
for i,j in zip(a,b):
    c.update({i:j})
print(c)
'''sorting dictionary by its value'''
from collections import Counter
c1=Counter(c)
print(c1.most_common())
    


test_data=pd.read_csv(r'C:\Users\Dipankar-PC\Downloads\test.csv',encoding='utf8')
test_data['sex']=le.fit_transform(test_data['Sex'])
test_data['embarked']=le.fit_transform(test_data['Embarked'].astype(str))
z=test_data['PassengerId']
test_data=test_data.drop(['PassengerId','Name','Ticket','Cabin','Sex','Embarked'],axis=1)
conti_te1=pd.DataFrame(test_data,columns=conti_cols)
cater_te1=pd.DataFrame(test_data,columns=cater_cols)

imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
imp.fit(conti_te1)
imputed_data5=imp.transform(conti_te1.values)
contitest_dataa6=pd.DataFrame(imputed_data5,columns=conti_cols)
contitest_dataa6.isnull().sum()

imp2=Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
imp2.fit(cater_te1)
imputed_data6=imp2.transform(cater_te1.values)
catertest_dataa6=pd.DataFrame(imputed_data6,columns=cater_cols)
catertest_dataa6.isnull().sum()

data_3=pd.concat([contitest_dataa6,catertest_dataa6],axis=1)

scl=StandardScaler()
data_scl3=scl.fit_transform(data_3)
data_pd3=pd.DataFrame(data_scl3,columns=data_2.columns)


data_pd3=data_pd3.drop(['Fare'],axis=1)
print(data_pd3.info())

from sklearn.decomposition import PCA

pca=PCA(n_components=4)
train_features1 = pca.fit_transform(data_pd)

rfe=RandomForestClassifier()
rfe.fit(train_features1,Y_train)

test_features1 = pca.transform(data_pd3)

y_pred1=rfe.predict(test_features1)

print(y_pred1)

