import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
imp=SimpleImputer(missing_values=0,strategy='mean')
param_grid={'n_neighbors':np.arange(1,50)}
wbc = pd.read_csv("C:\\Users\\ur trb\\Documents\\datasets\\wbc.csv")
y=wbc['diagnosis']
X=wbc.drop(['diagnosis','id'],axis=1)
imp.fit(X)
X=imp.transform(X)
#X.to_csv('C:\\Users\\ur trb\\Documents\\datasets\\wbc1.csv')
knn=KNeighborsClassifier()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.29,random_state=21)
#knn.fit(X_train,y_train)
knn_cv=GridSearchCV(knn,param_grid,cv=9)
knn_cv.fit(X_train,y_train)
pred=knn_cv.predict(X_test)
accuracy=knn_cv.score(X_test,y_test)
print('accuracy = ' , accuracy)
print(knn_cv.best_params_,knn_cv.best_score_)

