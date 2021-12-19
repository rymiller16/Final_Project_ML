#Ryan Miller
#Machine Learning 
#Final Project
#Kaggle Compeition 

import numpy as np 
from Data_Enc import X_test_arr, X_train_arr, y_train_arr
from sklearn.linear_model import LogisticRegression
import csv

dispatcher ={1:1,0:-1}
y_train = [dispatcher[lb] for lb in y_train_arr]
y_train_arr_2 = np.asarray(y_train)

#Logistic Regression
clf = LogisticRegression(multi_class='ovr')
clf.fit(X_train_arr,y_train_arr_2)
y_test_hat = clf.predict(X_test_arr)
err = clf.score(X_train_arr,y_train_arr_2)
print(err)

with open('Kaggle-LogisticRegression.csv', 'w', newline='')as file:
    writer = csv.writer(file)
    writer.writerow(['ID','Prediction'])
    for i in range(1,len(y_test_hat)+1):
        writer.writerow([i,y_test_hat[i-1]])




