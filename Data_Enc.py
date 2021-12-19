#Ryan Miller
#Machine Learning 
#Final Project
#Kaggle Compeition 

import numpy as np 
from sklearn import preprocessing
from ID3_Fill import X_test_final, X_train_final, y_train

#Script 1: Filling in Missing Features
#new split is training and testing
#seperate cat and numerical 
#then encode them -> similar but instead of missing and not missing split it is test and training
#apply algorithms

cols_cat = ["workclass", "education", "marital.status", "occupation", "relationship", "race", "sex", "native.country"]
cols_num = ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]

#X_train_final, X_test_final, y_train

X_train_cat = X_train_final.drop(columns = cols_num)
X_train_num = X_train_final.drop(columns = cols_cat)
X_test_cat = X_test_final.drop(columns = cols_num)
X_test_num = X_test_final.drop(columns = cols_cat)
X_cat = X_train_cat.values.tolist() + X_test_cat.values.tolist()

#X_train and X_test cat label encoding and to numpy array
le_X_cat = preprocessing.MultiLabelBinarizer()
le_X_cat.fit(X_cat)
X_cat_enc = le_X_cat.transform(X_cat)
X_cat_enc_arr = np.asarray(X_cat_enc)
X_cat_enc_train_arr = X_cat_enc_arr[:len(X_train_final),:]
X_cat_enc_test_arr = X_cat_enc_arr[len(X_train_final):,:]

#Numerical stuff to arrays as well
X_train_num_arr = np.asarray(X_train_num)
X_test_num_arr = np.asarray(X_test_num)
y_train_arr = np.asarray(y_train)

#Putting categorical and numerical together 
X_train_arr = np.concatenate((X_cat_enc_train_arr,X_train_num_arr),axis = 1)
X_test_arr = np.concatenate((X_cat_enc_test_arr,X_test_num_arr),axis = 1)


