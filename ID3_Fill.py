#Ryan Miller
#Machine Learning 
#Final Project
#Kaggle Compeition 

import numpy as np 
import pandas as pd
from sklearn import preprocessing
from sklearn import tree

#Script 1: Filling in Missing Features

def get_prediction_accuracy(y,y_hat):
  num_corr = np.sum(np.array([(y[i]==y_hat[i]) for i in range(len(y))]))
  return num_corr/len(y)

path1 = "data/test_final.csv"
path2 = "data/train_final.csv"
test_df = pd.read_csv(path1)
train_df = pd.read_csv(path2)

train_df_X = train_df.iloc[:,:-1]
test_df_X = test_df.iloc[:,1:]

mask_train = (train_df_X['workclass'] == '?') | (train_df_X['occupation'] =='?') | (train_df_X['native.country'] == '?')
mask_test = (test_df_X["workclass"] == '?') | (test_df_X["occupation"] =='?') | (test_df_X["native.country"] == '?')

X_train_miss = train_df_X.loc[mask_train]
X_train_full = train_df_X.loc[~mask_train]
X_test_full = test_df_X.loc[~mask_test]
X_test_miss = test_df_X.loc[mask_test]
y_train = train_df.iloc[:,-1]

cols = ["workclass","occupation", "native.country"]
cols_cat = ["education", "marital.status", "relationship", "race", "sex"]
cols_num = ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]

X_train_full_nm = X_train_full.drop(columns = cols)
X_train_full_cat = X_train_full_nm.drop(columns = cols_num)
X_train_full_num = X_train_full_nm.drop(columns = cols_cat)
y1_train_full = X_train_full.loc[:,cols[0]]
y2_train_full = X_train_full.loc[:,cols[1]]
y3_train_full = X_train_full.loc[:,cols[2]]

X_test_full_nm = X_test_full.drop(columns = cols)
X_test_full_cat = X_test_full_nm.drop(columns = cols_num)
X_test_full_num = X_test_full_nm.drop(columns = cols_cat)
y1_test_full = X_test_full.loc[:,cols[0]]
y2_test_full = X_test_full.loc[:,cols[1]]
y3_test_full = X_test_full.loc[:,cols[2]]

X_train_miss_nm = X_train_miss.drop(columns = cols)
X_train_miss_cat = X_train_miss_nm.drop(columns = cols_num)
X_train_miss_num = X_train_miss_nm.drop(columns = cols_cat)
y1_train_miss = X_train_miss.loc[:,cols[0]]
y2_train_miss = X_train_miss.loc[:,cols[1]]
y3_train_miss = X_train_miss.loc[:,cols[2]]

X_test_miss_nm = X_test_miss.drop(columns = cols)
X_test_miss_cat = X_test_miss_nm.drop(columns = cols_num)
X_test_miss_num = X_test_miss_nm.drop(columns = cols_cat)
y1_test_miss = X_test_miss.loc[:,cols[0]]
y2_test_miss = X_test_miss.loc[:,cols[1]]
y3_test_miss = X_test_miss.loc[:,cols[2]]

X_cat = X_train_full_cat.values.tolist() + X_test_full_cat.values.tolist()
X_num = X_train_full_num.values.tolist() + X_test_full_num.values.tolist()
y1 = y1_train_full.values.tolist() + y1_test_full.values.tolist()
y2 = y2_train_full.values.tolist() + y2_test_full.values.tolist()
y3 = y3_train_full.values.tolist() + y3_test_full.values.tolist()

le_X_cat = preprocessing.MultiLabelBinarizer()
le_y1 = preprocessing.LabelEncoder()
le_y2 = preprocessing.LabelEncoder()
le_y3 = preprocessing.LabelEncoder()

le_X_cat.fit(X_cat)
le_y1.fit(y1)
le_y2.fit(y2)
le_y3.fit(y3)

X_cat_enc = le_X_cat.transform(X_cat)
y1_enc = le_y1.transform(y1)
y2_enc = le_y2.transform(y2)
y3_enc = le_y3.transform(y3)

X_cat_enc_arr = np.asarray(X_cat_enc)
X_num_arr = np.asarray(X_num)
X_arr = np.concatenate((X_cat_enc_arr,X_num_arr),axis = 1)
y1_enc_arr = np.asarray(y1_enc)
y2_enc_arr = np.asarray(y2_enc)
y3_enc_arr = np.asarray(y3_enc)

clf_y1 = tree.DecisionTreeClassifier(criterion='entropy',max_depth=50)
clf_y1.fit(X_arr,y1_enc_arr)
y1_enc_hat = clf_y1.predict(X_arr)
print(f'training accuray: {get_prediction_accuracy(y1_enc_arr,y1_enc_hat)}')

clf_y2 = tree.DecisionTreeClassifier(criterion='entropy',max_depth=50)
clf_y2.fit(X_arr,y2_enc_arr)
y2_enc_hat = clf_y2.predict(X_arr)
print(f'training accuray: {get_prediction_accuracy(y2_enc_arr,y2_enc_hat)}')

clf_y3 = tree.DecisionTreeClassifier(criterion='entropy',max_depth=50)
clf_y3.fit(X_arr,y3_enc_arr)
y3_enc_hat = clf_y3.predict(X_arr)
print(f'training accuray: {get_prediction_accuracy(y3_enc_arr,y3_enc_hat)}')

X_cat_miss = X_train_miss_cat.values.tolist() + X_test_miss_cat.values.tolist()
X_num_miss = X_train_miss_num.values.tolist() + X_test_miss_num.values.tolist()

X_cat_miss_enc = le_X_cat.transform(X_cat_miss)
X_cat_miss_enc_arr = np.asarray(X_cat_miss_enc)
X_num_miss_arr = np.asarray(X_num_miss)
X_miss_arr = np.concatenate((X_cat_miss_enc_arr,X_num_miss_arr),axis = 1)

y1_miss_enc_hat = clf_y1.predict(X_miss_arr)
y1_miss_hat = le_y1.inverse_transform(y1_miss_enc_hat)
y1_miss_hat_train = y1_miss_hat[:y1_train_miss.shape[0]]
y1_miss_hat_test = y1_miss_hat[y1_train_miss.shape[0]:]

y2_miss_enc_hat = clf_y2.predict(X_miss_arr)
y2_miss_hat = le_y2.inverse_transform(y2_miss_enc_hat)
y2_miss_hat_train = y2_miss_hat[:y2_train_miss.shape[0]]
y2_miss_hat_test = y2_miss_hat[y2_train_miss.shape[0]:]

y3_miss_enc_hat = clf_y3.predict(X_miss_arr)
y3_miss_hat = le_y3.inverse_transform(y3_miss_enc_hat)
y3_miss_hat_train = y3_miss_hat[:y3_train_miss.shape[0]]
y3_miss_hat_test = y3_miss_hat[y3_train_miss.shape[0]:]

y1_train_miss = y1_train_miss.tolist()
y2_train_miss = y2_train_miss.tolist()
y3_train_miss = y3_train_miss.tolist()

y1_test_miss = y1_test_miss.tolist()
y2_test_miss = y2_test_miss.tolist()
y3_test_miss = y3_test_miss.tolist()

for i in range(len(y1_train_miss)):
  if not y1_train_miss[i] == '?':
    y1_miss_hat_train[i] = y1_train_miss[i]
  if not y2_train_miss[i] == '?':
    y2_miss_hat_train[i] = y2_train_miss[i]
  if not y3_train_miss[i] == '?':
    y3_miss_hat_train[i] = y3_train_miss[i]

for i in range(len(y1_test_miss)):
  if not y1_test_miss[i] == '?':
    y1_miss_hat_test[i] = y1_test_miss[i]
  if not y2_test_miss[i] == '?':
    y2_miss_hat_test[i] = y2_test_miss[i]
  if not y2_test_miss[i] == '?':
    y3_miss_hat_test[i] = y3_test_miss[i]
    
#add into X_test_miss and X_train_miss
X_train_miss.loc[:,'workclass'] = y1_miss_hat_train
X_train_miss.loc[:,'occupation'] = y2_miss_hat_train
X_train_miss.loc[:,'native.country'] = y3_miss_hat_train

X_test_miss.loc[:,'workclass'] = y1_miss_hat_test
X_test_miss.loc[:,'occupation'] = y2_miss_hat_test
X_test_miss.loc[:,'native.country'] = y3_miss_hat_test

#Now, X_test_miss and X_train_miss are full, without missing features -> add back into all
#add X_train_miss to X_train_full
X_train_final = X_train_full.append(X_train_miss, ignore_index = False)
X_train_final = X_train_final.sort_index(axis=0)
X_test_final = X_test_full.append(X_test_miss, ignore_index = False)
X_test_final = X_test_final.sort_index(axis=0)



