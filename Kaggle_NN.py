#Ryan Miller
#Machine Learning 
#Final Project
#Kaggle Compeition 

import numpy as np 
from Data_Enc import X_test_arr, X_train_arr, y_train_arr
import csv
import torch as T
import numpy as np 
import pandas as pd
import torch.nn as nn 
import torch.nn.functional as F
import csv

dispatcher ={1:1,0:0}
y_train = [dispatcher[lb] for lb in y_train_arr]
y_train_arr_2 = np.asarray(y_train)
y_train_arr_2 = np.reshape(y_train_arr_2,(len(y_train_arr_2),1))

def accuracy(model, x, y):
  X = T.Tensor(x)
  Y = T.ByteTensor(y)   
  ouput = model(X)            
  pred_y = ouput >= 0.5   
  num_correct = T.sum(Y==pred_y) 
  N = len(y)
  acc = (num_correct / N)  # scalar
  return acc

def accuracy2(model, x):
  X = T.Tensor(x)
  #Y = T.ByteTensor(y)   
  ouput = model(X)            
  pred_y = ouput >= 0.5   
  return pred_y

class Net(T.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    
    self.layer1 = T.nn.Linear(106, 400)  
    self.layer2 = T.nn.Linear(400, 800)
    self.layer3 = T.nn.Linear(800, 1600)  
    self.layer4 = T.nn.Linear(1600, 800)
    self.layer5 = T.nn.Linear(800, 400)
    self.layer6 = T.nn.Linear(400, 200)
    self.layer7 = T.nn.Linear(200, 100)
    self.layer8 = T.nn.Linear(100, 50)
    self.layer9 = T.nn.Linear(50, 25)
    self.layer10 = T.nn.Linear(25, 5)
    self.layer_out = T.nn.Linear(5, 1)
    
    T.nn.init.xavier_uniform_(self.layer1.weight)
    T.nn.init.zeros_(self.layer1.bias)
    T.nn.init.xavier_uniform_(self.layer2.weight)
    T.nn.init.zeros_(self.layer2.bias)
    T.nn.init.xavier_uniform_(self.layer3.weight)
    T.nn.init.zeros_(self.layer3.bias)
    T.nn.init.xavier_uniform_(self.layer4.weight)
    T.nn.init.zeros_(self.layer4.bias)
    T.nn.init.xavier_uniform_(self.layer5.weight)
    T.nn.init.zeros_(self.layer5.bias)
    T.nn.init.xavier_uniform_(self.layer6.weight)
    T.nn.init.zeros_(self.layer6.bias)
    T.nn.init.xavier_uniform_(self.layer7.weight)
    T.nn.init.zeros_(self.layer7.bias)
    T.nn.init.xavier_uniform_(self.layer8.weight)
    T.nn.init.zeros_(self.layer8.bias)
    T.nn.init.xavier_uniform_(self.layer9.weight)
    T.nn.init.zeros_(self.layer9.bias)
    T.nn.init.xavier_uniform_(self.layer10.weight)
    T.nn.init.zeros_(self.layer10.bias)
    T.nn.init.xavier_uniform_(self.layer_out.weight)
    T.nn.init.zeros_(self.layer_out.bias)
    
  def forward(self, inputs):
    x = T.relu(self.layer1(inputs))
    x = T.relu(self.layer2(x))
    x = T.relu(self.layer3(x))
    x = T.relu(self.layer4(x))
    x = T.relu(self.layer5(x))
    x = T.tanh(self.layer6(x))
    x = T.sigmoid(self.layer7(x))
    x = T.relu(self.layer8(x))
    x = T.relu(self.layer9(x))
    x = T.relu(self.layer10(x))
    x = T.sigmoid(self.layer_out(x))
    return x
    
def main():
  net = Net()
  net.train()
  #loss_fn = T.nn.BCELoss()  
  #loss_fn = T.nn.CrossEntropyLoss()
  loss_fn = T.nn.BCEWithLogitsLoss()
  optimizer = T.optim.Adam(net.parameters(), lr=10e-3)
  
  for epoch in range(100):
    if epoch % 10 == 0:
      print('epoch = %4d' % epoch)
      acc = accuracy(net, X_train_arr, y_train_arr_2)
      print('accuracy = %0.4f (Training Data)' % acc)
      #print(acc)
    for i in range(100):
      X_i = T.Tensor(X_train_arr[i])
      y_i = T.Tensor(y_train_arr_2[i])
      optimizer.zero_grad()
      ouput = net(X_i)
      loss_obj = loss_fn(ouput, y_i)
      loss_obj.backward()
      optimizer.step()

  net.eval()
  X_test_arr_T = T.Tensor(X_test_arr)
  y_test_hat = net(X_test_arr_T)
  y_test_hat = y_test_hat.detach().numpy()
  for i in range(len(y_test_hat)):
      if y_test_hat[i] > 0.5:
          y_test_hat[i] = 1
      else: 
          y_test_hat[i] = 0
  #y_test_hat = np.array(y_test_hat)
      
  # y_test_hat = net(X_test_arr)
  # after_training = loss_fn(y_test_hat)
  # print(after_training)
  #y_test_hat = accuracy2(net, X_test_arr)
  #print('Testing Accuracy = %0.4f' % acc_test)
  
  with open('Kaggle-NN.csv', 'w', newline='')as file:
     writer = csv.writer(file)
     writer.writerow(['ID','Prediction'])
     for i in range(1,len(y_test_hat)+1):
         writer.writerow([i,int(y_test_hat[i-1])])

if __name__=='__main__':
  main()

