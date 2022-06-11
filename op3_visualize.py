

from turtle import color
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
 

load_torch = torch.load('con_layer.pt')
dataMat = load_torch.clone().detach().cpu().numpy()
print(load_torch.size())
load_torch = torch.load('hidden1.pt')
dataMat2 = load_torch.clone().detach().cpu().numpy()

# load_torch2 = torch.load('fc_layer.pt')
# print(load_torch2.size())
load_torch3 = torch.load('final_layer.pt')
print(load_torch3.size())
# print(b)
# dataMat = np.random.random((600,8))
# dataMat = np.random.normal(size=(120,8))
# dataMat2 = np.random.normal(size=(120,8))
# pca_sk = PCA(n_components=2)
# newMat = pca_sk.fit_transform(dataMat) 
# newMat2 = pca_sk.fit_transform(dataMat2) 
# plt.scatter(newMat[:, 0], newMat[:, 1],marker='1',color = 'g', label = '0')
# plt.scatter(newMat2[:, 0], newMat2[:, 1],marker='1',color = 'r', label = '1')
# plt.title('PCA visulization of hidden layer output')
# plt.legend()
# plt.show()

# dataMat = np.random.normal(size=(120,8))
# dataMat2 = np.random.normal(size=(120,8))

# newMat = TSNE(n_components=2,learning_rate=100).fit_transform(dataMat)
# newMat2 = TSNE(n_components=2,learning_rate=100).fit_transform(dataMat2)
# plt.scatter(newMat[:, 0], newMat[:, 1],marker='1',color = 'g', label = '0')
# plt.scatter(newMat2[:, 0], newMat2[:, 1],marker='1',color = 'r', label = '1')
# plt.title('TSNE visulization of hidden layer output')
# plt.legend()
# plt.show()

# accuracy loss
epoch = np.array([0,1,2,3,4,5,6,7,8,9,10])
# accuracy = np.array([0.525,0.601,0.655,0.676,0.687,0.703,0.710,0.727,0.722,0.725])
# accuracy = np.array([0.505,0.571,0.625,0.656,0.657,0.663,0.670,0.687,0.682,0.685])
# loss = np.array([0.534,0.405,0.353,0.321,0.301,0.283,0.274,0.264,0.259,0.254])
# loss = np.array([0.539,0.477,0.3,0.549,0.629,0.621,0.676,0.606,0.620,0.574])
# loss = np.array([0.1042,0.0527,0.0401,0.034,0.0311,0.0300,0.0368,0.0298,0.03,0.0336])

accuracy = np.array([0,0.96,0.98,0.973,0.98,0.978,0.980,0.985,0.981,0.990,0.986])

plt.plot(epoch,accuracy,linestyle = '-',color = 'b')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.title('test accuracy')
plt.legend()
plt.show()
