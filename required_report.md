# report of required task

本报告为机器学习大作业required task的实验报告。
姓名：金佳绒
班级：F1903003
学号：519030910058

## 实验环境

编程语言：python 3.7.2
框架：pytorch 1.4.0
数据集：MNIST
数据集尺寸：
training set：60000 samples
test set：10000 samples

## 实验任务

```
design one or two neural network by yourself. Specifically, each designed neural network should contain 12-35 layers, including convolutional layers, ReLU layers, Batch Normalization layers, fully connected layers, and maxpooling layers
introduce architectures of neural networks designed by yourself and explain why you use such an architecture.
report the classification accuracy on the test set, report
the classification accuracy and loss on the training set
use both PCA and T-SNE to visualize features
select at least three layers of the aforementioned network onto conduct PCA and t-SNE, and visualize g′(X). The selected layers should at least include a convolutional layer, a fully-connected layer, and the final layer of the neural network.
```

## neural network设计结构

task：利用神经网络对MNIST数据集的数字分类

diagram:
![image](https://github.com/TinaJin0228/ml_final/blob/main/re_diagram.jpg)

设计思路：
网络包含两个卷积层，两个BN层，两个池化层，以及两个全连接层以及中间的ReLU层以及最后的log softmax层。
卷积层的作用是提取特征，同时保证了网络的稀疏性，防止过拟合。
池化层（maxP = max pooling）的作用是降维，在保留特征的同时降低卷积核的尺寸。
卷积层和池化层中间ReLU层作为激活函数，计算简单，输出稀疏点，是实践中非常常用的搭配。加上BN层可以加快网络的训练和收敛的速度，控制梯度爆炸防止梯度消失，防止过拟合
以上部分网络的作用是特征提取。
全连接层的作用：放在网络最后，形成一个端到端的结构，将提取到的特征分类。

##### visulization the intermeidate-layer features using PCA and t-SNE

对第二个convolutional layer、第一个fully-connected layer和输出层的特征分别进行PCA/tSNE降维操作，并可视化
![image](https://github.com/TinaJin0228/ml_final/blob/main/re_pca_con.png)
![image](https://github.com/TinaJin0228/ml_final/blob/main/re_tsne_con.png)
![image](https://github.com/TinaJin0228/ml_final/blob/main/re_pca_fc.png)
![image](https://github.com/TinaJin0228/ml_final/blob/main/re_tsne_fc.png)
![image](https://github.com/TinaJin0228/ml_final/blob/main/re_pca_fi.png)
![image](https://github.com/TinaJin0228/ml_final/blob/main/re_tsne_fi.png)

##### the classification loss and accuracy of both the training set and validation set

测试集上的平均正确率达到95%+，说明神经网络对MNIST数据集的分类效果很好。

![image](https://github.com/TinaJin0228/ml_final/blob/main/re_test_accuracy.png)

![image](https://github.com/TinaJin0228/ml_final/blob/main/re_train_accuracy.png)

![image](https://github.com/TinaJin0228/ml_final/blob/main/re_train_loss.png)
