# report of optional task 1

本报告为机器学习大作业optional task1的实验报告。
姓名：金佳绒
班级：F1903003
学号：519030910058

## 实验环境

编程语言：python 3.7.2
框架：pytorch 1.4.0
数据集：fashion-MNIST

## 实验任务

```
introduce the overall architecture of the VAE, which is designed by yourself. It would be better if you draw a diagram.
how you implement the VAE that is designed by yourself
visualize reconstructed images.
use interpolation to generate images with different properties
```

## VAE

VAE = Variational auto-encoder, 即变分自编码器。
VAE的motivition来自于所谓的自编码器，即包含input->code->output的过程的模型。自编码器包含一个编码器encoder和一个解码器decoder，满足$z = g(X), \hat{X} = f(z)$，其中g为编码器，f为解码器，z为编码，X为数据集。
我们希望数据X经过编码器和解码器后得到的$\hat{X}$可以和原来的X尽可能接近。如果能在此基础上显性的对$z = p(x)$的分布进行建模，使自编码器不止对少数编码z有效解码，成为一个合格的生成模型，我们就得到了变分自编码器。

#### VAE的一般架构

![image](https://github.com/TinaJin0228/ml_final/blob/main/op1_arch2.jpg)

我们通过神经网络建立、计算decoder和encoder的模型。
数据点$x_i$在encoder输入后，通过神经网络，我们得到隐变量z近似服从的后验概率分布$q(z|x_i)$的参数（我们认为后验分布是一个各向同性的高斯分布）。
得到近似后验分布后，我们先从$N(0,I)$中采样得到$\epsilon_i$，然后令$z_i = \mu_i + \sigma_i * \epsilon_i$.
(即**reparameterization trick**, 保证整个架构前向/后向传播的通畅)
我们假设decoder服从近似的先验概率分布$p(X|z_i)$，并且也认为这个分布近似服从一个各向同性的高斯分布。因此decoder也会输出相应的高斯分布的参数。
在得到$X|z_i$近似服从的分布后，理论上我们可以通过在这个分布中采样来得到可能的数据点$x_i$，在fashion-MNIST数据集上即为重建的图片。
在实际操作中往往将输出的均值参数$\mu_i$
作为重新生成的数据点
$x_i$。

#### 实际建立的VAE架构

![image](https://github.com/TinaJin0228/ml_final/blob/main/op1_arch3.jpg)

![image](https://github.com/TinaJin0228/ml_final/blob/main/op1_arch4.jpg)

#### result of interpolation

![image](https://github.com/TinaJin0228/ml_final/blob/main/op1_res.jpg)

## reference

[# 机器学习理论—优雅的模型（一）：变分自编码器（VAE）](https://zhuanlan.zhihu.com/p/348498294)
