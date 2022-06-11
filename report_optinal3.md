# report of optional task 3

本报告为机器学习大作业optional task 3的实验报告。
姓名：金佳绒
班级：F1903003
学号：519030910058

## 实验环境

编程语言：python 3.7.2
框架：pytorch 1.4.0
数据集：SST-2


## 实验任务

```
Design and implement an LSTM for binary sentiment classification task and introduce its architecture by drawing a diagram.
Use both PCA and T-SNE to visualize the intermeidate-layer features.
Report the classification loss and accuracy of both the training set and validation set.
Analyze the experimental results.
```

## LSTM

#### 设计思路与LSTM结构

task：利用LSTM对句子数据集的二分情感分析
思路：

##### 数据预处理

1. 利用pytorch提供的词汇处理工具，创建分词器和词汇表
   利用torchtext的get_tokenizer()函数构建分词器
   ![image](https://github.com/TinaJin0228/ml_final/blob/main/op3_tokenizer.jpg)
   利用torchtext中的datasets属性获得SST-2数据集，并使用该数据集和分词器创建词汇表。
   build_vocab_from_iterator 函数可以帮助我们使用训练数据集的迭代器构建词汇表，构建好词汇表后，输入分词后的结果，即可返回每个词语的 id。同时自定义了两个词语：“<pad>”和“<unk>”，分别表示占位符和词表外的词语。
   ![image](https://github.com/TinaJin0228/ml_final/blob/main/op3_vocab.jpg)
2. 生成训练数据dataloader
   由于数据集中每个评价句子的长度不同，为了生成相同长度的特征词向量，限定特征词向量的长度为25，长则截断、短则补0。
   ![image](https://github.com/TinaJin0228/ml_final/blob/main/op3_dataloader.jpg)

##### 模型定义

###### LSTM cell

LSTM由门控单元（一种可由数学运算后控制信息是否传递或保持的计算结构）组成，这种结构使得模型可以决定是在长期还是短期记忆中进行输出。
![image](https://github.com/TinaJin0228/ml_final/blob/main/op3_lstm_cell.jpg)

###### 遗忘门

![image](https://github.com/TinaJin0228/ml_final/blob/main/op3_lstm_f.jpg)

遗忘单元可以将输入信息和隐藏信息进行信息整合，并进行信息更替，更替步骤如右图公式

###### 输入门

![image](https://github.com/TinaJin0228/ml_final/blob/main/op3_lstm_i.jpg)

输入门将进入信息与隐藏信息整合并使用tanh激活函数控制定义域为(-1,1)

###### 输出门

![image](https://github.com/TinaJin0228/ml_final/blob/main/op3_lstm_o.jpg)

输出门也使用tanh激活函数将定义域控制为(-1,1)

###### myLSTM

![image](https://github.com/TinaJin0228/ml_final/blob/main/op3_diagram.jpg)

forgrt gate：

```
#f_t
    self.U_f = torch.nn.Parameter(torch.Tensor(input_sz, hidden_sz))
    self.V_f = torch.nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
    self.b_f = torch.nn.Parameter(torch.Tensor(hidden_sz))
```

input：

```
self.input_size=input_sz
    self.hidden_size=hidden_sz
    self.U_i = torch.nn.Parameter(torch.Tensor(input_sz,hidden_sz))
    self.V_i = torch.nn.Parameter(torch.Tensor(hidden_sz,hidden_sz))
    self.b_i = torch.nn.parameter(torch.Tensor(hidden_sz))
```

output：

```
#o_t
    self.U_o = torch.nn.Parameter(torch.Tensor(input_sz, hidden_sz))
    self.V_o = torch.nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
    self.b_o = torch.nn.Parameter(torch.Tensor(hidden_sz))
```

forward：

```
def forward(self,x,init_states=None):
      bs,seq_sz,_=x.size()
      hidden_seq=[]

      if init_states is None:
        h_t,c_t=(
            torch.zeros(bs,self.hidden_size).to(x.device),
            torch.zeros(bs,self.hidden_size).to(x.device)
        )
      else:
        h_t, c_t = init_states
      for t in range(seq_sz):
        x_t = x[:, t, :]

        i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
        f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
        g_t = torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
        o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        hidden_seq.append(h_t.unsqueeze(0))
      hidden_seq = torch.cat(hidden_seq, dim=0)
      hidden_seq = hidden_seq.transpose(0, 1).contiguous()
      return hidden_seq, (h_t, c_t)
```

##### visulization the intermeidate-layer features using PCA and t-SNE

![image](https://github.com/TinaJin0228/ml_final/blob/main/op3_pca1.png)

![image](https://github.com/TinaJin0228/ml_final/blob/main/op3_tsne1.png)

##### the classification loss and accuracy of both the training set and validation set

![image](https://github.com/TinaJin0228/ml_final/blob/main/op3_train_accuracy.png)

![image](https://github.com/TinaJin0228/ml_final/blob/main/op3_validation_accuracy.png)

![image](https://github.com/TinaJin0228/ml_final/blob/main/op3_train_loss.png)

![image](https://github.com/TinaJin0228/ml_final/blob/main/op3_validation_loss.png)



## reference

@incollection{SocherEtAl2013:RNTN,
title = {{Parsing With Compositional Vector Grammars}},
author = {Richard Socher and Alex Perelygin and Jean Wu and Jason Chuang and Christopher Manning and Andrew Ng and Christopher Potts},
booktitle = {{EMNLP}},
year = {2013}
}
