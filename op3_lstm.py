import torch
import torchtext
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
import tqdm
import sys
import numpy as np
import math

from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
 

train_iter = torchtext.datasets.SST2(root = "./data", split = "dev")
# print(train_iter)

# 创建分词器
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
print(tokenizer('here is an example'))

# 构建词汇表
def yield_tokens(data_iter):
    # for _, text in data_iter:
    for text, _ in data_iter:
        yield tokenizer(text)
# <pad>和<unk>，分别表示占位符和未登录词
vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])

print(vocab(tokenizer('here is the an example <pad> <pad>')))

# 数据处理pipeline
text_pipeline = lambda x: vocab(tokenizer(x))
print(text_pipeline("this is an exaple blll"))

# 生成训练数据dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    max_length = 25
    pad = text_pipeline('<pad>')
    label_list, text_list, length_list = [], [], []
    # for (_label, _text) in batch:
    for (_text, _label) in batch:
        #  label_list.append(label_pipeline(_label))
        label_list.append(_label)
        processed_text = text_pipeline(_text)[:max_length]
        length_list.append(len(processed_text))
        text_list.append((processed_text+pad*max_length)[:max_length])
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.tensor(text_list, dtype=torch.int64)
    length_list = torch.tensor(length_list, dtype=torch.int64)
    return label_list.to(device), text_list.to(device), length_list.to(device)

train_iter = torchtext.datasets.SST2(root='./data', split='train')
valid_iter = torchtext.datasets.SST2(root='./data', split='dev')
train_dataset = to_map_style_dataset(train_iter)
valid_dataset = to_map_style_dataset(valid_iter)


train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)

# 模型
class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional,
                 dropout_rate, pad_index=0):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional,
                            dropout=dropout_rate, batch_first=True)

        # mylstm = myLstm()
        # self.lstm = mylstm(embedding_dim, hidden_dim)

        self.fc = torch.nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        
    def forward(self, ids, length):
        embedded = self.dropout(self.embedding(ids))
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True, 
                                                            enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_length = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
        else:
            hidden = self.dropout(hidden[-1])
        prediction = self.fc(hidden)
        torch.save(hidden, 'hidden1.pt')
        return prediction

# self defined lstm

class myLstm(torch.nn.Module):
  def __intit(self,input_sz,hidden_sz):
    super().__init__()
    self.input_size=input_sz
    self.hidden_size=hidden_sz
    self.U_i = torch.nn.Parameter(torch.Tensor(input_sz,hidden_sz))
    self.V_i = torch.nn.Parameter(torch.Tensor(hidden_sz,hidden_sz))
    self.b_i = torch.nn.parameter(torch.Tensor(hidden_sz))


    #f_t
    self.U_f = torch.nn.Parameter(torch.Tensor(input_sz, hidden_sz))
    self.V_f = torch.nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
    self.b_f = torch.nn.Parameter(torch.Tensor(hidden_sz))
 
    #c_t
    self.U_c = torch.nn.Parameter(torch.Tensor(input_sz, hidden_sz))
    self.V_c = torch.nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
    self.b_c = torch.nn.Parameter(torch.Tensor(hidden_sz))
 
    #o_t
    self.U_o = torch.nn.Parameter(torch.Tensor(input_sz, hidden_sz))
    self.V_o = torch.nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
    self.b_o = torch.nn.Parameter(torch.Tensor(hidden_sz))
 
    self.init_weights()
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




# 实例化模型
vocab_size = len(vocab)
embedding_dim = 300
hidden_dim = 300
output_dim = 2
n_layers = 2
bidirectional = True
dropout_rate = 0.5

model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate)
model = model.to(device)

# 损失函数与优化方法
lr = 5e-4
criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy

def train(dataloader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(dataloader, desc='training...', file=sys.stdout):
        (label, ids, length) = batch
        label = label.to(device)
        ids = ids.to(device)
        length = length.to(device)
        prediction = model(ids, length)
        loss = criterion(prediction, label) # loss计算
        accuracy = get_accuracy(prediction, label)
        # 梯度更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return epoch_losses, epoch_accs

def evaluate(dataloader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc='evaluating...', file=sys.stdout):
            (label, ids, length) = batch
            label = label.to(device)
            ids = ids.to(device)
            length = length.to(device)
            prediction = model(ids, length)
            loss = criterion(prediction, label) # loss计算
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return epoch_losses, epoch_accs

n_epochs = 10
best_valid_loss = float('inf')

train_losses = []
train_accs = []
valid_losses = []
valid_accs = []

for epoch in range(n_epochs):
    train_loss, train_acc = train(train_dataloader, model, criterion, optimizer, device)
    valid_loss, valid_acc = evaluate(valid_dataloader, model, criterion, device)
    train_losses.extend(train_loss)
    train_accs.extend(train_acc)
    valid_losses.extend(valid_loss)
    valid_accs.extend(valid_acc) 
    epoch_train_loss = np.mean(train_loss)
    epoch_train_acc = np.mean(train_acc)
    epoch_valid_loss = np.mean(valid_loss)
    epoch_valid_acc = np.mean(valid_acc)    
    if epoch_valid_loss < best_valid_loss:
        best_valid_loss = epoch_valid_loss
        torch.save(model.state_dict(), 'lstm.pt')   
    print(f'epoch: {epoch+1}')
    print(f'train_loss: {epoch_train_loss:.3f}, train_acc: {epoch_train_acc:.3f}')
    print(f'valid_loss: {epoch_valid_loss:.3f}, valid_acc: {epoch_valid_acc:.3f}')