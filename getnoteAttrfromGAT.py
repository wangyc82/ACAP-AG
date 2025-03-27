#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:08:30 2024

@author: gabbywang
"""
import torch
import pandas as pd
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import Linear
import torch.nn as nn
import numpy as np

#get noteattribution
df = pd.read_csv('GDSCandCTRP-drug-MACCkeys.csv',delimiter=",")
df1=df.loc[:,df.columns != 'Unnamed: 0']
drug_IDs=df['Unnamed: 0']
note_attr=torch.tensor(df1.values,dtype=torch.float)
#lab = pd.Index(comps166_cids).get_indexer(drug_IDs)
#comps_maccs166_1=comps_maccs166.iloc[lab,:]
#note_attr=torch.tensor(comps_maccs166_1.values,dtype=torch.float)

df = pd.read_csv('GDSCandCTRP-DDI-adj.csv',delimiter=",")
df1=df.loc[:,df.columns != 'Unnamed: 0']
adj_t=torch.tensor(df1.values)
edge_index = adj_t.nonzero().t().contiguous()

df = pd.read_csv('GDSCandCTRP-drug_polorea.csv',delimiter=",")
y =torch.zeros(note_attr.shape[0],dtype=torch.long)
y[df['x']==1]=1
#df.loc[(df['x'] >= 73) & (df['x'] < 92), 'y'] = 1

data = Data(x=note_attr, edge_index=edge_index, y=y)

class GAT(torch.nn.Module):
    def __init__(self, input_dim, output_dim, att_dim, att_dim2, att_dim3, lin_dim):
        '''
        input_dim: the initial size/dimendion of each input sample
        output_dim: the final size/dimension of each input sample
        '''
        super().__init__()
        #the model including 3 convolutional layers and two linear transformation layers.
        self.conv1 = GATConv(in_channels = input_dim, out_channels = att_dim)
        self.conv2 = GATConv(in_channels = att_dim, out_channels = att_dim2)
        self.conv3 = GATConv(in_channels = att_dim2, out_channels = att_dim3)
        self.linear1 = Linear(att_dim3, lin_dim)
        self.linear2 = Linear(lin_dim, output_dim)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x_out = F.relu(x)
        x = self.linear1(x_out)
        x = F.relu(x)
        x = self.linear2(x)
        out = torch.sigmoid(x)


        return x_out, out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAT(note_attr.shape[1], 2, 256, 128, 50,50).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)


train_proportion = 0.8
num_train_samples = int(note_attr.shape[0] * train_proportion)

indices = np.arange(note_attr.shape[0])
np.random.shuffle(indices)

train_mask = torch.zeros(note_attr.shape[0], dtype=torch.bool)
train_mask[indices[:num_train_samples]] = True

test_mask = ~train_mask
data.train_mask=train_mask
data.test_mask=test_mask



def train():
    model.train()
    optimizer.zero_grad()
    x_new,out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) # multi-class loss
    loss.backward()
    optimizer.step()
    return x_new,out,loss.item()


# Test function
def test():
    model.eval()
    x_new,logits = model(data) 
    accs = []
    for mask in [data.train_mask, data.test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

for epoch in range(200):
    x_new,out,loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

compound=x_new.detach().numpy()
comps = pd.DataFrame(compound)
comps.index=drug_IDs
comps.to_csv("drugsGATrepresentation-cotargetDDI.csv", header = True)
