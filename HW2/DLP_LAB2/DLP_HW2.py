#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    model.eval()
    
    return model


# In[2]:


from dataloader import read_bci_data

train_data, train_label, test_data, test_label = read_bci_data()
fig, ax = plt.subplots(2, 1, figsize=(16, 5))
ax[0].set_ylabel("Channel 1")
ax[0].plot(np.arange(750), train_data[0,0,0,:])
ax[1].set_ylabel("Channel 2")
ax[1].plot(np.arange(750), train_data[0,0,1,:])


# # EEGNet
# ---

# In[3]:


class EEGNet(nn.Module):
    def __init__(self, func='ELU', drop_rate=0.3):
        super(EEGNet, self).__init__()
        self.myActfunc = {
            'ELU':nn.ELU(alpha=1.0),
            'ReLU':nn.ReLU(),
            'Leaky':nn.LeakyReLU()
        }
        self.firstConv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16)
        )
        
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            self.myActfunc[func],
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=drop_rate)
        )
        
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32),
            self.myActfunc[func],
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=drop_rate)
        )
        
        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )
        
    def forward(self, x):
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.reshape(-1, 736)
        output = self.classify(x)
        return output


# In[16]:


results = {}
EPOCH = 150


# In[92]:


func = 'ELU'
drop_rate = 0.33
lr = 0.002
BATCH_SIZE = 128

train_dataset = Data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label))
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)

test_dataset = Data.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

eegnet = EEGNet(func=func, drop_rate=drop_rate)

print('Activation function : {}'.format(func))
print('Dropout rate : {}'.format(drop_rate))
print('Learning rate : {}'.format(lr))
print('Batch size : {}\n\n'.format(BATCH_SIZE))

print(eegnet)

train_acc = []
test_acc = []

optimizer = optim.Adam(eegnet.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()

for epoch in range(1, EPOCH+1):
    if epoch % 10 == 0:
        print(epoch)

    for step, (x, y) in enumerate(train_loader):   
        output = eegnet(x.float())            
        loss = loss_func(output, y.type(torch.long))   
        optimizer.zero_grad()           
        loss.backward()                 
        optimizer.step()  

    train_acc.append(accuracy_score(output.argmax(dim=1), y))

    tmp = []
    for step, (x, y) in enumerate(test_loader):   
        pred = eegnet(x.float())
        tmp.append(accuracy_score(pred.argmax(dim=1), y))
        
    test_acc.append(np.array(tmp).mean())
    
    checkpoint = {
        'model': EEGNet(func=func, drop_rate=drop_rate),
        'state_dict': eegnet.state_dict(),
        'optimizer' : optimizer.state_dict()
    }
    torch.save(checkpoint, './eegnet_ELU/epoch{}.pkl'.format(str(epoch)))


results[func] = {
    'train_ACC' : train_acc,
    'test_ACC' : test_acc
}

print("{}_ACC: {:.2f}%".format(func, np.array(results[func]['test_ACC']).max()*100))


# In[93]:


plt.figure(figsize=(10, 6))
plt.title("Activation function comparision(EEGNet)")       
for i, cond in enumerate(results.keys()):
    plt.plot(range(EPOCH), results[cond]['train_ACC'], label=cond+'_train_acc')
    plt.plot(range(EPOCH), results[cond]['test_ACC'], label=cond+'_test_acc')
    print("{}_ACC: {:.2f}%".format(cond, np.array(results[cond]['test_ACC']).max()*100))
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')
plt.yticks(np.linspace(0.5, 1, 11))
plt.legend()
plt.show()


# In[94]:


all_acc = []
for e in range(1, 151):
    model = load_checkpoint('./eegnet_ELU/epoch{}.pkl'.format(str(e)))
#     print(model)

    tmp = []
    for step, (x, y) in enumerate(test_loader):   
        pred = model(x.float())
        tmp.append(accuracy_score(pred.argmax(dim=1), y))
        
    all_acc.append(np.array(tmp).mean())
#     tmp = np.array(tmp).mean()
all_acc = np.array(all_acc)
print("[MAX] {}:  model_ACC: {}".format(all_acc.argmax(), all_acc.max()))


# In[95]:


model = load_checkpoint('./eegnet_ELU/epoch113.pkl')
print(model)

tmp = []
for step, (x, y) in enumerate(test_loader):   
    pred = model(x.float())
    tmp.append(accuracy_score(pred.argmax(dim=1), y))
tmp = np.array(tmp).mean()

print("model_ACC: {}".format(tmp))


# # Save model

# In[96]:


# print("The state dict keys: \n\n", eegnet.state_dict().keys())

# checkpoint = {'model': EEGNet(func=func, drop_rate=drop_rate),
#               'state_dict': eegnet.state_dict(),
#               'optimizer' : optimizer.state_dict()}

# torch.save(checkpoint, 'eegnet_ELU.pkl')


# ## Load model

# In[97]:


# model = load_checkpoint('eegnet_ELU.pkl')
# print(model)

# tmp = []
# for step, (x, y) in enumerate(test_loader):   
#     pred = model(x.float())
#     tmp.append(accuracy_score(pred.argmax(dim=1), y))
# tmp = np.array(tmp).mean()

# print("model_ACC: {}".format(tmp))


# In[51]:


func = 'ReLU'
drop_rate = 0.35
lr = 0.0017
BATCH_SIZE = 128

train_dataset = Data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label))
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)

test_dataset = Data.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

eegnet = EEGNet(func=func, drop_rate=drop_rate)

print('Activation function : {}'.format(func))
print('Dropout rate : {}'.format(drop_rate))
print('Learning rate : {}'.format(lr))
print('Batch size : {}\n\n'.format(BATCH_SIZE))

print(eegnet)

train_acc = []
test_acc = []

optimizer = optim.Adam(eegnet.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()

for epoch in range(1, EPOCH+1):
    if epoch % 10 == 0:
        print(epoch)

    for step, (x, y) in enumerate(train_loader):   
        output = eegnet(x.float())            
        loss = loss_func(output, y.type(torch.long))   
        optimizer.zero_grad()           
        loss.backward()                 
        optimizer.step()  

    train_acc.append(accuracy_score(output.argmax(dim=1), y))

    tmp = []
    for step, (x, y) in enumerate(test_loader):   
        pred = eegnet(x.float())
        tmp.append(accuracy_score(pred.argmax(dim=1), y))
    test_acc.append(np.array(tmp).mean())
    
    checkpoint = {
        'model': EEGNet(func=func, drop_rate=drop_rate),
        'state_dict': eegnet.state_dict(),
        'optimizer' : optimizer.state_dict()
    }
    torch.save(checkpoint, './eegnet_ReLU/epoch{}.pkl'.format(str(epoch)))
    

results[func] = {
    'train_ACC' : train_acc,
    'test_ACC' : test_acc
}

print("{}_ACC: {:.2f}%".format(func, np.array(results[func]['test_ACC']).max()*100))


# In[57]:


plt.figure(figsize=(10, 6))
plt.title("Activation function comparision(EEGNet)")       
for i, cond in enumerate(results.keys()):
    plt.plot(range(EPOCH), results[cond]['train_ACC'], label=cond+'_train_acc')
    plt.plot(range(EPOCH), results[cond]['test_ACC'], label=cond+'_test_acc')
    print("{}_ACC: {:.2f}%".format(cond, np.array(results[cond]['test_ACC']).max()*100))
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')
plt.yticks(np.linspace(0.5, 1, 11))
plt.legend()
plt.show()


# In[38]:


checkpoint = {'model': EEGNet(func=func, drop_rate=drop_rate),
              'state_dict': eegnet.state_dict(),
              'optimizer' : optimizer.state_dict()}

torch.save(checkpoint, 'eegnet_ReLU.pkl')


# In[53]:


all_acc = []
for e in range(1, 151):
    model = load_checkpoint('./eegnet_ReLU/epoch{}.pkl'.format(str(e)))
#     print(model)

    tmp = []
    for step, (x, y) in enumerate(test_loader):   
        pred = model(x.float())
        tmp.append(accuracy_score(pred.argmax(dim=1), y))
        
    all_acc.append(np.array(tmp).mean())
#     tmp = np.array(tmp).mean()
all_acc = np.array(all_acc)
print("[MAX] {}:  model_ACC: {}".format(all_acc.argmax(), all_acc.max()))


# In[146]:


all_acc


# In[56]:


np.array(results["ReLU"]['test_ACC'])


# In[69]:


model = load_checkpoint('./eegnet_ReLU/epoch135.pkl')
print(model)

tmp = []
for step, (x, y) in enumerate(test_loader):   
    pred = model(x.float())
    tmp.append(accuracy_score(pred.argmax(dim=1), y))
tmp = np.array(tmp).mean()

print("model_ACC: {}".format(tmp))


# In[62]:


func = 'Leaky'
drop_rate = 0.4
lr = 0.0025
BATCH_SIZE = 256

train_dataset = Data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label))
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)

test_dataset = Data.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

eegnet = EEGNet(func=func, drop_rate=drop_rate)

print('Activation function : {}'.format(func))
print('Dropout rate : {}'.format(drop_rate))
print('Learning rate : {}'.format(lr))
print('Batch size : {}\n\n'.format(BATCH_SIZE))

print(eegnet)

train_acc = []
test_acc = []

optimizer = optim.Adam(eegnet.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()

for epoch in range(1, EPOCH+1):
    if epoch % 10 == 0:
        print(epoch)

    for step, (x, y) in enumerate(train_loader):   
        output = eegnet(x.float())            
        loss = loss_func(output, y.type(torch.long))   
        optimizer.zero_grad()           
        loss.backward()                 
        optimizer.step()  

    train_acc.append(accuracy_score(output.argmax(dim=1), y))

    tmp = []
    for step, (x, y) in enumerate(test_loader):   
        pred = eegnet(x.float())
        tmp.append(accuracy_score(pred.argmax(dim=1), y))
    test_acc.append(np.array(tmp).mean())
    
    checkpoint = {
        'model': EEGNet(func=func, drop_rate=drop_rate),
        'state_dict': eegnet.state_dict(),
        'optimizer' : optimizer.state_dict()
    }
    torch.save(checkpoint, './eegnet_Leaky/epoch{}.pkl'.format(str(epoch)))

results[func] = {
    'train_ACC' : train_acc,
    'test_ACC' : test_acc
}

print("{}_ACC: {:.2f}%".format(func, np.array(results[func]['test_ACC']).max()*100))


# In[63]:


plt.figure(figsize=(10, 6))
plt.title("Activation function comparision(EEGNet)")       
for i, cond in enumerate(results.keys()):
    plt.plot(range(EPOCH), results[cond]['train_ACC'], label=cond+'_train_acc')
    plt.plot(range(EPOCH), results[cond]['test_ACC'], label=cond+'_test_acc')
    print("{}_ACC: {:.2f}%".format(cond, np.array(results[cond]['test_ACC']).max()*100))
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')
plt.yticks(np.linspace(0.5, 1, 11))
plt.legend()
plt.show()


# In[64]:


all_acc = []
for e in range(1, 151):
    model = load_checkpoint('./eegnet_Leaky/epoch{}.pkl'.format(str(e)))
#     print(model)

    tmp = []
    for step, (x, y) in enumerate(test_loader):   
        pred = model(x.float())
        tmp.append(accuracy_score(pred.argmax(dim=1), y))
        
    all_acc.append(np.array(tmp).mean())
#     tmp = np.array(tmp).mean()
all_acc = np.array(all_acc)
print("[MAX] {}:  model_ACC: {}".format(all_acc.argmax(), all_acc.max()))


# In[65]:


model = load_checkpoint('./eegnet_Leaky/epoch103.pkl')
print(model)

tmp = []
for step, (x, y) in enumerate(test_loader):   
    pred = model(x.float())
    tmp.append(accuracy_score(pred.argmax(dim=1), y))
tmp = np.array(tmp).mean()

print("model_ACC: {}".format(tmp))


# In[123]:


a = pd.DataFrame(data=results['ELU']).rename(columns={'train_ACC':'ELU_train', 'test_ACC':'ELU_test'})
b = pd.DataFrame(data=results['ReLU']).rename(columns={'train_ACC':'ReLU_train', 'test_ACC':'ReLU_test'})
c = pd.DataFrame(data=results['Leaky']).rename(columns={'train_ACC':'Leaky_train', 'test_ACC':'Leaky_test'})

res = pd.concat([a, b, c], axis=1)
res.plot()
res.to_csv('eegnet.csv', index=False)


# # DeepConvNet
# ---

# In[124]:


EPOCH = 150
BATCH_SIZE = 256

train_dataset = Data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label))
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)

test_dataset = Data.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)


# In[125]:


class DeepConvNet(nn.Module):
    def __init__(self, func='ELU', drop_rate=0.3):
        super(DeepConvNet, self).__init__()
        
        self.myActfunc = {
            'ELU':nn.ELU(alpha=1.0),
            'ReLU':nn.ReLU(),
            'Leaky':nn.LeakyReLU()
        }
        
        self.firstLayer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=False),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(2, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(25),
            self.myActfunc[func],
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),
            nn.Dropout(p=drop_rate)
        )
        
        self.secondLayer = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=False),
            nn.BatchNorm2d(50),
            self.myActfunc[func],
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),
            nn.Dropout(p=drop_rate)
        )
        
        self.thirdLayer = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=False),
            nn.BatchNorm2d(100),
            self.myActfunc[func],
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),
            nn.Dropout(p=drop_rate)
        )
        
        self.fourthLayer = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=False),
            nn.BatchNorm2d(200),
            self.myActfunc[func],
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),
            nn.Dropout(p=drop_rate)
        )
        
        self.classify = nn.Sequential(
            nn.Linear(in_features=9200, out_features=2, bias=True)
        )
        
    def forward(self, x):
        x = self.firstLayer(x)
        x = self.secondLayer(x)
        x = self.thirdLayer(x)
        x = self.fourthLayer(x)
        x = x.reshape(-1, 9200)
        output = self.classify(x)
        return output


# In[132]:


results2 = {}
EPOCH = 150


# In[133]:


func = 'ELU'
drop_rate = 0.35
lr = 0.0015
BATCH_SIZE = 256

train_dataset = Data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label))
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)

test_dataset = Data.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

deepconvnet = DeepConvNet(func=func, drop_rate=drop_rate)
print('Activation function : {}'.format(func))
print('Dropout rate : {}'.format(drop_rate))
print('Learning rate : {}'.format(lr))
print('Batch size : {}\n\n'.format(BATCH_SIZE))
print(deepconvnet)

optimizer = optim.Adam(deepconvnet.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()

train_acc2 = []
test_acc2 = []

for epoch in range(1, EPOCH+1):
    if epoch % 10 == 0:
        print(epoch)
        
    for step, (x, y) in enumerate(train_loader):   
        output = deepconvnet(x.float())               
        loss = loss_func(output, y.type(torch.long))   
        optimizer.zero_grad()           
        loss.backward()                 
        optimizer.step()  

    train_acc2.append(accuracy_score(output.argmax(dim=1), y))
    tmp = []
    for step, (x, y) in enumerate(test_loader):   
        pred = deepconvnet(x.float())
        tmp.append(accuracy_score(pred.argmax(dim=1), y))
    test_acc2.append(np.array(tmp).mean())
    
    checkpoint = {
        'model': DeepConvNet(func=func, drop_rate=drop_rate),
        'state_dict': deepconvnet.state_dict(),
        'optimizer' : optimizer.state_dict()
    }
    torch.save(checkpoint, './deepconv_ELU/epoch{}.pkl'.format(str(epoch)))

results2[func] = {
    'train_ACC' : train_acc2,
    'test_ACC' : test_acc2
}
print("{}_ACC: {:.2f}%".format(func, np.array(results2[func]['test_ACC']).max()*100))


# In[134]:


plt.figure(figsize=(10, 6))
plt.title("Activation function comparision(DeepConvNet)")       
for i, cond in enumerate(results2.keys()):
    plt.plot(range(EPOCH), results2[cond]['train_ACC'], label=cond+'_train_acc')
    plt.plot(range(EPOCH), results2[cond]['test_ACC'], label=cond+'_test_acc')
    print("{}_ACC: {:.2f}%".format(cond, np.array(results2[cond]['test_ACC']).max()*100))
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')
plt.yticks(np.linspace(0.5, 1, 11))
plt.legend()
plt.show()


# In[135]:


all_acc = []
for e in range(1, 151):
    model = load_checkpoint('./deepconv_ELU/epoch{}.pkl'.format(str(e)))
#     print(model)

    tmp = []
    for step, (x, y) in enumerate(test_loader):   
        pred = model(x.float())
        tmp.append(accuracy_score(pred.argmax(dim=1), y))
        
    all_acc.append(np.array(tmp).mean())
#     tmp = np.array(tmp).mean()
all_acc = np.array(all_acc)
print("[MAX] {}:  model_ACC: {}".format(all_acc.argmax(), all_acc.max()))


# In[136]:


model = load_checkpoint('./deepconv_ELU/epoch120.pkl')
print(model)

tmp = []
for step, (x, y) in enumerate(test_loader):   
    pred = model(x.float())
    tmp.append(accuracy_score(pred.argmax(dim=1), y))
tmp = np.array(tmp).mean()

print("model_ACC: {}".format(tmp))


# In[137]:


func = 'ReLU'
drop_rate = 0.35
lr = 0.0015
BATCH_SIZE = 256

train_dataset = Data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label))
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)

test_dataset = Data.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

deepconvnet = DeepConvNet(func=func, drop_rate=drop_rate)
print('Activation function : {}'.format(func))
print('Dropout rate : {}'.format(drop_rate))
print('Learning rate : {}'.format(lr))
print('Batch size : {}\n\n'.format(BATCH_SIZE))
print(deepconvnet)

optimizer = optim.Adam(deepconvnet.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()

train_acc2 = []
test_acc2 = []

for epoch in range(1, EPOCH+1):
    if epoch % 10 == 0:
        print(epoch)
        
    for step, (x, y) in enumerate(train_loader):   
        output = deepconvnet(x.float())               
        loss = loss_func(output, y.type(torch.long))   
        optimizer.zero_grad()           
        loss.backward()                 
        optimizer.step()  

    train_acc2.append(accuracy_score(output.argmax(dim=1), y))
    tmp = []
    for step, (x, y) in enumerate(test_loader):   
        pred = deepconvnet(x.float())
        tmp.append(accuracy_score(pred.argmax(dim=1), y))
    test_acc2.append(np.array(tmp).mean())
    
    checkpoint = {
        'model': DeepConvNet(func=func, drop_rate=drop_rate),
        'state_dict': deepconvnet.state_dict(),
        'optimizer' : optimizer.state_dict()
    }
    torch.save(checkpoint, './deepconv_ReLU/epoch{}.pkl'.format(str(epoch)))

results2[func] = {
    'train_ACC' : train_acc2,
    'test_ACC' : test_acc2
}
print("{}_ACC: {:.2f}%".format(func, np.array(results2[func]['test_ACC']).max()*100))


# In[138]:


plt.figure(figsize=(10, 6))
plt.title("Activation function comparision(DeepConvNet)")       
for i, cond in enumerate(results2.keys()):
    plt.plot(range(EPOCH), results2[cond]['train_ACC'], label=cond+'_train_acc')
    plt.plot(range(EPOCH), results2[cond]['test_ACC'], label=cond+'_test_acc')
    print("{}_ACC: {:.2f}%".format(cond, np.array(results2[cond]['test_ACC']).max()*100))
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')
plt.yticks(np.linspace(0.5, 1, 11))
plt.legend()
plt.show()


# In[139]:


all_acc = []
for e in range(1, 151):
    model = load_checkpoint('./deepconv_ReLU/epoch{}.pkl'.format(str(e)))
#     print(model)

    tmp = []
    for step, (x, y) in enumerate(test_loader):   
        pred = model(x.float())
        tmp.append(accuracy_score(pred.argmax(dim=1), y))
        
    all_acc.append(np.array(tmp).mean())
#     tmp = np.array(tmp).mean()
all_acc = np.array(all_acc)
print("[MAX] {}:  model_ACC: {}".format(all_acc.argmax(), all_acc.max()))


# In[140]:


model = load_checkpoint('./deepconv_ReLU/epoch74.pkl')
print(model)

tmp = []
for step, (x, y) in enumerate(test_loader):   
    pred = model(x.float())
    tmp.append(accuracy_score(pred.argmax(dim=1), y))
tmp = np.array(tmp).mean()

print("model_ACC: {}".format(tmp))


# In[141]:


func = 'Leaky'
drop_rate = 0.4
lr = 0.0015
BATCH_SIZE = 256

train_dataset = Data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label))
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)

test_dataset = Data.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

deepconvnet = DeepConvNet(func=func, drop_rate=drop_rate)
print('Activation function : {}'.format(func))
print('Dropout rate : {}'.format(drop_rate))
print('Learning rate : {}'.format(lr))
print('Batch size : {}\n\n'.format(BATCH_SIZE))
print(deepconvnet)

optimizer = optim.Adam(deepconvnet.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()

train_acc2 = []
test_acc2 = []

for epoch in range(1, EPOCH+1):
    if epoch % 10 == 0:
        print(epoch)
        
    for step, (x, y) in enumerate(train_loader):   
        output = deepconvnet(x.float())               
        loss = loss_func(output, y.type(torch.long))   
        optimizer.zero_grad()           
        loss.backward()                 
        optimizer.step()  

    train_acc2.append(accuracy_score(output.argmax(dim=1), y))
    tmp = []
    for step, (x, y) in enumerate(test_loader):   
        pred = deepconvnet(x.float())
        tmp.append(accuracy_score(pred.argmax(dim=1), y))
    test_acc2.append(np.array(tmp).mean())
    
    checkpoint = {
        'model': DeepConvNet(func=func, drop_rate=drop_rate),
        'state_dict': deepconvnet.state_dict(),
        'optimizer' : optimizer.state_dict()
    }
    torch.save(checkpoint, './deepconv_Leaky/epoch{}.pkl'.format(str(epoch)))

results2[func] = {
    'train_ACC' : train_acc2,
    'test_ACC' : test_acc2
}
print("{}_ACC: {:.2f}%".format(func, np.array(results2[func]['test_ACC']).max()*100))


# In[142]:


plt.figure(figsize=(10, 6))
plt.title("Activation function comparision(DeepConvNet)")       
for i, cond in enumerate(results2.keys()):
    plt.plot(range(EPOCH), results2[cond]['train_ACC'], label=cond+'_train_acc')
    plt.plot(range(EPOCH), results2[cond]['test_ACC'], label=cond+'_test_acc')
    print("{}_ACC: {:.2f}%".format(cond, np.array(results2[cond]['test_ACC']).max()*100))
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')
plt.yticks(np.linspace(0.5, 1, 11))
plt.legend()
plt.show()


# In[143]:


all_acc = []
for e in range(1, 151):
    model = load_checkpoint('./deepconv_Leaky/epoch{}.pkl'.format(str(e)))
#     print(model)

    tmp = []
    for step, (x, y) in enumerate(test_loader):   
        pred = model(x.float())
        tmp.append(accuracy_score(pred.argmax(dim=1), y))
        
    all_acc.append(np.array(tmp).mean())
#     tmp = np.array(tmp).mean()
all_acc = np.array(all_acc)
print("[MAX] {}:  model_ACC: {}".format(all_acc.argmax(), all_acc.max()))


# In[144]:


model = load_checkpoint('./deepconv_Leaky/epoch142.pkl')
print(model)

tmp = []
for step, (x, y) in enumerate(test_loader):   
    pred = model(x.float())
    tmp.append(accuracy_score(pred.argmax(dim=1), y))
tmp = np.array(tmp).mean()

print("model_ACC: {}".format(tmp))


# In[145]:


a = pd.DataFrame(data=results2['ELU']).rename(columns={'train_ACC':'ELU_train', 'test_ACC':'ELU_test'})
b = pd.DataFrame(data=results2['ReLU']).rename(columns={'train_ACC':'ReLU_train', 'test_ACC':'ReLU_test'})
c = pd.DataFrame(data=results2['Leaky']).rename(columns={'train_ACC':'Leaky_train', 'test_ACC':'Leaky_test'})

res = pd.concat([a, b, c], axis=1)
res.plot()
res.to_csv('deepconv.csv', index=False)


# In[ ]:




