import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gatv2_gcn import GATv2_GCN
from utils import *
from tqdm import tqdm

dataset = "kiba"
modeling = GATv2_GCN
model_st = modeling.__name__

cuda_name = "cuda:0"

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
LR = 0.0001
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * TRAIN_BATCH_SIZE,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for idx, data in enumerate(tqdm(loader)):
            data = data[0].to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


# Main program: iterate over different datasets

print('Running on ', model_st + '_' + dataset )

train_data = Dataset(root='data', dataset=dataset+'_train')
test_data = Dataset(root='data', dataset=dataset+'_test')
        
train_size = int(0.8 * len(train_data))
valid_size = len(train_data) - train_size
train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

# make data PyTorch mini-batch processing ready
train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

# training the model
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
model = modeling().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_mse = 1000
best_test_mse = 1000
best_test_ci = 0
best_epoch = -1
model_file_name = 'model_' + model_st + '_' + dataset +  '.model'
result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'
for epoch in range(NUM_EPOCHS):
    train(model, device, train_loader, optimizer, epoch+1)
    print('predicting for valid data')
    ##### CHANGED TO TEST_LOADER
    G,P = predicting(model, device, test_loader)
    val = mse(G,P)
    if val<best_mse:
        best_mse = val
        best_epoch = epoch+1
        torch.save(model.state_dict(), model_file_name)
        print('predicting for test data')
        G,P = predicting(model, device, test_loader)
        ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
        with open(result_file_name,'w') as f:
            f.write(','.join(map(str,ret)))
        best_test_mse = ret[1]
        best_test_ci = ret[-1]
        print('rmse improved at epoch ', best_epoch, '; best_test_mse,best_test_ci:', best_test_mse,best_test_ci,model_st,dataset)
    else:
        print(ret[1],'No improvement since epoch ', best_epoch, '; best_test_mse,best_test_ci:', best_test_mse,best_test_ci,model_st,dataset)

