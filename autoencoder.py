import helper_functions as hf
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.datasets as datasets
import global_var as glval
from torch import nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class autoencoder(nn.Module):
    def __init__(self, size_hidden, dropout):
        super(autoencoder, self).__init__()
        self.linear1  = nn.Linear(784, size_hidden)
        self.drop     = nn.Dropout(dropout)
        self.linear2  = nn.Linear(size_hidden, 784)
        self.relu     = nn.ReLU()
        self.flatten  = Flatten()
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        return x

def train_epoch(epoch, autoenc, optimizer, lossFunc, flatten, train_loader,
        log, log_interval, train_losses_drop):
    autoenc.train()
    train_loss_drop = 0
    for batch_n, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        output = autoenc(data)
        loss = lossFunc(output, flatten(data))
        train_loss_drop += loss.detach().item()
        loss.backward()
        optimizer.step()
        if (log and batch_n % log_interval == 0):
            hf.myprint('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_n * len(data), len(train_loader.dataset),
                100. * batch_n / len(train_loader), loss.item()))
    train_loss_drop /= len(train_loader)
    hf.myprint(f"Train loss Drop: {train_loss_drop}")
    train_losses_drop.append(train_loss_drop)

def testEncoder(autoenc, test_loader, train_loader, lossFunc, flatten):
    autoenc.eval()
    test_loss  = 0
    train_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            output = autoenc(data)
            test_loss += lossFunc(output, flatten(data)).item()
        for data, _ in train_loader:
            output = autoenc(data)
            train_loss += lossFunc(output, flatten(data)).item()
    train_loss /= len(train_loader)
    test_loss /= len(test_loader)
    return (train_loss, test_loss)

def get_learning_rate_update(epoch, learning_rate):
    if(epoch%10==0 and epoch!=0):
        new_learning_rate = learning_rate/1.5
        hf.myprint(f"Lowering learning rate from {learning_rate} to {new_learning_rate}")
        return new_learning_rate
    else:
        return learning_rate

def trainAutoencoder(
    amount_hidden,
    mnist_data,
    mnist_test,
    saved_encoder  = None,
    n_epochs       = glval.n_epochs,
    optimizerFunc  = glval.optimizerFunc,
    lossFunc       = glval.lossFunc,
    learning_rate  = glval.learning_rate,
    update_lr      = False,
    has_momentum   = glval.has_momentum,
    momentum       = glval.momentum,
    batch_size     = glval.batch_size,
    dropout_rate   = glval.dropout_rate,
    log            = True,
    log_interval   = glval.log_interval):
    hf.myprint(f"Training autoencoder with of {amount_hidden} hidden layers with {n_epochs} epochs")
    mom = 0 if not has_momentum else momentum
    hf.myprint(f"learning rate: {learning_rate} - dropout rate: {dropout_rate} - momentum: {mom}")
    hf.myprint(f"Testing dataset size:{len(mnist_test)} - Training dataset size:{len(mnist_data)}")
    train_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size)
    if saved_encoder == None:
        a_encoder = autoencoder(size_hidden=amount_hidden, dropout=dropout_rate)
    else:
        hf.myprint("Loaded saved network")
        a_encoder = saved_encoder
    if has_momentum:
        optimizer = optimizerFunc(a_encoder.parameters(), lr=learning_rate, momentum=momentum)
    else:
        optimizer = optimizerFunc(a_encoder.parameters(), lr=learning_rate)
    flatten = Flatten()
    train_losses_drop = []
    train_error_arr = []
    test_error_arr = [] 
    test_results = testEncoder(a_encoder, test_loader, train_loader, lossFunc, flatten)
    train_error_arr.append(test_results[0])
    test_error_arr.append(test_results[1])
    if log:
        hf.myprint(f"Starting Train Loss: {test_results[0]} \n\tStarting Test  Loss: {test_results[1]}")
    for epoch in range(n_epochs):
        learning_rate = get_learning_rate_update(epoch, learning_rate)
        if update_lr:
            if has_momentum:
                optimizer = optimizerFunc(a_encoder.parameters(), lr=learning_rate, momentum=momentum)
            else:
                optimizer = optimizerFunc(a_encoder.parameters(), lr=learning_rate)
        train_epoch(epoch, a_encoder, optimizer, lossFunc, flatten, train_loader, log, log_interval, train_losses_drop)
        test_results = testEncoder(a_encoder, test_loader, train_loader, lossFunc, flatten)
        train_error_arr.append(test_results[0])
        test_error_arr.append(test_results[1])
        if log:
            hf.myprint(f"Current av Train Loss: {test_results[0]} \n\tCurrent av Test  Loss: {test_results[1]}")
    return_dict = {
        "autoencoder": a_encoder,
        "train_losses_drop": train_losses_drop,
        "train_error_arr": train_error_arr,
        "test_error_arr": test_error_arr
    }
    return return_dict

def compareEncoder(autoenc, mnist_test, number_range, save=False, basename='', tight=False):
    flatten = Flatten()
    if type(number_range) == type(tuple):
        rang = range(number_range[0], number_range[1])
    else:
        rang = number_range
    for i in rang:
        _, axes = plt.subplots(figsize=(10, 6), ncols=2, nrows=1)
        axes[0].imshow(mnist_test[i][0][0], cmap='gray')
        img = autoenc(flatten(mnist_test[i][0]).unsqueeze(0)).detach().numpy().reshape([28,28])
        axes[1].imshow(img, cmap = 'gray',)
        if save:
            filename = basename+f'_comp{i}.png'
            if tight:
                plt.axis('off') #doesn't work
                plt.savefig( filename,dpi=150, bbox_inches='tight', transparent="True", pad_inches=0)
            else:
                plt.savefig(filename, dpi=200)
            
            hf.myprint("\tSaved "+filename)
            plt.clf()
        else:
            plt.show()

def getOptimizer(mode):
    if mode == 0:
        hf.myprint("Using SGD as optimizer")
        dic = {
            "optimizerFunc": torch.optim.SGD,
            "has_momentum": True,
            "momentum": 0.5,
            "learning_rate": 0.5,
            "update_lr": True
        }
        return dic
    elif mode == 1:
        hf.myprint("Using Adam as optimizer")
        dic = {
            "optimizerFunc": torch.optim.Adam,
            "has_momentum": False,
            "momentum": None,
            "learning_rate": 0.001,
            "update_lr": True
        }
        return dic
    elif mode == 2:
        hf.myprint("Using RMSprop as optimizer")
        dic = {
            "optimizerFunc": torch.optim.RMSprop,
            "has_momentum": True,
            "momentum": 0.1,
            "learning_rate": 0.001,
            "update_lr": False
        }
        return dic
    else:
        raise ValueError("Incorrect Mode in getOptimizer()")    

def trainForAsigment(n_epochs, n_hidden_layers, mode, mnist_data, mnist_test):
    optimizer_dict = getOptimizer(mode)
    optimizerFunc = optimizer_dict["optimizerFunc"]
    has_momentum = optimizer_dict["has_momentum"]
    momentum = optimizer_dict["momentum"]
    learning_rate = optimizer_dict["learning_rate"]
    update_lr = optimizer_dict["update_lr"]
    lossFunc = nn.MSELoss()
    hf.myprint("Using MSE as loss function")
    dataset_full = torch.utils.data.ConcatDataset([mnist_data,mnist_test])
    new_train, new_test = torch.utils.data.random_split(dataset_full, [50000,20000])
    return trainAutoencoder(
        n_hidden_layers, new_train, new_test, saved_encoder=None,
        n_epochs=n_epochs, optimizerFunc=optimizerFunc, lossFunc=lossFunc,
        learning_rate=learning_rate, update_lr=update_lr, has_momentum=has_momentum, momentum=momentum, batch_size=1000,
        dropout_rate=0.1, log=True, log_interval=10
    )
