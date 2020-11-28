# Agustin Marcelo Dominguez - Nov 2020

import helper_functions as hf
hf.lineprint("Loading libraries...")
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.datasets as datasets
from torch import nn
import autoencoder as ac
import global_var as glval
import warnings
import seaborn as sns
import json

hf.lineprint("Loading datasets and parameters...")
warnings.filterwarnings("ignore", category=UserWarning)
sns.set_style('darkgrid')
sns.set_context('talk')
sns.set(font_scale=0.7)
hf.myprint("Completed")
torch.manual_seed(12345678)

try:
    hf.myprint(f"Running on {torch.cuda.get_device_name(0)}")
except Exception:
    hf.myprint("No GPU available")

mnist_data = datasets.MNIST('./data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()]))

mnist_test = datasets.MNIST('./data', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()]))

def runAutoencoder(
    number_of_epochs, 
    number_hidden=64,
    mode=0,
    plot_graph=False,
    save_network=False,
    save_training_log=False,
    compare=True,
    log=True,
    show=True):
    hf.lineprint("Processing autoencoder...")
    trained_dict = ac.trainForAsigment(number_of_epochs, number_hidden, mode, mnist_data=mnist_data, mnist_test=mnist_test)
    trained_encoder = trained_dict["autoencoder"]
    basename = f"epch{number_of_epochs}_h{number_hidden}_mode{mode}"
    if compare:
        ac.compareEncoder(trained_encoder, mnist_test, (0, 50), save=True, basename=basename)
    if save_network:
        torch.save(trained_encoder, basename+".savednn")
    if save_training_log:
        with open(basename+"_traintest.txt",'w') as f:
            writedict = trained_dict
            del writedict["autoencoder"]
            f.write(json.dumps(writedict))
    if plot_graph:
        _, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlim(-0.2, number_of_epochs+0.5)
        ax.set_ylabel('Error')
        ax.set_yscale("log")
        ax.set_xlabel('Epoch')
        x_values = [x for x in range(number_of_epochs + 1)]
        ax.plot(np.array(x_values), np.array(trained_dict["test_error_arr"]), label="Test Error")
        ax.plot(np.array(x_values), np.array(trained_dict["train_error_arr"]), label="Training Error")
        ax.legend(loc="upper right")
        filename = f'asignmentGraph_a_h{number_hidden}_epch{number_of_epochs}_m{mode}.png'
        plt.savefig(filename, dpi=200)
        hf.myprint("\tSaved "+filename)
        plt.clf()

        _, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlim(5, number_of_epochs)
        ax.set_ylabel('Error')
        ax.set_yscale("log")
        ax.set_xlabel('Epoch')
        x_values = [x for x in range(5, number_of_epochs + 1)]
        ax.plot(np.array(x_values), np.array(trained_dict["test_error_arr"][5:]), label="Test Error")
        ax.plot(np.array(x_values), np.array(trained_dict["train_error_arr"][5:]), label="Training Error")
        ax.legend(loc="upper right")
        filename =f'asignmentGraph_a_h{number_hidden}_epch{number_of_epochs}_m{mode}_zoomed.png'
        plt.savefig(filename, dpi=200)
        hf.myprint("\tSaved "+filename)
        plt.clf()

def plotComparison():
    #print(type(hf.loadTrainingLog(64, 0)))
    hiddenlis = [64, 128, 256, 512]
    for m in range(3):
        dic_lis = [hf.loadTrainingLog(h, m) for h in hiddenlis]
        _, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlim(-0.2, 40+0.5)
        ax.set_ylabel('Error')
        ax.set_yscale("log")
        ax.set_xlabel('Epoch')
        optimFunc = ["SGD", "Adam", "RMSprop"]
        plt.title(label=f"Testing Loss with {optimFunc[m]}")
        x_values = [x for x in range(40 + 1)]
        for i, dic in enumerate(dic_lis):
            ax.plot(np.array(x_values), np.array(dic["test_error_arr"]), label=f"{hiddenlis[i]} hidden neurons")
        ax.legend(loc="upper right")
        filename = f'graph_b_m{m}.png'
        plt.savefig(filename, dpi=200)
        hf.myprint("\tSaved "+filename)
        plt.clf()

    for h in hiddenlis:
        dic_lis = [hf.loadTrainingLog(h, m) for m in range(3)]
        _, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlim(-0.2, 40+0.5)
        ax.set_ylabel('Error')
        ax.set_yscale("log")
        ax.set_xlabel('Epoch')
        optimFunc = ["SGD", "Adam", "RMSprop"]
        plt.title(label=f"Testing Loss with {h} neurons on the hidden layer")
        x_values = [x for x in range(40 + 1)]
        for m, dic in enumerate(dic_lis):
            ax.plot(np.array(x_values), np.array(dic["test_error_arr"]), label=f"{optimFunc[m]}")
        ax.legend(loc="upper right")
        filename = f'graph_c_h{h}.png'
        plt.savefig(filename, dpi=200)
        hf.myprint("\tSaved "+filename)
        plt.clf()


def plotNNresults():
    rng = [4,43,45,47,91,37,30,84,79,80,12]
    hiddenlis = [64, 128, 256, 512]
    for hid in hiddenlis:
        for m in range(3):
            trained_network = torch.load(f"neural_networks/epch40_h{hid}_mode{m}.savednn")
            ac.compareEncoder(trained_network, mnist_test, rng, save=True, basename=f"result_h{hid}_m{m}")

def evaluateSavedNetwork(filename):
    train_loader = torch.utils.data.DataLoader(mnist_data, batch_size=glval.batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(mnist_test, batch_size=glval.batch_size)
    saved_encoder = torch.load(filename)
    test_results = ac.testEncoder(
        saved_encoder,
        test_loader,
        train_loader,
        nn.MSELoss(),
        ac.Flatten())
    hf.myprint(f"Saved Network Train Loss: {test_results[0]} \n\tSaved Network Test Loss: {test_results[1]}")
    ac.compareEncoder(saved_encoder, mnist_test, (0, 20), save=True)

plotComparison()