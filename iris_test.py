import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from IPython import display

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import seaborn as sns
import numpy as np
from customer_mm import theModelClass
display.set_matplotlib_formats("svg")
plt.style.use("ggplot")


def draw_plt(train_accuracy, test_accuracy, dropout_rate)->None:

    fig = plt.figure(figsize=(10, 5))
    plt.plot(train_accuracy, "b--", alpha=.8, color="#D7A9A8")
    plt.plot(test_accuracy, "r-", color="#7E102C")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend(["Train", "Test"])
    plt.title("Dropout rate = %g"%dropout_rate)
    plt.show()

def createANewModel(dropout_rate:float):

    ANNiris = theModelClass(dropout_rate)
    lossfun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ANNiris.parameters(), lr=0.05)

    return ANNiris, lossfun, optimizer

def trainTheModel(train_loader, test_loader, dropout_rate:float, numepochs:int):

    ANNiris, lossfun, optimizer = createANewModel(dropout_rate)
    trianAccuracy = []
    testAccuracy = []

    for epochi in range(numepochs):
        
        ANNiris.train()
        batchAccuarcy = []
        for X, y in train_loader:
            
            yHat = ANNiris(X)
            loss = lossfun(yHat, y)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batchAccuarcy.append(100*torch.mean((torch.argmax(yHat,axis=1)==y).float()).item())
        
        trianAccuracy.append(np.mean(batchAccuarcy))

        ANNiris.eval()
        X, y = next(iter(test_loader))
        predlables = torch.argmax(ANNiris(X), axis=1)
        testAccuracy.append(100*torch.mean((predlables ==y).float()).item())

    draw_plt(trianAccuracy, testAccuracy, dropout_rate)
    return trianAccuracy, testAccuracy


if __name__ == "__main__":
    iris = sns.load_dataset("iris")

    data = torch.tensor(iris[iris.columns[0:4]].values).float()

    labels = torch.zeros(len(data), dtype=torch.long)
    labels[iris.species == "versicolor"] = 1
    labels[iris.species == "virginica"] = 2

    train_data, test_data, \
        train_labels, test_lables = train_test_split(data, labels, 
                                                    test_size = .3)

    train_data = torch.utils.data.TensorDataset(train_data, train_labels)
    test_data = torch.utils.data.TensorDataset(test_data, test_lables)
    print("test size: ", test_data.tensors[0].shape[0] / data.shape[0])

    numepochs = 250
    batchsize = 16
    dropout_rate = .1
    train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])

    train_accuracy, test_accuracy = trainTheModel(train_loader, test_loader, dropout_rate, numepochs)
