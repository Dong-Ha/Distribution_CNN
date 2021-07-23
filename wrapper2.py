import torch
import torch.nn
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

from model import CNN

mnist_train = dset.MNIST(root='./CNN', train=True, transform=transforms.ToTensor(),download=True)
mnist_test = dset.MNIST(root='./CNN', train=False, transform=transforms.ToTensor(),download=True)


def train_model(model, batch_size=100, epochs=3,learning_rate =0.001,device='cpu', save_file=None):
    
    data_loader = DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = torch.nn.CrossEntropyLoss().to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)

    for epoch in range(epochs):
        avg_cost = 0

        for X, Y in data_loader:
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis = model(X)
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()

            avg_cost += cost / len(data_loader)

        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))    

    if save_file is None:
        save_file = "MyCNN.pt"

    torch.save(model.state_dict(), save_file)


def predict(model,device):
    model.load_state_dict(torch.load("MyCNN.pt"))
    model = model.to(device)    

    test_loader = DataLoader(dataset=mnist_test, batch_size=len(mnist_test), shuffle=True)

    with torch.no_grad():
        for X_test ,Y_test in test_loader: 
            X_test = X_test.to(device)
            Y_test = Y_test.to(device)

            prediction = model(X_test)
            correct_prediction = torch.argmax(prediction, 1) == Y_test
            accuracy = correct_prediction.float().mean()
            print('Accuracy:', accuracy.item())