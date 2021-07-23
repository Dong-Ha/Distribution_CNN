import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

mnist_train = dset.MNIST(root='/home/dhshin/Desktop/CNN', train=True, transform=transforms.ToTensor(),download=True)
mnist_test = dset.MNIST(root='/home/dhshin/Desktop/CNN', train=False, transform=transforms.ToTensor(),download=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)