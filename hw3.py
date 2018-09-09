import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision


class DatasetTrain(Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        X = np.load('cifar10-data/train_images.npy')
        y = np.load('cifar10-data/train_labels.npy')

        self.len = X.shape[0]
        self.x_data = torch.from_numpy(X).float()
        self.y_data = torch.from_numpy(y).float()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

class DatasetNormTrain(Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        X = np.load('cifar10-data/train_images.npy')
        y = np.load('cifar10-data/train_labels.npy')

        self.len = X.shape[0]

        X = X.astype(np.float64)

        for i in range((X.shape[0])):
            X[i][:][:][:] = (X[i][:][:][:] -
                             np.mean(X[i][:][:][:]).astype(np.float64)) / np.std(X[i][:][:][:]).astype(np.float64)

        self.x_data = torch.from_numpy(X).float()
        self.y_data = torch.from_numpy(y).float()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


class DatasetTest(Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        X = np.load('cifar10-data/test_images.npy')
        y = np.load('cifar10-data/test_labels.npy')

        self.len = X.shape[0]
        self.x_data = torch.from_numpy(X).float()
        self.y_data = torch.from_numpy(y).float()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

class DatasetNormTest(Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        X = np.load('cifar10-data/test_images.npy')
        y = np.load('cifar10-data/test_labels.npy')

        self.len = X.shape[0]

        X = X.astype(np.float64)

        for i in range((X.shape[0])):
            X[i][:][:][:] = (X[i][:][:][:] -
                             np.mean(X[i][:][:][:]).astype(np.float64)) / np.std(X[i][:][:][:]).astype(np.float64)

        self.x_data = torch.from_numpy(X).float()
        self.y_data = torch.from_numpy(y).float()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


class FFNet(nn.Module):
    def __init__(self):
        super(FFNet, self).__init__()
        self.fc1 = nn.Linear(3072, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        out = F.sigmoid(self.fc1(x))
        out = F.sigmoid(self.fc2(out))

        return out

class CNet2k5(nn.Module):
    def __init__(self):
        super(CNet2k5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, 1, 0)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)
        self.fc1 = nn.Linear(16 * 10 * 10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.max_pool(out)
        out = F.relu(self.conv2(out))

        out = out.view(-1, 16 * 10 * 10)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))

        return out

class CNet2k3(nn.Module):
    def __init__(self):
        super(CNet2k3, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.max_pool(out)
        out = F.relu(self.conv2(out))

        out = out.view(-1, 16 * 13 * 13)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))

        return out
#
## 2 FC layers: dim = 1024, 10
# class CNet2k3(nn.Module):
#     def __init__(self):
#         super(CNet2k3, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
#         self.max_pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
#         self.fc1 = nn.Linear(16 * 13 * 13, 1024)
#         self.fc2 = nn.Linear(1024, 10)
#
#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         out = self.max_pool(out)
#         out = F.relu(self.conv2(out))
#
#         out = out.view(-1, 16 * 13 * 13)
#         out = F.relu(self.fc1(out))
#         out = F.sigmoid(self.fc2(out))
#
#         return out
#
# # 2 FC layers: dim = 2048, 10
# class CNet2k3(nn.Module):
#     def __init__(self):
#         super(CNet2k3, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
#         self.max_pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
#         self.fc1 = nn.Linear(16 * 13 * 13, 2048)
#         self.fc2 = nn.Linear(2048, 10)
#
#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         out = self.max_pool(out)
#         out = F.relu(self.conv2(out))
#
#         out = out.view(-1, 16 * 13 * 13)
#         out = F.relu(self.fc1(out))
#         out = F.sigmoid(self.fc2(out))
#
#         return out
#
## 2 FC layers: dim = 32, 10
# class CNet2k3(nn.Module):
#     def __init__(self):
#         super(CNet2k3, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
#         self.max_pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
#         self.fc1 = nn.Linear(16 * 13 * 13, 32)
#         self.fc2 = nn.Linear(32, 10)
#
#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         out = self.max_pool(out)
#         out = F.relu(self.conv2(out))
#
#         out = out.view(-1, 16 * 13 * 13)
#         out = F.relu(self.fc1(out))
#         out = F.sigmoid(self.fc2(out))
#
#         return out
#
## 1 FC layer: dim = 10
# class CNet2k3(nn.Module):
#     def __init__(self):
#         super(CNet2k3, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
#         self.max_pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
#         self.fc1 = nn.Linear(16 * 13 * 13, 10)
#
#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         out = self.max_pool(out)
#         out = F.relu(self.conv2(out))
#
#         out = out.view(-1, 16 * 13 * 13)
#         out = F.sigmoid(self.fc1(out))
#
#         return out
#
## 3 FC layers: dim = 256, 512, 10
# class CNet2k3(nn.Module):
#     def __init__(self):
#         super(CNet2k3, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
#         self.max_pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
#         self.fc1 = nn.Linear(16 * 13 * 13, 256)
#         self.fc2 = nn.Linear(256, 512)
#         self.fc3 = nn.Linear(512, 10)
#
#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         out = self.max_pool(out)
#         out = F.relu(self.conv2(out))
#
#         out = out.view(-1, 16 * 13 * 13)
#         out = F.relu(self.fc1(out))
#         out = F.relu(self.fc2(out))
#         out = F.sigmoid(self.fc3(out))
#
#         return out

## 3 FC layers: dim = 64, 128, 10
# class CNet2k3(nn.Module):
#     def __init__(self):
#         super(CNet2k3, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
#         self.max_pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
#         self.fc1 = nn.Linear(16 * 13 * 13, 64)
#         self.fc2 = nn.Linear(64, 128)
#         self.fc3 = nn.Linear(128, 10)
#
#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         out = self.max_pool(out)
#         out = F.relu(self.conv2(out))
#
#         out = out.view(-1, 16 * 13 * 13)
#         out = F.relu(self.fc1(out))
#         out = F.relu(self.fc2(out))
#         out = F.sigmoid(self.fc3(out))
#
#         return out
#
## 4 FC layers: dim = 64, 128, 256, 10
# class CNet2k3(nn.Module):
#     def __init__(self):
#         super(CNet2k3, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
#         self.max_pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
#         self.fc1 = nn.Linear(16 * 13 * 13, 64)
#         self.fc2 = nn.Linear(64, 128)
#         self.fc3 = nn.Linear(128, 256)
#         self.fc4 = nn.Linear(256, 10)
#
#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         out = self.max_pool(out)
#         out = F.relu(self.conv2(out))
#
#         out = out.view(-1, 16 * 13 * 13)
#         out = F.relu(self.fc1(out))
#         out = F.relu(self.fc2(out))
#         out = F.relu(self.fc3(out))
#         out = F.sigmoid(self.fc4(out))
#
#         return out
#
## 4 FC layers: dim = 1024, 512, 256, 10
# class CNet2k3(nn.Module):
#     def __init__(self):
#         super(CNet2k3, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
#         self.max_pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
#         self.fc1 = nn.Linear(16 * 13 * 13, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512, 256)
#         self.fc4 = nn.Linear(256, 10)
#
#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         out = self.max_pool(out)
#         out = F.relu(self.conv2(out))
#
#         out = out.view(-1, 16 * 13 * 13)
#         out = F.relu(self.fc1(out))
#         out = F.relu(self.fc2(out))
#         out = F.relu(self.fc3(out))
#         out = F.sigmoid(self.fc4(out))
#
#         return out
#
## 4 FC layers: dim = 32, 1024, 2048, 10
# class CNet2k3(nn.Module):
#     def __init__(self):
#         super(CNet2k3, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
#         self.max_pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
#         self.fc1 = nn.Linear(16 * 13 * 13, 32)
#         self.fc2 = nn.Linear(32, 1024)
#         self.fc3 = nn.Linear(1024, 2048)
#         self.fc4 = nn.Linear(2048, 10)
#
#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         out = self.max_pool(out)
#         out = F.relu(self.conv2(out))
#
#         out = out.view(-1, 16 * 13 * 13)
#         out = F.relu(self.fc1(out))
#         out = F.relu(self.fc2(out))
#         out = F.relu(self.fc3(out))
#         out = F.sigmoid(self.fc4(out))
#
#         return out

# Default CNN, 7x7 kernel
class CNet2k7(nn.Module):
    def __init__(self):
        super(CNet2k7, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 7, 1, 0)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 7, 1, 0)
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.max_pool(out)
        out = F.relu(self.conv2(out))

        out = out.view(-1, 16 * 7 * 7)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))

        return out

# 4-convolutional-layer CNN
class CNet4(nn.Module):
    def __init__(self):
        super(CNet4, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, 1, 0)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)
        self.conv3 = nn.Conv2d(16, 16, 5, 1, 0)
        self.fc1 = nn.Linear(16 * 1 * 1, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.max_pool(out)
        out = F.relu(self.conv2(out))
        out = self.max_pool(out)
        out = F.relu(self.conv3(out))

        out = out.view(-1, 16 * 1 * 1)
        out = F.sigmoid(self.fc1(out))

        return out

# Function that takes a tensor as input and normalizes its value
def normalize(tensor):
    values = tensor.numpy()
    for i in range(len(tensor)):
        values[i][:][:][:] = (values[i][:][:][:] - np.mean(values[i][:][:][:])) / np.std(values[i][:][:][:])
    return torch.FloatTensor(values)

# Function that takes a string clf and runs a given classifier
def FFNN(batch=64, learning_rate=0.001, mom=0.90):
    # Specify the newtork architecture
    net = FFNet()

    # Specify the training dataset
    dataset = DatasetTrain()
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch,
                              shuffle=True)

    # Specify the testing dataset
    dataset_test = DatasetTest()
    test_loader = DataLoader(dataset=dataset_test,
                              batch_size=dataset_test.len)

    # Visualize the dataset
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.title('Visualize the dataset')
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        #plt.savefig('image batch')


    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))

    # Specify the loss function
    criterion = nn.CrossEntropyLoss()

    # Specify the optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=mom)

    max_epochs = 100

    loss_np = np.zeros((max_epochs))
    accuracy = np.zeros((max_epochs))

    for epoch in range(max_epochs):
        epoch_accuracy = []
        for i, data in enumerate(train_loader, 0):

            # Get inputs and labels from data loader
            inputs, labels = data

            # Feed the input data into the network
            inputs, labels = Variable(inputs.view(inputs.size(0), -1)), Variable(labels.long())

            # zero gradient
            optimizer.zero_grad()

            y_pred = net(inputs)

            # Calculate the loss using predicted labels and ground truth labels
            loss = criterion(y_pred, labels)

            # backpropogates to compute gradient
            loss.backward()

            # updates the weghts
            optimizer.step()

            # convert predicted labels into numpy
            y_pred_np = y_pred.data.numpy()

            # calculate the training accuracy of the current model
            pred_np = np.empty(batch)
            for k in range(len(y_pred_np)):
                pred_np[k] = np.argmax(y_pred_np[k])

            pred_np = pred_np.reshape(len(pred_np), 1)

            label_np = labels.data.numpy().reshape(len(labels), 1)

            correct = 0
            for j in range(y_pred_np.shape[0]):
                if pred_np[j] == label_np[j]:
                    correct += 1
            epoch_accuracy.append(float(correct) / float(len(label_np)))

        print("epoch: ", epoch, "loss: ", loss.data[0])
        loss_np[epoch] = loss.data.numpy()
        accuracy[epoch] = np.mean(np.asarray(epoch_accuracy))


    print("Final Training Accuracy: ", accuracy[max_epochs - 1])

    for test_inputs, test_labels in test_loader:
        test_inputs, test_labels = Variable(test_inputs.view(test_inputs.size(0), -1)), Variable(test_labels.long())
        y_pred_test = net(test_inputs)
        y_pred_test_np = y_pred_test.data.numpy()
        pred_test_np = np.empty(len(y_pred_test_np))
        for k in range(len(y_pred_test_np)):
            pred_test_np[k] = np.argmax(y_pred_test_np[k])

        pred_test_np = pred_test_np.reshape(len(pred_test_np), 1)

        label_test_np = test_labels.data.numpy().reshape(len(test_labels), 1)

        correct_test = 0
        for j in range(y_pred_test_np.shape[0]):
            if pred_test_np[j] == label_test_np[j]:
                correct_test += 1
        print("Test Accuracy: ", (float(correct_test) / float(len(label_test_np))))

    epoch_number = np.arange(0, max_epochs, 1)

    # # Plot the loss over epoch
    # plt.figure()
    # plt.plot(epoch_number, loss_np)
    # plt.title('Loss Over Epochs')
    # plt.xlabel('Number of Epoch')
    # plt.ylabel('Loss')
    # plt.savefig('Loss Over Epochs Feed Forward NN')
    #
    # # Plot the training accuracy over epoch
    # plt.figure()
    # plt.plot(epoch_number, accuracy)
    # plt.title('Training Accuracy Over Epochs')
    # plt.xlabel('Number of Epoch')
    # plt.ylabel('Accuracy')
    # plt.savefig('Training Accuracy Over Epochs Feed Forward NN')

def CNN(type='CNN2k5', norm=False, learning_rate=0.001, batch=64, mom=0.90):
    # Specify the newtork architecture
    if type == 'CNN2k5':
        net = CNet2k5()
    elif type == 'CNN2k3':
        net = CNet2k3()
    elif type == 'CNN2k7':
        net = CNet2k7()
    elif type == 'CNN4':
        net = CNet4()

    if norm == False:
        # Specify the training dataset
        dataset = DatasetTrain()
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=batch,
                                  shuffle=True)
        # Specify the testing dataset
        dataset_test = DatasetTest()
        test_loader = DataLoader(dataset=dataset_test,
                                 batch_size=dataset_test.len)
    else:
        # Specify the training dataset
        dataset = DatasetNormTrain()
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=batch,
                                  shuffle=True)
        # Specify the testing dataset
        dataset_test = DatasetNormTest()
        test_loader = DataLoader(dataset=dataset_test,
                                 batch_size=dataset_test.len)

    # Visualize the dataset
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.title('Visualize the dataset')
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        # plt.savefig('image batch')

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))

    # Specify the loss function
    criterion = nn.CrossEntropyLoss()

    # Specify the optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=mom)

    max_epochs = 100

    loss_np = np.zeros((max_epochs))
    accuracy = np.zeros((max_epochs))

    for epoch in range(max_epochs):
        epoch_accuracy = []
        for i, data in enumerate(train_loader, 0):

            # Get inputs and labels from data loader
            inputs, labels = data

            # Feed the input data into the network
            inputs, labels = Variable(inputs), Variable(labels.long())

            # zero gradient
            optimizer.zero_grad()

            y_pred = net(inputs)

            # Calculate the loss using predicted labels and ground truth labels
            loss = criterion(y_pred, labels)

            # backpropogates to compute gradient
            loss.backward()

            # updates the weghts
            optimizer.step()

            # convert predicted labels into numpy
            y_pred_np = y_pred.data.numpy()

            # calculate the training accuracy of the current model
            pred_np = np.empty(batch)
            for k in range(len(y_pred_np)):
                pred_np[k] = np.argmax(y_pred_np[k])

            pred_np = pred_np.reshape(len(pred_np), 1)

            label_np = labels.data.numpy().reshape(len(labels), 1)

            correct = 0
            for j in range(y_pred_np.shape[0]):
                if pred_np[j] == label_np[j]:
                    correct += 1
            epoch_accuracy.append(float(correct) / float(len(label_np)))

        print("epoch: ", epoch, "loss: ", loss.data[0])
        loss_np[epoch] = loss.data.numpy()
        accuracy[epoch] = np.mean(np.asarray(epoch_accuracy))

    print("Final Training Accuracy: ", accuracy[max_epochs - 1])

    for test_inputs, test_labels in test_loader:
        test_inputs, test_labels = Variable(test_inputs), Variable(test_labels.long())
        y_pred_test = net(test_inputs)
        y_pred_test_np = y_pred_test.data.numpy()
        pred_test_np = np.empty(len(y_pred_test_np))
        for k in range(len(y_pred_test_np)):
            pred_test_np[k] = np.argmax(y_pred_test_np[k])

        pred_test_np = pred_test_np.reshape(len(pred_test_np), 1)

        label_test_np = test_labels.data.numpy().reshape(len(test_labels), 1)

        correct_test = 0
        for j in range(y_pred_test_np.shape[0]):
            if pred_test_np[j] == label_test_np[j]:
                correct_test += 1
        print("Test Accuracy: ", (float(correct_test) / float(len(label_test_np))))

    epoch_number = np.arange(0, max_epochs, 1)

    # # Plot the loss over epoch
    # plt.figure()
    # plt.plot(epoch_number, loss_np)
    # plt.title('Loss Over Epochs')
    # plt.xlabel('Number of Epoch')
    # plt.ylabel('Loss')
    # plt.savefig('Loss Over Epochs CNN')
    #
    # # Plot the training accuracy over epoch
    # plt.figure()
    # plt.plot(epoch_number, accuracy)
    # plt.title('Training Accuracy Over Epochs')
    # plt.xlabel('Number of Epoch')
    # plt.ylabel('Accuracy')
    # plt.savefig('Training Accuracy Over Epochs CNN')


#FFNN()

#CNN()

CNN(type='CNN2k5', norm=True, batch=128, learning_rate=0.003)

#CNN(type='CNN4', norm=True, batch=128, learning_rate=0.003)

# # Hyperparameter search
# learning_rates = [0.005, 0.003, 0.001]
# batch_sizes = [64, 128, 256]
# kernel_sizes = [3, 5, 7]
# H_dimensions = [32, 64, 96, 128, 160, 192, 224, 256, 512, 1024, 2048]
# num_fc = [1, 2, 3]


