import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


class Net(nn.Module):
    def __init__(self, in_size=28*28, hid_size=256):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_size, hid_size, bias=False)
        self.fc2 = nn.Linear(hid_size, hid_size, bias=False)
        self.fc3 = nn.Linear(hid_size, 10, bias=False)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return nn.functional.log_softmax(x, dim=1)


def vis_fc(act_list, layer_idx):
    fc_act = act_list[layer_idx][0].data.cpu().numpy()
    assert (len(fc_act) % 20 == 0)
    plt.imshow(fc_act.reshape(10, len(fc_act) // 10), cmap='gray')
    plt.suptitle('fc layer result, shape (1,500) ', fontsize=20)
    plt.savefig("fc.png")
    plt.show()


class MLP:
    def __init__(self, model_loc="./model/mlp_model_1layer_256", in_size=28*28, hid_size=256):
        self.USE_GPU = True
        self.device = torch.device("cuda" if self.USE_GPU else "cpu")

        # train/test loader
        self.trainset = torchvision.datasets.MNIST('./data', train=True, transform=torchvision.transforms.ToTensor())
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=16, shuffle=True, num_workers=2)
        self.testset = torchvision.datasets.MNIST('./data', train=False, transform=torchvision.transforms.ToTensor())
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=1000, shuffle=False, num_workers=6)

        # initial model
        self.init_model = Net(in_size, hid_size)
        self.init_model.load_state_dict(torch.load(model_loc))

    def spead_test(self, model, data, target):
        # make a test loader
        data = torch.Tensor(data).to(self.device)
        target = torch.LongTensor(target).to(self.device)

        # perform testing
        model.to(self.device).eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

    def train(self, train_data, train_label, test_data, test_label, in_size=28*28, hid_size=256):
        # my testing function
        def my_test(model, data, target):
            model.to(self.device).eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                output = model(data)
                test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(self.test_loader.dataset)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(self.test_loader.dataset),
                100. * correct / len(self.test_loader.dataset)))

        # make a train loader
        batch_size = 10
        my_train_loader = []
        my_train_label = []
        for i in range(train_data.shape[0]//batch_size):
            my_train_loader.append(torch.Tensor(train_data[i*batch_size:(i+1)*batch_size, :]))
            my_train_label.append(torch.LongTensor(train_label[i*batch_size:(i+1)*batch_size]).to(self.device))

        # make a test loader
        my_test_loader = torch.Tensor(test_data).to(self.device)
        my_test_label = torch.LongTensor(test_label).to(self.device)

        # make a local copy of model
        model = Net(in_size, hid_size).to(self.device)
        learning_rate = 1e-2
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # start training
        model.train()
        for epoch in range(1, 151):
            # adjusting learning rate
            if epoch % 50 == 0:
                learning_rate = learning_rate/10
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

            # train
            for i, data in enumerate(my_train_loader):
                data = data.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = torch.nn.functional.nll_loss(output, my_train_label[i])
                loss.backward()
                optimizer.step()

            # test in each epoch
            print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch, loss.item()))
            my_test(model, my_test_loader, my_test_label)

        return model

    def test(self, model):
        model.to(self.device).eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

    def get_data_matrix(self):
        mnist_train_data = []
        mnist_train_label = []
        mnist_test_data = []
        mnist_test_label = []

        for image, label in self.trainset:
            img = np.array(image.view(-1))
            mnist_train_data.append(img)
            mnist_train_label.append(label)

        for image, label in self.testset:
            img = np.array(image.view(-1))
            mnist_test_data.append(img)
            mnist_test_label.append(label)

        mnist_train_data = np.vstack(mnist_train_data)
        mnist_train_label = np.vstack(mnist_train_label).reshape(-1)
        mnist_test_data = np.vstack(mnist_test_data)
        mnist_test_label = np.vstack(mnist_test_label).reshape(-1)

        print("trainset matrix/label shape", mnist_train_data.shape, mnist_train_label.shape)
        print("testset matrix/label shape", mnist_test_data.shape, mnist_test_label.shape)

        return mnist_train_data, mnist_train_label, mnist_test_data, mnist_test_label


if __name__=="__main__":
    # an example to train a 256x256 model
    Mlp = MLP()
    mnist_train_data, mnist_train_label, mnist_test_data, mnist_test_label = Mlp.get_data_matrix()

    # change device
    Mlp.device = torch.device("cuda:1")

    # PCA
    pca = PCA(91)
    train_data_reduced = pca.fit_transform(mnist_train_data)
    test_data_reduced = pca.transform(mnist_test_data)
    data_singular = pca.singular_values_
    data_explained_ratio = sum(pca.explained_variance_ratio_)
    print("PCA has reserved {} principal components".format(91))

    # training model
    print("training on model input {} hiden {}".format(91, 32))
    new_model = Mlp.train(mnist_train_data, mnist_train_label, mnist_test_data, mnist_test_label, 28*28, 16)
    # new_model = Mlp.train(train_data_reduced, mnist_train_label, test_data_reduced, mnist_test_label, 91, 256)

    # saving new model
    torch.save(new_model.state_dict(), "model/model_original_16")
