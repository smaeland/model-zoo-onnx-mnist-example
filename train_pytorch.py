
import gzip
import numpy as np
import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt




def load_mnist(infile_data, infile_labels):

    with gzip.open(infile_labels, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(infile_data, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(len(labels), 784)

    return images, labels


def preprocess(data):
    
    out = data / 255
    out = np.reshape(out, newshape=(data.shape[0], 1, 28, 28))

    return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch):

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))



if __name__ == '__main__':

    use_cuda = False
    plot = False

    device = torch.device("cuda" if use_cuda else "cpu")

    x_train, y_train = load_mnist('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
    x_train = preprocess(x_train)
    print(x_train.shape)

    if plot:
        ncols, nrows = 5, 5
        fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10))
        for i in range(nrows):
            for j in range(ncols):
                img_num = i*10 + j
                axs[i, j].imshow(x_train[img_num, :, :, 0], cmap='gray')
                axs[i, j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                axs[i, j].set_xlabel(f'{class_names[y_train[img_num]]}')
        plt.show()


    print('y_train.shape:', y_train.shape)

    train_kwargs = {'batch_size': 64, 'shuffle': True}
    if use_cuda:
        train_kwargs.update(
            {
                'num_workers': 1,
                'pin_memory': True,
            }
        )

    x_tensor = torch.Tensor(x_train)
    y_tensor = torch.Tensor(y_train).type(torch.LongTensor)
    train_ds = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(train_ds, **train_kwargs)

    model = Net().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1)

    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, epoch)
    
    # Save model in pytorch format
    torch.save(model.state_dict(), "pytorch-cnn-model.pt")

    # Save the model in ONNX format
    input_spec = torch.randn(train_kwargs['batch_size'], 1, 28, 28)
    torch.onnx.export(
        model,
        input_spec,
        'onnx-cnn-model.onnx',
        verbose=True,
        input_names = ['input'],
        output_names = ['output'],
        dynamic_axes={
            'input' : {0 : 'batch_size'},
            'output' : {0 : 'batch_size'}
        }
    )
