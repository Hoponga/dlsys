import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    fn_block = nn.Sequential(
        nn.Linear(dim, hidden_dim), 
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )
    return nn.Sequential(nn.Residual(fn_block), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    modules = [
        nn.Linear(dim, hidden_dim),
        nn.ReLU()
    ]
    for i in range(num_blocks):
        modules.append(ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob))
    modules.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*modules)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
    else:
        model.eval()
    total_loss = 0
    total_correct = 0
    total = 0
    total_step = 0
    loss_fn = nn.SoftmaxLoss()
    for i, (X, y) in enumerate(dataloader): 
        print(i)
        #print(X.shape, y.shape)
        if y: 
            X = X.reshape((y.shape[0], -1))

        
        if opt is not None:
            opt.reset_grad()

        
        out = model(X)
        loss = loss_fn(out, y)
        out_labels = np.argmax(out.numpy(), axis = 1)
        total_correct += np.sum(out_labels == y.numpy())
        total += y.shape[0]

        total_loss += loss 

        if opt is not None:
            loss.backward()
            opt.step()
        total_step += 1
    ### END YOUR SOLUTION

    return 1 - total_correct / total, total_loss.numpy() / total_step

def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train = ndl.data.MNISTDataset(data_dir + "/train-images-idx3-ubyte.gz", data_dir + "/train-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    test = ndl.data.MNISTDataset(data_dir + "/t10k-images-idx3-ubyte.gz", data_dir + "/t10k-labels-idx1-ubyte.gz")
    test_dataloader = ndl.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    model = MLPResNet(784, hidden_dim, 3, 10)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    stats = None 
    for i in range(epochs):
        print("hi", len(train))
        train_acc, train_loss = epoch(train_dataloader, model, opt)
        test_acc, test_loss = epoch(test_dataloader, model)
        print("Epoch: {}, Train Loss: {}, Test Loss: {}, Train Acc: {}, Test Acc: {}".format(i, train_loss, test_loss, train_acc, test_acc))
        stats = (train_acc, train_loss, test_acc, test_loss)
    return stats 

    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
