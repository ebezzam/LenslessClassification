"""
Modified from this example:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

TODO : specify dataset creation, and create if not available!!
TODO : save models


Original MNIST
```
python scripts/train_logistic_reg.py
```

Lens
```
python scripts/train_logistic_reg.py --dataset data/MNIST_lens_down128
```

Tape
```
python scripts/train_logistic_reg.py --dataset data/MNIST_tape_down128
```

SLM
```
python scripts/train_logistic_reg.py --dataset data/MNIST_adafruit_down128
```


"""

from lenslessclass.models import MultiClassLogistic
import torch
import torch.nn as nn
from lenslessclass.datasets import MNISTAugmented
import torch.optim as optim
import torchvision.transforms as transforms
import time
import click
import os
import torchvision.datasets as dset


@click.command()
@click.option(
    "--dataset",
    type=str,
    help="Path to dataset.",
)
@click.option("--lr", type=float, help="Learning rate.", default=0.01)
@click.option("--momentum", type=float, help="Momentum (for learning).", default=0.01)
@click.option("--n_epoch", type=int, help="Number of epochs to train.", default=10)
@click.option("--batch_size", type=int, help="Batch size.", default=100)
def train_logistic_reg(dataset, lr, momentum, n_epoch, batch_size):

    ## load mnist dataset
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("CUDA available, using GPU.")
    else:
        print("CUDA not available, using CPU.")

    if dataset is None:
        # use original dataset
        print("\nNo dataset provided, using original MNIST dataset!\n")

        root = "./data"
        if not os.path.exists(root):
            os.mkdir(root)
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]
        )  # need to cast to float tensor for training
        train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
        test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)
        output_dim = (28, 28)
    else:
        # use prepared dataset
        # -- first determine mean and standard deviation (of training set)
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)])
        train_set = MNISTAugmented(path=dataset, train=True, transform=trans)
        mean, std = train_set.get_stats()
        print("Dataset mean : ", mean)
        print("Dataset standard deviation : ", std)

        # -- normalize according to training set stats
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_set = MNISTAugmented(path=dataset, train=True, transform=trans)
        test_set = MNISTAugmented(path=dataset, train=False, transform=trans)
        output_dim = train_set.output_dim

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False
    )

    print("==>>> total training batch number: {}".format(len(train_loader)))
    print("==>>> total testing batch number: {}".format(len(test_loader)))

    ## training
    model = MultiClassLogistic(input_shape=output_dim)
    if use_cuda:
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    criterion = nn.CrossEntropyLoss()

    print("Start training...")
    start_time = time.time()
    for epoch in range(n_epoch):
        # training
        running_loss = 0.0
        for i, (x, target) in enumerate(train_loader):
            # get inputs
            if use_cuda:
                x, target = x.cuda(), target.cuda()

            # zero parameters gradients
            optimizer.zero_grad()

            # x, target = Variable(x), Variable(target)

            # forward, backward, optimize
            out = model(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % batch_size == (batch_size - 1):  # print every X mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / batch_size:.3f}")
                running_loss = 0.0

        # testing
        correct_cnt, running_loss = 0, 0
        total_cnt = 0
        for i, (x, target) in enumerate(test_loader):

            # get inputs
            if use_cuda:
                x, target = x.cuda(), target.cuda()
            # x, target = Variable(x, volatile=True), Variable(target, volatile=True)

            # forward, and compute loss
            out = model(x)
            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()
            running_loss += loss.item() / batch_size
        print(
            "==>>> epoch: {}, test loss: {:.6f}, acc: {:.3f}".format(
                epoch + 1, running_loss, correct_cnt * 1.0 / total_cnt
            )
        )

    proc_time = time.time() - start_time
    print(f"Processing time [m] : {proc_time / 60}")
    print("Finished Training")

    # save model
    PATH = f"./{os.path.basename(dataset)}_logistic_reg.pth"
    torch.save(model.state_dict(), PATH)


if __name__ == "__main__":
    train_logistic_reg()
