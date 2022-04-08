"""

Train SLM as well as logistic regression.

Test locally with CPU
```
python scripts/train_slm_logistic_reg.py --cpu \
--dataset data/MNIST_no_psf_down6_1000files --batch_size 20
```

On server
```
python scripts/train_slm_logistic_reg.py  \
--dataset data/MNIST_no_psf_down2 --batch_size 50 \
--mean 0.0011 --std 0.0290
```

"""

from turtle import pu
from lenslessclass.models import SLMMultiClassLogistic
import torch
import os
import torch.nn as nn
from lenslessclass.datasets import MNISTAugmented
import torch.optim as optim
import torchvision.transforms as transforms
import time
import click
from waveprop.devices import SLMOptions, SensorOptions, slm_dict, sensor_dict, SensorParam


@click.command()
@click.option(
    "--dataset",
    type=str,
    help="Path to dataset.",
)
@click.option("--slm", type=str, help="Which SLM to use.", default=SLMOptions.ADAFRUIT.value)
@click.option("--sensor", type=str, help="Which sensor to use.", default=SensorOptions.RPI_HQ.value)
@click.option(
    "--crop_fact",
    type=float,
    default=0.7,
    help="Fraction of sensor that is left uncropped, centered.",
)
@click.option("--simple", is_flag=True, help="Don't take into account deadspace.")
@click.option("--scene2mask", type=float, default=0.4, help="Scene to SLM/mask distance in meters.")
@click.option(
    "--mask2sensor", type=float, default=0.004, help="SLM/mask to sensor distance in meters."
)
@click.option("--cpu", is_flag=True, help="Use CPU even if GPU if available.")
@click.option("--lr", type=float, help="Learning rate.", default=0.01)
@click.option("--momentum", type=float, help="Momentum (for learning).", default=0.01)
@click.option("--n_epoch", type=int, help="Number of epochs to train.", default=10)
@click.option("--batch_size", type=int, help="Batch size.", default=30)
@click.option(
    "--print_epoch",
    type=int,
    help="How many batches to wait before printing epoch progress.",
    default=None,
)
@click.option(
    "--mean",
    type=float,
    help="Mean of original dataset to normalize, if not provided it will be computed.",
)
@click.option(
    "--sensor_act",
    type=str,
    help="Activation at sensor. If not provided, none will applied.",
    default=None,
)
@click.option(
    "--std",
    type=float,
    help="Standard deviation of original dataset to normalize, if not provided it will be computed.",
)
@click.option("--down_out", type=float, help="Factor by which to downsample output.", default=128)
def train_slm_logistic_reg(
    dataset,
    slm,
    sensor,
    crop_fact,
    simple,
    scene2mask,
    mask2sensor,
    cpu,
    lr,
    momentum,
    n_epoch,
    batch_size,
    mean,
    std,
    down_out,
    print_epoch,
    sensor_act,
):

    if print_epoch is None:
        print_epoch = batch_size

    if sensor_act is not None:
        # https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions
        # ReLU doesn't make sense as activation as image intensity is always positive!
        # if sensor_act == "relu":
        #     sensor_act = nn.ReLU()
        # elif sensor_act == "leaky":
        #     sensor_act = nn.LeakyReLU(float=0.1)
        if sensor_act == "tanh":
            sensor_act = nn.Tanh()
        else:
            raise ValueError("Not supported activation.")

    sensor_param = sensor_dict[sensor]
    if down_out:
        output_dim = tuple((sensor_param[SensorParam.SHAPE] * 1 / down_out).astype(int))

    ## load mnist dataset
    use_cuda = torch.cuda.is_available()
    if cpu:
        device = "cpu"
        use_cuda = False
    else:
        if use_cuda:
            print("CUDA available, using GPU.")
            device = "cuda"
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                multi_gpu = True
                print(f"-- using {n_gpus} GPUs")
            else:
                multi_gpu = False
            # if n_gpus > 1:
            #     device_model = "cuda:1"
            # else:
            #     device_model = device

        else:
            device = "cpu"
            print("CUDA not available, using CPU.")

    # -- first determine mean and standard deviation (of training set)
    if mean is None and std is None:
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)])
        print("\nComputing stats...")
        train_set = MNISTAugmented(path=dataset, train=True, transform=trans)
        mean, std = train_set.get_stats(batch_size=batch_size)
        print("Dataset mean : ", mean)
        print("Dataset standard deviation : ", std)

        del train_set

    # -- normalize according to training set stats
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = MNISTAugmented(path=dataset, train=True, transform=trans)
    test_set = MNISTAugmented(path=dataset, train=False, transform=trans)
    input_shape = train_set.output_dim

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False
    )

    print("==>>> total training batch number: {}".format(len(train_loader)))
    print("==>>> total testing batch number: {}".format(len(test_loader)))

    ## hybrid neural network
    model = SLMMultiClassLogistic(
        input_shape=input_shape,
        slm_config=slm_dict[slm],
        sensor_config=sensor_param,
        crop_fact=crop_fact,
        device=device,
        deadspace=not simple,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
        device_mask_creation="cpu",  # TODO: bc doesn't fit on GPU
        output_dim=output_dim,
        sensor_activation=sensor_act,
        multi_gpu=multi_gpu,
    )

    # # TODO : doesn't work since PSF generation happens on CPU?
    # if multi_gpu:
    #     model = nn.parallel.DistributedDataParallel(model)

    if use_cuda:
        model = model.to(device)

    # TODO : try ADAM
    # set different learning rates: https://pytorch.org/docs/stable/optim.html
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    criterion = nn.CrossEntropyLoss()

    # Print model and optimizer state_dict
    print("\nModel's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print()
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    print("\nStart training...")
    start_time = time.time()
    for epoch in range(n_epoch):
        # training
        running_loss = 0.0
        for i, (x, target) in enumerate(train_loader):

            # print(i)
            # mask_old = model.slm_vals.clone()
            # weights_old = model.multiclass_logistic_reg[0].weight.clone()

            # get inputs
            if use_cuda:
                x, target = x.to(device), target.to(device)

            # zero parameters gradients
            optimizer.zero_grad()

            # forward, backward, optimize
            out = model(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            # SLM values have updated after backward
            # TODO : move into forward?
            model.compute_intensity_psf()

            # mask_new = model.slm_vals.clone()
            # weights_new = model.multiclass_logistic_reg[0].weight.clone()
            # import pudb; pudb.set_trace()
            # print("mask change", (mask_new - mask_old).sum())
            # (weights_new - weights_old).sum()
            # model.state_dict()

            # print statistics
            running_loss += loss.item()
            if (i + 1) % print_epoch == 0:  # print every `print_epoch` mini-batches
                proc_time = (time.time() - start_time) / 60.0
                print(
                    f"[{epoch + 1}, {i + 1:5d}, {proc_time:.2f} min] loss: {running_loss / print_epoch:.3f}"
                )
                running_loss = 0.0

            # if i % batch_size == (batch_size - 1):  # print every X mini-batches
            #     print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / batch_size:.3f}")
            #     running_loss = 0.0

        # testing
        correct_cnt, running_loss = 0, 0
        total_cnt = 0
        for i, (x, target) in enumerate(test_loader):

            # get inputs
            if use_cuda:
                x, target = x.cuda(), target.cuda()

            # forward, and compute loss
            out = model(x)
            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()
            running_loss += loss.item() / batch_size
        proc_time = (time.time() - start_time) / 60.0
        print(
            "==>>> epoch: {}, , {:.2f} min, test loss: {:.6f}, acc: {:.3f}".format(
                epoch + 1, proc_time, running_loss, correct_cnt * 1.0 / total_cnt
            )
        )

    proc_time = time.time() - start_time
    print(f"Processing time [m] : {proc_time / 60}")
    print("Finished Training")

    # save model
    PATH = f"./{os.path.basename(dataset)}_logistic_reg.pth"
    torch.save(model.state_dict(), PATH)


if __name__ == "__main__":
    train_slm_logistic_reg()
