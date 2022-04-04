"""

Train SLM as well as logistic regression.

Test locally with CPU
```
python scripts/train_slm_logistic_reg.py --cpu \
--dataset data/MNIST_no_psf_down6_1000files --batch_size 20
```

"""

from lenslessclass.models import SLMMultiClassLogistic
import torch
import torch.nn as nn
from lenslessclass.datasets import MNISTAugmented
import torch.optim as optim
import torchvision.transforms as transforms
import time
import click
from waveprop.devices import SLMOptions, SensorOptions, slm_dict, sensor_dict


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
):

    ## load mnist dataset
    use_cuda = torch.cuda.is_available()
    if cpu:
        device = "cpu"
        use_cuda = False
    else:
        if use_cuda:
            device = "cuda"
            print("CUDA available, using GPU.")
        else:
            device = "cpu"
            print("CUDA not available, using CPU.")

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

    # # TODO : remove! just testing downsampling factor to still have GPU memory
    # from waveprop.devices import SensorParam
    # down = 6
    # sensor_config = sensor_dict[sensor]
    # output_dim = sensor_config[SensorParam.SHAPE] // down

    ## hybrid neural network
    model = SLMMultiClassLogistic(
        input_shape=output_dim,
        slm_config=slm_dict[slm],
        sensor_config=sensor_dict[sensor],
        crop_fact=crop_fact,
        device=device,
        deadspace=not simple,
        scene2mask=scene2mask,
        mask2sensor=mask2sensor,
    )
    if use_cuda:
        model = model.cuda()

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

            print(i)

            # mask_old = model.slm_vals.clone()
            # weights_old = model.multiclass_logistic_reg[0].weight.clone()

            # get inputs
            if use_cuda:
                x, target = x.cuda(), target.cuda()

            # zero parameters gradients
            optimizer.zero_grad()

            # forward, backward, optimize
            out = model(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            # SLM values have updates after backward
            model.compute_intensity_psf()

            # mask_new = model.slm_vals.clone()
            # weights_new = model.multiclass_logistic_reg[0].weight.clone()

            # import pudb; pudb.set_trace()
            # print("mask change", (mask_new - mask_old).sum())
            # (weights_new - weights_old).sum()
            # model.state_dict()

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


if __name__ == "__main__":
    train_slm_logistic_reg()
