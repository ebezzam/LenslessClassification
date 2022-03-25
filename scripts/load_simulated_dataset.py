"""
Load a prepared dataset.

Lens
```
python scripts/load_simulated_dataset.py --dataset data/MNIST_lens_down128
```

Tape
```
python scripts/load_simulated_dataset.py --dataset data/MNIST_tape_down128
```

SLM
```
python scripts/load_simulated_dataset.py --dataset data/MNIST_fixedslm_down128
```
"""


import numpy as np
import click
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from lensless.plot import plot_image
from lenslessclass.datasets import MNISTAugmented


@click.command()
@click.option(
    "--dataset",
    type=str,
    help="Path to dataset.",
)
@click.option(
    "--test",
    is_flag=True,
    help="Load test set, otherwise train.",
)
@click.option("--idx", type=int, help="Example to plot.", default=0)
@click.option(
    "--gamma",
    default=None,
    type=float,
    help="Gamma factor for plotting.",
)
@click.option(
    "--normalize_plot",
    is_flag=True,
    help="Whether to normalize simulation plot.",
)
def load_simulated_dataset(dataset, test, idx, gamma, normalize_plot):

    # -- load dataset
    # scale between [0, 1] to convert numpy array into torch tensors
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1)])
    ds = MNISTAugmented(path=dataset, train=not test, transform=trans)
    print("Number of examples :", len(ds))

    # -- get statistics
    mean, std = ds.get_stats()
    print("Dataset mean : ", mean)
    print("Dataset standard deviation : ", std)

    # -- check one file
    input_image, label = ds[idx]
    print(f"\nSample {idx}")
    print("label", label)
    print("device", input_image.device)
    print("shape", input_image.shape)
    print("dtype", input_image.dtype)
    print("minimum : ", input_image.min().item())
    print("maximum : ", input_image.max().item())

    # plot sample
    input_image_cpu = np.transpose(input_image.cpu(), (1, 2, 0))
    plot_image(input_image_cpu, gamma=gamma, normalize=normalize_plot)

    plt.show()


if __name__ == "__main__":
    load_simulated_dataset()
