import numpy as np
import os
import random
import pathlib as plib
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from lenslessclass.models import SingleHidden
from lenslessclass.datasets import CelebAAugmented, CELEBA_ATTR
import json
from pprint import pprint
import seaborn as sn
import pandas as pd
import torchvision.datasets as dset
from torch.utils.data import Subset
from lenslessclass.model_dict import model_dict
from lenslessclass.util import device_checks
import torch.optim as optim
from datetime import datetime
import time
import click
from sklearn.model_selection import train_test_split



"""
train for:
- 100'000 attacks
- 1, 10, 100 masks
- Gender and Smiling
"""


@click.command()
@click.option(
    "--attr",
    type=click.Choice(CELEBA_ATTR, case_sensitive=True),
    help="Attribute to predict.",
    default="Male"
)
@click.option(
    "--n_masks",
    type=int,
    help="Number of masks.",
    default=1,
)
@click.option(
    "--n_epoch",
    type=int,
    help="Number of masks.",
    default=50,
)
@click.option(
    "--n_plaintext",
    type=int,
    help="Number of plaintext attacks.",
    default=100000,
)
def train_plaintext(attr, n_masks, n_epoch, n_plaintext):

    celeba_root = "/scratch"
    MODEL_DIR = "models/celeba_decoders"
    models = model_dict["celeba_decoder"]
    DATA_DIR = "data"  # to overwrite parent directory in metadata of model

    # choose source of data
    # model_dir = MODEL_DIR / plib.Path(models["100000"]["1 mask"])
    # model_dir = MODEL_DIR / plib.Path(models["100000"]["10 masks"])
    # model_dir = MODEL_DIR / plib.Path(models["100000"]["100 masks"])

    if n_masks > 1:
        model_dir = MODEL_DIR / plib.Path(models[f"{n_plaintext}"][f"{n_masks} masks"])
    else:
        model_dir = MODEL_DIR / plib.Path(models[f"{n_plaintext}"]["1 mask"])

    output_dim = [24, 32]

    # training param
    lr = 0.001
    batch_size = 64
    sched = 10
    # n_epoch = 50
    hidden_dim = 800
    single_gpu = False
    device = "cuda:0"
    # attr = "Male"   # try gender and smiling
    n_class = 1
    dropout = None
    batch_norm = True

    print_epoch = batch_size


    # Opening JSON file
    f = open(str(model_dir / "metadata.json"))
    metadata = json.load(f)
    pprint(metadata)

    # load metadata
    offset = metadata["offset"] if "offset" in metadata.keys() else 0
    _path = DATA_DIR / plib.Path(os.path.basename(metadata["dataset"]["path"]))
    print(_path)

    seed = metadata["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device, use_cuda, multi_gpu, device_ids = device_checks(device=device, single_gpu=single_gpu)

    # load data
    label_idx = CELEBA_ATTR.index(attr)

    # -- train set
    train_indices = np.load(model_dir / "train_indices.npy")
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(metadata["dataset"]["mean"], metadata["dataset"]["std"])])
    all_data = CelebAAugmented(
        path=_path, 
        transform=trans,
        target_dim=metadata["target_dim"],
        offset=offset
    )
    train_set = Subset(all_data, train_indices - offset)
    print("Training data size : ", len(train_set))

    # -- test set (from classification training)
    if n_masks == 1:
        ds_path = "data/celeba_1_learned_mixed_mask_out768_offset0_nfiles100000"
    elif n_masks == 10:
        ds_path = "data/celeba_10_learned_mixed_mask_out768_offset0_nfiles100000"
    elif n_masks == 100:
        ds_path = "data/celeba_100_random_mixed_mask_nonlinTrue_out768_offset0_nfiles100000"
    else:
        raise ValueError("Need to create dataset...")
    all_data = CelebAAugmented(
        path=ds_path, 
        transform=trans,
        target_dim=metadata["target_dim"],
        offset=0
    )

    # ----- get test split
    n_files_class = 100000
    test_size = 0.15
    test_size = int(n_files_class * test_size)
    train_size = n_files_class - test_size
    ds = dset.CelebA(
        root=celeba_root,
        split="all",
        download=False,
        transform=trans,
    )
    label_idx = ds.attr_names.index(attr)
    labels = ds.attr[:, label_idx][:n_files_class]
    train_indices, test_indices, _, _ = train_test_split(
        range(n_files_class),
        labels,
        train_size=train_size,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )
    test_set = Subset(all_data, test_indices)
    print("Test data size : ", len(test_set))

    # -- test set (from plaintext attack)
    # test_indices = np.load(model_dir / "test_indices.npy")
    # test_set = Subset(all_data, test_indices - offset)
    # print("Testing data size : ", len(test_set))

    # create Data Loader
    train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=batch_size, shuffle=True
        )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False
    )

    # create model
    model = SingleHidden(
        input_shape=output_dim,
        hidden_dim=hidden_dim,
        n_class=n_class,
        dropout=dropout,
        bn=batch_norm,
    )
    model_name = model.name()
    if multi_gpu:
        model = nn.DataParallel(model, device_ids=device_ids)
    if use_cuda:
        model = model.to(device)

    # set, optimizer, with default params
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False,
    )
    if sched:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=sched, gamma=0.1, verbose=True
        )
    else:
        scheduler = None
        sched = None
    criterion = nn.BCELoss()

    # TODO : save model
    bn = f"{attr}"
    timestamp = datetime.now().strftime("%d%m%Y_%Hh%M")
    model_output_dir = f"./{bn}_{n_masks}_plaintext{n_plaintext}_{n_epoch}epoch_sched{sched}_batch{batch_size}_seed{seed}_{model_name}_{timestamp}"
    model_output_dir = plib.Path(model_output_dir)
    model_output_dir.mkdir(exist_ok=True)
    model_file = model_output_dir / "state_dict.pth"

    metadata = {
        "dataset": str(_path),
        "seed": int(seed),
        "attr": attr,
        "timestamp (DDMMYYYY_HhM)": timestamp,
        "model": model_name,
        "device_ids": device_ids,
        "batch_size": int(batch_size),
        "hidden_dim": int(hidden_dim),
        "dropout": dropout,
        "single_gpu": single_gpu,
    }

    metadata_fp = model_output_dir / "metadata.json"
    with open(metadata_fp, "w") as fp:
        json.dump(metadata, fp)

    test_loss_fp = model_output_dir / "test_loss.npy"
    train_loss_fp = model_output_dir / "train_loss.npy"
    test_acc_fp = model_output_dir / "test_acc.npy"
    train_acc_fp = model_output_dir / "train_acc.npy"

    print(f"Model saved to : {str(model_output_dir)}")

    # train
    print("Start training...")
    start_time = time.time()
    test_loss = []
    train_loss = []
    test_accuracy = []
    train_accuracy = []
    best_test_acc = 0
    best_test_acc_epoch = 0
    for epoch in range(n_epoch):

        # training
        model.train()
        running_loss = 0.0
        correct_cnt, total_cnt = 0, 0
        for i, (x, target) in enumerate(train_loader):

            # get inputs
            if use_cuda:
                x, target = x.to(device), target.to(device)
            target = target[:, label_idx]
            target = target.unsqueeze(1)
            target = target.to(x)

            # zero parameters gradients
            optimizer.zero_grad()

            # forward, backward, optimize
            out = model(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            # train accuracy
            pred_label = out.round()
            correct_cnt += (pred_label == target).sum()
            total_cnt += x.data.size()[0]
            running_acc = (correct_cnt * 1.0 / total_cnt).item()

            # print statistics
            running_loss += loss.item()
            if (i + 1) % print_epoch == 0:  # print every `print_epoch` mini-batches
                proc_time = (time.time() - start_time) / 60.0
                print(
                    f"[{epoch + 1}, {i + 1:5d} / {len(train_loader)}, {proc_time:.2f} min] loss: {running_loss  / (i + 1):.3f}, acc: {running_acc:.3f}"
                )

        _loss = running_loss / len(train_loader)
        train_loss.append(_loss)
        train_acc = (correct_cnt * 1.0 / total_cnt).item()
        train_accuracy.append(train_acc)
        print(f"training loss : {_loss:.3f}")
        print(f"training accuracy : {train_acc:.3f}")

        # testing
        model.eval()
        correct_cnt, running_loss = 0, 0
        total_cnt = 0

        with torch.no_grad():
            for i, (x, target) in enumerate(test_loader):

                # get inputs
                if use_cuda:
                    x, target = x.to(device), target.to(device)
                target = target[:, label_idx]
                target = target.unsqueeze(1)
                target = target.to(x)

                # forward, and compute loss
                out = model(x)
                loss = criterion(out, target)

                # accumulate loss and accuracy
                pred_label = out.round()
                correct_cnt += (pred_label == target).sum()
                total_cnt += x.data.size()[0]

                running_loss += loss.item()

        _loss = running_loss / len(test_loader)
        _acc = (correct_cnt * 1.0 / total_cnt).item()
        print("==>>> epoch: {}, test loss: {:.6f}, acc: {:.3f}".format(epoch + 1, _loss, _acc))
        test_loss.append(_loss)
        test_accuracy.append(_acc)

        if _acc > best_test_acc:
            # save model param
            best_test_acc = _acc
            best_test_acc_epoch = epoch + 1
            torch.save(model.state_dict(), str(model_file))

        # save losses
        with open(test_loss_fp, "wb") as f:
            np.save(f, np.array(test_loss))
        with open(train_loss_fp, "wb") as f:
            np.save(f, np.array(train_loss))
        with open(test_acc_fp, "wb") as f:
            np.save(f, np.array(test_accuracy))
        with open(train_acc_fp, "wb") as f:
            np.save(f, np.array(train_accuracy))

        if scheduler is not None:
            scheduler.step()

    proc_time = time.time() - start_time
    print(f"Processing time [m] : {proc_time / 60}")
    print("Finished Training")

    ## save model metadata
    metadata.update(
        {
            "best_test_acc": best_test_acc,
            "best_test_acc_epoch": best_test_acc_epoch,
        }
    )
    metadata_fp = model_output_dir / "metadata.json"
    with open(metadata_fp, "w") as fp:
        json.dump(metadata, fp)

    print(f"Model saved to : {str(model_output_dir)}")


if __name__ == "__main__":
    train_plaintext()


