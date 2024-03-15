from __future__ import annotations
import os
import typing as tp
import math
from itertools import islice
import collections
import pathlib

import torch
import numpy as np
from torch.utils.data import (
    DataLoader,
    Sampler,
    WeightedRandomSampler,
    RandomSampler,
    SequentialSampler,
)
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Path(pathlib.Path):
    """
    A pathlib.Path child class that allows concatenation with strings
    by overloading the addition operator.

    In addition, it implements the ``startswith`` and ``endswith`` methods
    just like in the base :obj:`str` type.

    The ``replace_`` implementation is meant to be an implementation closer
    to the :obj:`str` type.

    Iterating over a directory with ``iterdir`` that does not exists
    will return an empty iterator instead of throwing an error.

    Creating a directory with ``mkdir`` allows existing directory and
    creates parents by default.
    """

    _flavour = (
        pathlib._windows_flavour  # type: ignore[attr-defined]  # pylint: disable=W0212
        if os.name == "nt"
        else pathlib._posix_flavour  # type: ignore[attr-defined]  # pylint: disable=W0212
    )

    def __add__(self, string: str) -> Path:
        return Path(str(self) + string)

    def startswith(self, string: str) -> bool:
        return str(self).startswith(string)

    def endswith(self, string: str) -> bool:
        return str(self).endswith(string)

    def iterdir(self) -> tp.Generator:
        if self.exists():
            yield from [Path(x) for x in pathlib.Path(str(self)).iterdir()]
        yield from []

    def unlink(self, missing_ok: bool = True) -> Path:
        super().unlink(missing_ok=missing_ok)
        return self

    def mkdir(self, mode=0o777, parents: bool = True, exist_ok: bool = True) -> Path:
        super().mkdir(mode=mode, parents=parents, exist_ok=exist_ok)
        return self

    def glob(self, pattern: str) -> tp.Generator:
        # to support ** with symlinks: https://bugs.python.org/issue33428
        from glob import glob

        if "**" in pattern:
            sep = "/" if self.is_dir() else ""
            yield from map(
                Path,
                glob(self.as_posix() + sep + pattern, recursive=True),
            )
        else:
            yield from super().glob(pattern)


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
            indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def filter_kwargs_by_callable(
    kwargs: tp.Dict[str, tp.Any], callabl: tp.Callable, exclude: tp.List[str] = None
) -> tp.Dict[str, tp.Any]:
    """Filter a dictionary keeping only the keys which are part of a function signature."""
    from inspect import signature

    args = signature(callabl).parameters.keys()
    return {k: v for k, v in kwargs.items() if (k in args) and k not in (exclude or [])}


def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]


def collate_features(batch, with_coords: bool = False):
    img = torch.cat([item[0] for item in batch], dim=0)
    if not with_coords:
        return img
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]


def get_split_loader(split_dataset, training=False, testing=False, weighted=False):
    """
    return either the validation loader or training loader
    """
    kwargs = {"num_workers": 4} if device.type == "cuda" else {}
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(
                    split_dataset,
                    batch_size=1,
                    sampler=WeightedRandomSampler(weights, len(weights)),
                    collate_fn=collate_MIL,
                    **kwargs,
                )
            else:
                loader = DataLoader(
                    split_dataset,
                    batch_size=1,
                    sampler=RandomSampler(split_dataset),
                    collate_fn=collate_MIL,
                    **kwargs,
                )
        else:
            loader = DataLoader(
                split_dataset,
                batch_size=1,
                sampler=SequentialSampler(split_dataset),
                collate_fn=collate_MIL,
                **kwargs,
            )

    else:
        ids = np.random.choice(
            np.arange(len(split_dataset), int(len(split_dataset) * 0.1)), replace=False
        )
        loader = DataLoader(
            split_dataset,
            batch_size=1,
            sampler=SubsetSequentialSampler(ids),
            collate_fn=collate_MIL,
            **kwargs,
        )

    return loader


def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.reg,
        )
    elif args.opt == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.reg,
        )
    else:
        raise NotImplementedError
    return optimizer


def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print("Total number of parameters: %d" % num_params)
    print("Total number of trainable parameters: %d" % num_params_train)


def generate_split(
    cls_ids,
    val_num,
    test_num,
    samples,
    n_splits=5,
    seed=7,
    label_frac=1.0,
    custom_test_ids=None,
):
    indices = np.arange(samples).astype(int)

    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []

        if custom_test_ids is not None:  # pre-built test split, do not need to sample
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(
                cls_ids[c], indices
            )  # all indices of this class
            val_ids = np.random.choice(
                possible_indices, val_num[c], replace=False
            )  # validation ids

            remaining_ids = np.setdiff1d(
                possible_indices, val_ids
            )  # indices of this class left after validation
            all_val_ids.extend(val_ids)

            if custom_test_ids is None:  # sample test split
                test_ids = np.random.choice(remaining_ids, test_num[c], replace=False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)

            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)

            else:
                sample_num = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator, n, None), default)


def calculate_error(Y_hat, Y):
    error = 1.0 - Y_hat.float().eq(Y.float()).float().mean().item()

    return error


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [
        N / len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))
    ]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)


def is_url(url: str) -> bool:
    return url.startswith("http")


def download_file(
    url: str, dest: Path | str | None = None, overwrite: bool = False
) -> Path:
    import tempfile
    import requests

    if dest is None:
        dest = Path(tempfile.NamedTemporaryFile().name)

    if Path(dest).exists() and not overwrite:
        return Path(dest)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return Path(dest)
