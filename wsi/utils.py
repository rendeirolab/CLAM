import typing as tp
from pathlib import Path
import tempfile

import requests
import h5py
import numpy as np
import openslide
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class WholeSlideBag(Dataset):
    def __init__(
        self,
        file_path,
        wsi=None,
        pretrained=False,
        custom_transforms=None,
        custom_downsample=1,
        target_patch_size=-1,
        target=None,
    ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
            target_patch_size (int): Custom defined image size before embedding
        """
        self.target = target

        self.pretrained = pretrained
        if wsi is None:
            wsi = openslide.open_slide(path)
        self.wsi = wsi
        if not custom_transforms:
            self.roi_transforms = default_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f["coords"]
            self.patch_level = f["coords"].attrs["patch_level"]
            self.patch_size = f["coords"].attrs["patch_size"]
            self.length = len(dset)
            if target_patch_size > 0:
                self.target_patch_size = (target_patch_size,) * 2
            elif custom_downsample > 1:
                self.target_patch_size = (self.patch_size // custom_downsample,) * 2
            else:
                self.target_patch_size = None
        # self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file["coords"]
        for name, value in dset.attrs.items():
            print(name, value)

        # print("\nfeature extraction settings")
        # print("target patch size: ", self.target_patch_size)
        # print("pretrained: ", self.pretrained)
        # print("transformations: ", self.roi_transforms)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, "r") as hdf5_file:
            coord = hdf5_file["coords"][idx]
        img = self.wsi.read_region(
            coord, self.patch_level, (self.patch_size, self.patch_size)
        ).convert("RGB")

        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        if self.target is None:
            return img, coord
        return img, self.target


class ContourCheckingFn(object):
    # Defining __call__ method
    def __call__(self, pt):
        raise NotImplementedError


class isInContourV1(ContourCheckingFn):
    def __init__(self, contour):
        self.cont = contour

    def __call__(self, pt):
        return (
            1
            if cv2.pointPolygonTest(self.cont, tuple(np.array(pt).astype(float)), False)
            >= 0
            else 0
        )


class isInContourV2(ContourCheckingFn):
    def __init__(self, contour, patch_size):
        self.cont = contour
        self.patch_size = patch_size

    def __call__(self, pt):
        pt = np.array(
            (pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2)
        ).astype(float)
        return (
            1
            if cv2.pointPolygonTest(self.cont, tuple(np.array(pt).astype(float)), False)
            >= 0
            else 0
        )


# Easy version of 4pt contour checking function - 1 of 4 points need to be in the contour for test to pass
class isInContourV3_Easy(ContourCheckingFn):
    def __init__(self, contour, patch_size, center_shift=0.5):
        self.cont = contour
        self.patch_size = patch_size
        self.shift = int(patch_size // 2 * center_shift)

    def __call__(self, pt):
        center = (pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2)
        if self.shift > 0:
            all_points = [
                (center[0] - self.shift, center[1] - self.shift),
                (center[0] + self.shift, center[1] + self.shift),
                (center[0] + self.shift, center[1] - self.shift),
                (center[0] - self.shift, center[1] + self.shift),
            ]
        else:
            all_points = [center]

        for points in all_points:
            if (
                cv2.pointPolygonTest(
                    self.cont, tuple(np.array(points).astype(float)), False
                )
                >= 0
            ):
                return 1
        return 0


# Hard version of 4pt contour checking function - all 4 points need to be in the contour for test to pass
class isInContourV3_Hard(ContourCheckingFn):
    def __init__(self, contour, patch_size, center_shift=0.5):
        self.cont = contour
        self.patch_size = patch_size
        self.shift = int(patch_size // 2 * center_shift)

    def __call__(self, pt):
        center = (pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2)
        if self.shift > 0:
            all_points = [
                (center[0] - self.shift, center[1] - self.shift),
                (center[0] + self.shift, center[1] + self.shift),
                (center[0] + self.shift, center[1] - self.shift),
                (center[0] - self.shift, center[1] + self.shift),
            ]
        else:
            all_points = [center]

        for points in all_points:
            if (
                cv2.pointPolygonTest(
                    self.cont, tuple(np.array(points).astype(float)), False
                )
                < 0
            ):
                return 0
        return 1


def filter_kwargs_by_callable(
    kwargs: tp.Dict[str, tp.Any],
    callabl: tp.Callable,
    exclude: tp.List[str] | None = None,
) -> tp.Dict[str, tp.Any]:
    """Filter a dictionary keeping only the keys which are part of a function signature."""
    from inspect import signature

    args = signature(callabl).parameters.keys()
    return {k: v for k, v in kwargs.items() if (k in args) and k not in (exclude or [])}


def screen_coords(scores, coords, top_left, bot_right):
    bot_right = np.array(bot_right)
    top_left = np.array(top_left)
    mask = np.logical_and(
        np.all(coords >= top_left, axis=1), np.all(coords <= bot_right, axis=1)
    )
    scores = scores[mask]
    coords = coords[mask]
    return scores, coords


def to_percentiles(scores):
    from scipy.stats import rankdata

    scores = rankdata(scores, "average") / len(scores) * 100
    return scores


def collate_features(batch, with_coords: bool = False):
    img = torch.cat([item[0] for item in batch], dim=0)
    if not with_coords:
        return img
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]


def is_url(url: str | Path) -> bool:
    import pathlib

    if isinstance(url, Path):
        url = url.as_posix()
    return url.startswith("http")


def download_file(
    url: str, dest: Path | str | None = None, overwrite: bool = False
) -> Path:
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


def default_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    trnsfrms_val = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )

    return trnsfrms_val


def save_hdf5(output_path, asset_dict, attr_dict=None, mode="a"):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1,) + data_shape[1:]
            maxshape = (None,) + data_shape[1:]
            dset = file.create_dataset(
                key,
                shape=data_shape,
                maxshape=maxshape,
                chunks=chunk_shape,
                dtype=data_type,
            )
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0] :] = val
    file.close()
    return output_path
