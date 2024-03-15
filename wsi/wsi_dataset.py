from torchvision import transforms
from .util_classes import (
    isInContourV1,
    isInContourV2,
    isInContourV3_Easy,
    isInContourV3_Hard,
)


def default_transforms(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    t = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )
    return t
