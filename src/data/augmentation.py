"""
src/data/augmentation.py — Albumentations pipelines for train / val / TTA.
"""

import albumentations as A

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=10, border_mode=0, p=0.5),
        A.Resize(image_size, image_size),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.HueSaturationValue(10, 20, 10, p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 30.0)),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
        ], p=0.4),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=5),
        ], p=0.2),
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        A.CoarseDropout(max_holes=4, max_height=20, max_width=20, p=0.2),
        # NOTE: No ToTensorV2 — Dataset handles tensor conversion
    ])


def get_val_transform(image_size: int = 224) -> A.Compose:
    return A.Compose([A.Resize(image_size, image_size)])


def get_tta_transforms(image_size: int = 224, n: int = 5) -> list:
    base = [get_val_transform(image_size)]
    tta  = [
        A.Compose([A.HorizontalFlip(p=1.0), A.Resize(image_size, image_size)]),
        A.Compose([A.RandomBrightnessContrast(0.1, 0.1, p=1.0), A.Resize(image_size, image_size)]),
        A.Compose([A.GaussNoise(var_limit=(5., 15.), p=1.0), A.Resize(image_size, image_size)]),
        A.Compose([A.ShiftScaleRotate(0.03, 0.05, 5, p=1.0), A.Resize(image_size, image_size)]),
    ]
    return (base + tta)[:n]
