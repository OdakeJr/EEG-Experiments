
# vision/lib/transforms.py

from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_image_transforms(weights=None, image_size=224):
    resize_size = round(image_size * 256 / 224)

    if weights is not None:
        eval_transform = weights.transforms(
            crop_size=image_size, resize_size=resize_size
        )
        mean, std = eval_transform.mean, eval_transform.std
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        eval_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, eval_transform
