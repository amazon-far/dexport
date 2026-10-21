"""Image transforms: the training augmentation and the deterministic eval/deploy transform.

Input is a float tensor ``(N, 3, H, W)`` with values in [0, 255]. One call transforms every
frame and camera of a sample, so they share the same random parameters. Train: color jitter,
random crop (``im_aug_crop_ratio`` of the side, 212 of 224) resized back, random grayscale,
random Gaussian blur, then the ImageNet normalization DINOv2 expects. Eval: center crop of the
same size, resize, normalize.
"""

from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_image_transforms(config):
    """``(train_transform, eval_transform)``, or ``(None, None)`` when the policy uses no images."""
    data = config.data
    if "img" not in data.data_key:
        return None, None
    h, w = data.im_height, data.im_width
    crop = (int(h * data.im_aug_crop_ratio), int(w * data.im_aug_crop_ratio))
    to_unit = T.Normalize(mean=(0.0, 0.0, 0.0), std=(255.0, 255.0, 255.0))
    resize = T.Resize((h, w), antialias=True)
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    brightness, contrast, saturation, hue = data.im_aug_jitter
    train_ops = [
        to_unit,
        T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
        T.RandomCrop(crop),
        resize,
    ]
    # A probability of 0 drops the op entirely (it would still consume a random draw).
    if data.im_grayscale_p > 0:
        train_ops.append(T.RandomGrayscale(p=data.im_grayscale_p))
    if data.im_blur_p > 0:
        # GaussianBlur needs an odd kernel: an even size rounds up, and the smallest is 3.
        k = max(3, data.im_blur_kernel + 1 - data.im_blur_kernel % 2)
        blur = T.GaussianBlur(kernel_size=k, sigma=tuple(data.im_blur_sigma))
        train_ops.append(T.RandomApply([blur], p=data.im_blur_p))
    train = T.Compose(train_ops + [normalize])
    evaluate = T.Compose([to_unit, T.CenterCrop(crop), resize, normalize])
    return train, evaluate
