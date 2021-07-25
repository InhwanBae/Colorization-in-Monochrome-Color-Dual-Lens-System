import torch
import random


class Grayscale(object):
    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert NRGB (torch.tensor) to NYCbCr (torch.tensor)"""
    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    cb: torch.Tensor = (b - y) * 0.564 + delta
    cr: torch.Tensor = (r - y) * 0.713 + delta
    return torch.stack([y, cb, cr], -3)


def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert NYCbCr (torch.tensor) to NRGB (torch.tensor)"""
    y: torch.Tensor = image[..., 0, :, :]
    cb: torch.Tensor = image[..., 1, :, :]
    cr: torch.Tensor = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted: torch.Tensor = cb - delta
    cr_shifted: torch.Tensor = cr - delta

    r: torch.Tensor = y + 1.403 * cr_shifted
    g: torch.Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b: torch.Tensor = y + 1.773 * cb_shifted
    return torch.stack([r, g, b], -3)





