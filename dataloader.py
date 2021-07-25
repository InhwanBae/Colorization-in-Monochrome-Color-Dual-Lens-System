import os
import random
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from utils import AddGaussianNoise, Grayscale, rgb_to_ycbcr, ycbcr_to_rgb

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']

IMAGENET_STATS = {'mean': [0.485, 0.456, 0.406],
                  'std': [0.229, 0.224, 0.225]}


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def get_transform(std=0., grayscale=False, normalize=IMAGENET_STATS):
    if grayscale:
        transform_list = [transforms.ToTensor(),
                          Grayscale(),
                          AddGaussianNoise(mean=0., std=std),
                          transforms.Normalize(**normalize)]
    else:
        transform_list = [transforms.ToTensor(),
                          AddGaussianNoise(mean=0., std=std),
                          transforms.Normalize(**normalize)]
    return transforms.Compose(transform_list)


class SceneFlowDataloader(Dataset):
    def __init__(self, dataset_dir='./SceneFlow/', training=True, size=(512, 256)):
        subdir1 = ['frames_cleanpass']
        subdir2 = ['TRAIN' if training else 'TEST']
        subdir3 = ['A', 'B', 'C']

        left_all = []
        right_all = []

        for i in subdir1:
            for j in subdir2:
                for k in subdir3:
                    _subdir = dataset_dir + i + '/' + j + '/' + k + '/'
                    subdir4 = os.listdir(_subdir)
                    for l in subdir4:
                        left_list = os.listdir(_subdir + l + '/left/')
                        right_list = os.listdir(_subdir + l + '/right/')
                        left_all.extend([_subdir + l + '/left/' + m for m in left_list])
                        right_all.extend([_subdir + l + '/right/' + m for m in right_list])

                        if not left_list.sort() == right_list.sort():
                            assert 'Not the same left-right_all image list.'

        for files in left_all + right_all:
            if not is_image_file(files):
                assert 'Not image file included.'

        self.size = size
        self.len = len(left_all)
        self.left = left_all
        self.right = right_all
        self.training = training

        self.color_transform = get_transform(std=0.03)  # Setup1
        # self.color_transform = get_transform(std=0.07)  # Setup2
        self.mono_transform = get_transform(std=0.01)
        self.gt_transform = get_transform()

    def __getitem__(self, index):
        left_dir = self.left[index]
        right_dir = self.right[index]
        left_img = default_loader(left_dir)
        right_img = default_loader(right_dir)

        if self.training:
            w, h = left_img.size
            w_i, h_i = self.size
            dw_r = random.randint(0, w - w_i)
            dh_r = random.randint(0, h - h_i)
        else:
            w, h = left_img.size
            # w_i, h_i = (w // 32) * 32, (h // 32) * 32,
            w_i, h_i = self.size
            dw_r = (w - w_i) // 2
            dh_r = (h - h_i) // 2

        left_img = left_img.crop((dw_r, dh_r, dw_r + w_i, dh_r + h_i))
        right_img = right_img.crop((dw_r, dh_r, dw_r + w_i, dh_r + h_i))

        mono_img = self.mono_transform(left_img)[:1]
        color_img = self.color_transform(right_img)
        gt_img = self.gt_transform(left_img)
        return mono_img, rgb_to_ycbcr(color_img), rgb_to_ycbcr(gt_img)

    def __len__(self):
        return self.len


class MiddleburyDataloader(Dataset):
    def __init__(self, dataset_dir='./Middlebury/', training=False, size=(384, 352)):
        left_all = []
        right_all = []

        subdir = os.listdir(dataset_dir)
        for l in subdir:
            left_all.append(dataset_dir + l + '/view1.png')
            right_all.append(dataset_dir + l + '/view5.png')

        for files in left_all + right_all:
            if not is_image_file(files):
                assert 'Not image file included.'

        self.size = size
        self.len = len(left_all)
        self.left = left_all
        self.right = right_all
        self.training = training

        self.color_transform = get_transform(std=0.03)  # Setup1
        # self.color_transform = get_transform(std=0.07)  # Setup2
        self.mono_transform = get_transform(std=0.01)
        self.gt_transform = get_transform()

    def __getitem__(self, index):
        left_dir = self.left[index]
        right_dir = self.right[index]
        left_img = default_loader(left_dir)
        right_img = default_loader(right_dir)

        if self.training:
            w, h = left_img.size
            w_i, h_i = self.size
            dw_r = random.randint(0, w - w_i)
            dh_r = random.randint(0, h - h_i)
        else:
            w, h = left_img.size
            w_i, h_i = self.size
            dw_r = (w - w_i) // 2
            dh_r = (h - h_i) // 2

        left_img = left_img.crop((dw_r, dh_r, dw_r + w_i, dh_r + h_i))
        right_img = right_img.crop((dw_r, dh_r, dw_r + w_i, dh_r + h_i))

        mono_img = self.mono_transform(left_img)[:1]
        color_img = self.color_transform(right_img)
        gt_img = self.gt_transform(left_img)
        return mono_img, rgb_to_ycbcr(color_img), rgb_to_ycbcr(gt_img)

    def __len__(self):
        return self.len


if __name__ == "__main__":
    temp_dataset = SceneFlowDataloader()
    temp_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=4, shuffle=True, num_workers=0)
    m_img, c_img, gt_img = next(iter(temp_loader))
    print(m_img.shape, c_img.shape, gt_img.shape)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(m_img.clamp(min=0., max=1.).numpy()[0, 0], cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(ycbcr_to_rgb(c_img).clamp(min=0., max=1.).numpy().transpose([0, 2, 3, 1])[0])
    plt.subplot(1, 3, 3)
    plt.imshow(ycbcr_to_rgb(gt_img).clamp(min=0., max=1.).numpy().transpose([0, 2, 3, 1])[0])
    plt.show()
