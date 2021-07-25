import os
import pickle
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from model import MCDLNet
from dataloader import MiddleburyDataloader, SceneFlowDataloader
from torch.utils.data import DataLoader
from utils import ycbcr_to_rgb

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='tag', help='Personal tag for the model')
parser.add_argument('--dataset', default='./Middlebury/', help='Dataset folder')
parser.add_argument('--visualize', action="store_true", default=False, help='Visualize output image')
test_args = parser.parse_args()

# Get arguments for training
checkpoint_dir = './checkpoint/' + test_args.tag + '/'
model_path = checkpoint_dir + 'MCDLNet_best.pth'

args_path = checkpoint_dir + '/args.pkl'
with open(args_path, 'rb') as f:
    args = pickle.load(f)

# Dataloader
test_dataset = MiddleburyDataloader(dataset_dir=test_args.dataset, training=False, size=(384, 352))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
# test_dataset = SceneFlowDataloader(dataset_dir=test_args.dataset, training=False, size=(960, 512))
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# Model preparation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MCDLNet(n=32, d=160)
model = model.to(device)
model.load_state_dict(torch.load(model_path), strict=False)

IMAGENET_STATS = {'mean': [0.485, 0.456, 0.406],
                  'std': [0.229, 0.224, 0.225]}
imagenet_mean = torch.tensor(IMAGENET_STATS['mean']).to(device)
imagenet_std = torch.tensor(IMAGENET_STATS['std']).to(device)


def test():
    global model
    model.eval()
    PSNR_all = []
    progressbar = tqdm(range(len(test_loader)))
    progressbar.set_description('Testing {}'.format(test_args.tag))

    for batch_idx, batch in enumerate(test_loader):
        m_img, c_img, gt_img = [tensor.to(device) for tensor in batch]

        Cb = model(m_img, c_img[:, 0:1], c_img[:, 1:2]).detach()
        Cr = model(m_img, c_img[:, 0:1], c_img[:, 2:3]).detach()

        pred_img = torch.zeros_like(c_img)
        pred_img[:, 0] = m_img.squeeze(dim=1)
        pred_img[:, 1] = Cb
        pred_img[:, 2] = Cr

        pred_img_rgb = ycbcr_to_rgb(pred_img).squeeze(dim=0).permute(1, 2, 0)
        gt_img_rgb = ycbcr_to_rgb(gt_img).squeeze(dim=0).permute(1, 2, 0)

        pred_img_rgb = (pred_img_rgb * imagenet_std + imagenet_mean).clamp(min=0., max=1.)
        gt_img_rgb = (gt_img_rgb * imagenet_std + imagenet_mean).clamp(min=0., max=1.)

        if test_args.visualize:
            plt.figure()
            plt.imshow(m_img.squeeze().detach().cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.savefig('middlebury_' + str(batch_idx) + '_input.png')
            plt.close()

            plt.figure()
            plt.imshow(pred_img_rgb.detach().cpu().numpy())
            plt.axis('off')
            plt.savefig('middlebury_' + str(batch_idx) + '_pred.png')
            plt.close()

            plt.figure()
            plt.imshow(gt_img_rgb.detach().cpu().numpy())
            plt.axis('off')
            plt.savefig('middlebury_' + str(batch_idx) + '_gt.png')
            plt.close()

        mse = ((pred_img_rgb - gt_img_rgb) ** 2).mean().item()
        psnr = 10 * np.log10(1/mse)
        PSNR_all.append(psnr)

        progressbar.update(1)

    progressbar.close()

    PSNR_average = sum(PSNR_all) / len(PSNR_all)
    return PSNR_average


def main():
    average_PSNR = test()
    print("Evaluating model: {}, Average PSNR: {}".format(test_args.tag, average_PSNR))


if __name__ == "__main__":
    main()
