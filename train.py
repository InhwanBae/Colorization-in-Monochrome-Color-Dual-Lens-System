import os
import pickle
import argparse
import torch

from tqdm import tqdm
from model import MCDLNet
from dataloader import SceneFlowDataloader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# Argument parsing
parser = argparse.ArgumentParser()

# Model specific parameters
parser.add_argument('--dataset', default='./SceneFlow/', help='Dataset folder')
parser.add_argument('--input_h', type=int, default=256)
parser.add_argument('--input_w', type=int, default=512)

# Training specific parameters
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=1024, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=128, help='Number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true", default=False, help='Use lr rate scheduler')
parser.add_argument('--tag', default='tag', help='Personal tag for the model')

args = parser.parse_args()

# Dataloader
train_dataset = SceneFlowDataloader(dataset_dir=args.dataset, training=True, size=(512, 256))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

valid_dataset = SceneFlowDataloader(dataset_dir=args.dataset, training=False, size=(512, 256))
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

# Model preparation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MCDLNet(n=32, d=160)
model = model.to(device)
model = torch.nn.DataParallel(model)

optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
if args.use_lrschd:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.8)

# Loss function
criterion = torch.nn.MSELoss()

# Train logging
checkpoint_dir = './checkpoint/' + args.tag + '/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
with open(checkpoint_dir + 'args.pkl', 'wb') as f:
    pickle.dump(args, f)

writer = SummaryWriter(checkpoint_dir)
metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 1e10}
test_metrics = {'min_test_epoch': -1, 'min_test_loss': 1e10}


def train(epoch):
    global metrics, model
    model.train()
    loss_all = 0.
    #loader_len = len(train_loader)
    loader_len = 100
    progressbar = tqdm(range(loader_len))
    progressbar.set_description('Train Epoch: {0} Loss: {1:.8f}'.format(epoch, 0))

    for batch_idx, batch in enumerate(train_loader):
        m_img, c_img, gt_img = [tensor.to(device) for tensor in batch]

        optimizer.zero_grad()
        Cb = model(m_img, c_img[:, 0:1], c_img[:, 1:2])
        Cb_loss = criterion(Cb, gt_img[:, 1:2])
        Cb_loss.backward()
        Cb_loss = Cb_loss.item()

        Cr = model(m_img, c_img[:, 0:1], c_img[:, 2:3])
        Cr_loss = criterion(Cr, gt_img[:, 2:3])
        Cr_loss.backward()
        Cr_loss = Cr_loss.item()

        # replaced with gradient accumulation method because of memory efficiency
        optimizer.step()

        loss = Cb_loss + Cr_loss
        loss_all += loss
        iter_idx = epoch * loader_len + batch_idx
        writer.add_scalar('Loss/Train', loss, iter_idx)
        writer.add_scalar('Loss_Components/Train_Cb', Cb_loss, iter_idx)
        writer.add_scalar('Loss_Components/Train_Cr', Cr_loss, iter_idx)

        progressbar.set_description('Train Epoch: {0} Loss: {1:.8f}'.format(epoch, loss))
        progressbar.update(1)

        if batch_idx == loader_len - 1:
            break

    progressbar.close()
    metrics['train_loss'].append(loss_all / loader_len)


def valid(epoch):
    global metrics, constant_metrics, model
    model.eval()
    loss_all = 0.
    #loader_len = len(valid_loader)
    loader_len = 10
    progressbar = tqdm(range(loader_len))
    progressbar.set_description('Valid Epoch: {0} Loss: {1:.8f}'.format(epoch, 0))

    for batch_idx, batch in enumerate(valid_loader):
        m_img, c_img, gt_img = [tensor.to(device) for tensor in batch]

        Cb = model(m_img, c_img[:, 0:1], c_img[:, 1:2]).detach()
        Cb_loss = criterion(Cb, gt_img[:, 1:2]).item()

        Cr = model(m_img, c_img[:, 0:1], c_img[:, 2:3]).detach()
        Cr_loss = criterion(Cr, gt_img[:, 2:3]).item()

        loss = Cb_loss + Cr_loss

        loss_all += loss
        iter_idx = epoch * loader_len + batch_idx
        writer.add_scalar('Loss/Valid', loss, iter_idx)
        writer.add_scalar('Loss_Components/Valid_Cb', Cb_loss, iter_idx)
        writer.add_scalar('Loss_Components/Valid_Cr', Cr_loss, iter_idx)

        progressbar.set_description('Valid Epoch: {0} Loss: {1:.8f}'.format(epoch, loss))
        progressbar.update(1)

        if batch_idx == loader_len - 1:
            break

    progressbar.close()
    metrics['val_loss'].append(loss_all / loader_len)

    # Save model
    torch.save(model.state_dict(), checkpoint_dir + 'MCDLNet_' + str(epoch) + '.pth')
    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'MCDLNet_best.pth')


def main():
    for epoch in range(args.num_epochs):
        train(epoch)
        valid(epoch)

        if args.use_lrschd:
            scheduler.step()

        print(" ")
        print("Dataset: {0}, Epoch: {1}".format(args.tag, epoch))
        print("Train_loss: {0}, Val_los: {1}".format(metrics['train_loss'][-1], metrics['val_loss'][-1]))
        print("Min_val_epoch: {0}, Min_val_loss: {1}".format(constant_metrics['min_val_epoch'],
                                                             constant_metrics['min_val_loss']))
        print(" ")

        with open(checkpoint_dir + 'metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f)

        with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as f:
            pickle.dump(constant_metrics, f)


if __name__ == "__main__":
    main()

writer.close()

