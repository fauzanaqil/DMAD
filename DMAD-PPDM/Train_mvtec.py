import os
import cv2
import time
import torch
import random
import numpy as np
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from test import evaluation, cal_anomaly_map, min_max_norm, cvt2heatmap, show_cam_on_image
from dataset import MVTecDataset
from resnet import wide_resnet50_2
from torch.nn import functional as F
from dataset import get_data_transforms
from de_resnet import de_wide_resnet50_2
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.ndimage import gaussian_filter


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.backends.cudnn.benchmark = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

ifgeom = ['screw', 'carpet', 'metal_nut']

def setup_tensorboard(log_dir):
    return SummaryWriter(log_dir)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def loss_function(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        # print(a[item].shape)
        # print(b[item].shape)
        loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1), b[item].view(b[item].shape[0], -1)))
    return loss

def loss_concat(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map, 1)
    b_map = torch.cat(b_map, 1)
    loss += torch.mean(1 - cos_loss(a_map, b_map))
    return loss

def train_with_tensorboard(_class_, root='./mvtec/', ckpt_path='./ckpt/', ifgeom=None, tensorboard_log_dir='./runs/DMAD/'):
    print(_class_)
    epochs = 50
    image_size = 256
    mode = "sp"
    gamma = 1
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vq = mode == "sp"
    print(device)

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = root + _class_ + '/train'
    test_path = root + _class_
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    ckp_path = ckpt_path + 'wres50_' + _class_ + ('_I.pth' if mode == "sp" else '_P.pth')
    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn, offset = wide_resnet50_2(pretrained=True, vq=vq, gamma=gamma)
    encoder = encoder.to(device)
    bn = bn.to(device)
    offset = offset.to(device)
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    encoder.eval()

    optimizer = torch.optim.AdamW(list(offset.parameters()) + list(decoder.parameters()) + list(bn.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    writer = setup_tensorboard(tensorboard_log_dir)

    step = 0
    order = 0
    for epoch in range(epochs):
        start_time = time.time()
        losses = []

        offset.train()
        bn.train()
        decoder.train()
        loss_rec = {"main": [0],
                    "offset": [0],
                    "vq": [0]}
        for k, (img, label) in tqdm(enumerate(train_dataloader)):
            img = img.to(device)
            _, img_, offset_loss = offset(img)
            inputs = encoder(img_)
            vq, vq_loss = bn(inputs)
            outputs = decoder(vq)

            main_loss = loss_function(inputs, outputs)
            loss = main_loss + offset_loss + vq_loss
            
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            
            loss_rec["main"].append(main_loss.item())
            loss_rec["offset"].append(offset_loss.item())
            try:
                loss_rec["vq"].append(vq_loss.item())
            except:
                loss_rec["vq"].append(0)
        end_time = time.time()
        epoch_time = end_time - start_time
        print('epoch [{}/{}], main_loss:{:.4f}, offset_loss:{:.4f}, vq_loss:{:.4f}, epoch_time:{}m{}s'.format(
            epoch + 1, epochs, 
            np.mean(loss_rec["main"]), np.mean(loss_rec["offset"]), np.mean(loss_rec["vq"]), 
            int(epoch_time // 60), int(epoch_time % 60)))

        if (epoch + 1) % 10 == 0:
            auroc = evaluation(offset, encoder, bn, decoder, test_dataloader, device, _class_, mode, ifgeom, order)
            writer.add_scalar("AUC-ROC", auroc, global_step=epoch)
            torch.save({
                'offset': offset.state_dict(),
                'bn': bn.state_dict(),
                'decoder': decoder.state_dict()}, ckp_path)
            print('AUC-ROC:{:.3f}'.format(auroc))

        writer.add_scalar("Training loss", loss, global_step=step)
        writer.add_scalar("Training main loss", main_loss, global_step=step)
        writer.add_scalar("Training offset loss", offset_loss, global_step=step)
        writer.add_scalar("Training vq loss", vq_loss, global_step=step)

        step += 1

        scheduler.step()

if __name__ == '__main__':
    root_path = "D:\\Fauzan\\Study PhD\\Research\\DMAD\\mvtec_anomaly_detection\\"
    ckpt_path = "D:\\Fauzan\\Study PhD\\Research\\DMAD\\dataset\\ckpt\\ppdm\\"
    setup_seed(111)
    learning_rate = 0.005
    batch_size = 8
    item_list = ['capsule', 'cable','screw','pill','carpet', 'bottle', 'hazelnut','leather', 'grid','transistor', 'metal_nut', 'toothbrush', 'zipper', 'tile', 'wood']
    for i in item_list:
        start_total_time = time.time()

        tensorboard_log_dir = f"D:\\Fauzan\\Study PhD\\Research\\DMAD\\runs\\DMAD\\BatchSize{batch_size}_LR{learning_rate}\\{i}"
        train_with_tensorboard(i, root_path, ckpt_path, ifgeom=i in ifgeom, tensorboard_log_dir=tensorboard_log_dir)

        end_total_time = time.time()  # Record the end time of the entire training
        total_training_time = end_total_time - start_total_time  # Calculate the total training time

        total_hours, remainder = divmod(total_training_time, 3600)
        total_minutes, total_seconds = divmod(remainder, 60)

        print('Total Training Time: {} hours {} minutes {} seconds'.format(int(total_hours), int(total_minutes), int(total_seconds)))