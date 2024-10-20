import argparse
import os
from datetime import datetime
from glob import glob

import cv2
import numpy as np
import torch

from utils import AvgMeter, clip_gradient
from torchvision import transforms

from model import FusionModel
from losses import CharbonnierLoss_IR, CharbonnierLoss_VI, tv_vi, tv_ir, ssim_loss
from test import get_scores


class Dataset(torch.utils.data.Dataset):

    def __init__(self, source_1_paths, source_2_paths, transform=None, type=None):
        self.source_1_paths = source_1_paths
        self.source_2_paths = source_2_paths
        self.transform = transform
        self.type = type

    def __len__(self):
        return len(self.source_1_paths)

    def __getitem__(self, idx):
        source_1_path = self.source_1_paths[idx]
        source_2_path = self.source_2_paths[idx]

        if self.type == "CT":
            source_1 = cv2.imread(source_1_path, cv2.IMREAD_GRAYSCALE)
            source_2 = cv2.imread(source_2_path, cv2.IMREAD_GRAYSCALE)

            if self.transform is not None:
                source_1 = self.transform(source_1)
                source_2 = self.transform(source_2)

            return np.asarray(source_1), np.asarray(source_2), np.asarray(source_2)
        else:
            img1 = cv2.imread(source_1_path)
            img2 = cv2.imread(source_2_path, cv2.IMREAD_GRAYSCALE)

            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)  # CT/PET/SPECT 256,256,3

            img1_Y = img1[:, :, 0:1]
            img1_CrCb = img1[:, :, 1:3].transpose(2, 0, 1)

            if self.transform:
                img1_Y = self.transform(img1_Y)
                img2 = self.transform(img2)

            return np.asarray(img1_Y), np.asarray(img2), np.asarray(img1_CrCb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=100, help="epoch number")
    parser.add_argument("--init_lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batchsize", type=int, default=4, help="training batch size")
    parser.add_argument(
        "--init_trainsize", type=int, default=256, help="training dataset size"
    )
    parser.add_argument(
        "--clip", type=float, default=0.5, help="gradient clipping margin"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="./data/Havard-Medical-Image-Fusion-Datasets/MyDatasets",
        help="path to train dataset",
    )
    parser.add_argument("--train_save", type=str, default="ours")

    args = parser.parse_args()

    device = torch.device("cuda")

    epochs = args.num_epochs
    ds = ["CT-MRI", "PET-MRI", "SPECT-MRI"]

    for _ds in ds:
        save_path = "snapshots/{}/{}/".format(args.train_save, _ds)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        else:
            print("Save path existed")

        dataset_path = _ds.split("-")

        train_img_paths = glob(
            "{}/{}/train/{}/*".format(args.train_path, _ds, dataset_path[0])
        )
        train_mask_paths = glob(
            "{}/{}/train/{}/*".format(args.train_path, _ds, dataset_path[1])
        )
        train_img_paths.sort()
        train_mask_paths.sort()

        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.ToTensor(),
            ]
        )
        train_dataset = Dataset(
            train_img_paths, train_mask_paths, transform=transform, type=dataset_path[0]
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batchsize,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        test_img_paths = glob(
            "{}/{}/train/{}/*".format(args.train_path, _ds, dataset_path[0])
        )
        test_mask_paths = glob(
            "{}/{}/train/{}/*".format(args.train_path, _ds, dataset_path[1])
        )
        test_img_paths.sort()
        test_mask_paths.sort()

        test_dataset = Dataset(
            test_img_paths, test_mask_paths, transform=transform, type=dataset_path[0]
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        model = FusionModel().to(device)

        # ---- flops and params ----
        params = model.parameters()
        total_step = len(train_loader)

        optimizer = torch.optim.Adam(params, args.init_lr, betas=(0.9, 0.999), eps=1e-8)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(train_loader) * epochs,
            eta_min=args.init_lr / 1000,
        )
        start_epoch = 1
        best_ssim = 0

        loss_record = AvgMeter()
        loss_1_record = AvgMeter()
        loss_2_record = AvgMeter()
        loss_3_record = AvgMeter()
        loss_4_record = AvgMeter()

        criterion_CharbonnierLoss_IR = CharbonnierLoss_IR
        criterion_CharbonnierLoss_VI = CharbonnierLoss_VI
        criterion_tv_ir = tv_ir
        criterion_tv_vi = tv_vi
        criterion_ssim = ssim_loss

        print("#" * 20, "Start Training", "#" * 20)
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            with torch.autograd.set_detect_anomaly(True):
                for i, pack in enumerate(train_loader, start=1):
                    if epoch <= 1:
                        optimizer.param_groups[0]["lr"] = (
                            (epoch * i) / (1.0 * total_step) * args.init_lr
                        )
                    else:
                        lr_scheduler.step()

                    optimizer.zero_grad()
                    # ---- data prepare ----
                    img_1, img_2, img_Y = pack
                    img_1 = img_1.to(device)
                    img_2 = img_2.to(device)

                    # ---- forward ----
                    out = model(img_1, img_2)

                    _CharbonnierLoss_IR = criterion_CharbonnierLoss_IR(out, img_1)
                    _CharbonnierLoss_VI = criterion_CharbonnierLoss_VI(out, img_2)
                    loss_ssim = criterion_ssim(out, img_1, img_2)

                    loss = _CharbonnierLoss_IR + _CharbonnierLoss_VI + loss_ssim

                    loss.backward()

                    clip_gradient(optimizer, args.clip)
                    optimizer.step()
                    # ---- recording loss ----
                    loss_record.update(loss.data, args.batchsize)
                    loss_1_record.update(_CharbonnierLoss_IR.data, args.batchsize)
                    loss_2_record.update(_CharbonnierLoss_VI.data, args.batchsize)
                    loss_3_record.update(loss_ssim.data, args.batchsize)

                torch.cuda.synchronize()

                # ---- train visualization ----
                print(
                    "{} Training Epoch [{:03d}/{:03d}], "
                    "[loss: {:0.4f}, CharbonnierLoss_IR: {:0.4f}, CharbonnierLoss_VI: {:0.4f}, SSIM: {:0.4f}]".format(
                        datetime.now(),
                        epoch,
                        args.num_epochs,
                        loss_record.show(),
                        loss_1_record.show(),
                        loss_2_record.show(),
                        loss_3_record.show(),
                    )
                )

        ckpt_path = save_path + "last.pth"
        print("[Saving Checkpoint:]", ckpt_path)
        torch.save(model.state_dict(), ckpt_path)
