import argparse
import os
from datetime import datetime
from glob import glob

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from utils import AvgMeter, clip_gradient
from loss import loss_total
from metrics import ssim_f, SCD
from PIL import Image
from torchvision import transforms

# from val import inference
from model import FusionModel
import pytorch_msssim


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
            # source_1 = cv2.imread(source_1_path, cv2.IMREAD_GRAYSCALE)
            # source_2 = cv2.imread(source_2_path, cv2.IMREAD_GRAYSCALE)
            source_1 = Image.open(source_1_path).convert("L")
            source_2 = Image.open(source_2_path).convert("L")

            if self.transform is not None:
                source_1 = self.transform(source_1)
                source_2 = self.transform(source_2)
            return np.asarray(source_1), np.asarray(source_2), np.asarray(source_2)
        else:
            img1 = Image.open(source_1_path).convert("RGB")
            img2 = Image.open(source_2_path).convert("L")

            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)  # CT/PET/SPECT 256,256,3

            img1_Y = img1[:, :, 0:1]
            img1_CrCb = img1[:, :, 1:3].transpose(2, 0, 1)

            if self.transform:
                img1_Y = self.transform(img1_Y)
                img2 = self.transform(img2)

            return np.asarray(img1_Y), np.asarray(img2), np.asarray(img1_CrCb)


def get_scores(src_1, src_2, src_3, preds):
    for i1, i2, i3, pr in zip(src_1, src_2, src_3, preds):
        pass


@torch.no_grad()
def inference(model, dataloader, args=None):
    print("#" * 20)
    torch.cuda.empty_cache()
    model.eval()
    device = torch.device("cuda")

    src_1 = []
    src_2 = []
    src_3 = []
    prs = []
    for i, pack in enumerate(test_loader, start=1):
        img_1, img_2, img_3 = pack

        img_1 = img_1.to(device)
        img_2 = img_2.to(device)
        if img_3 != None:
            img_3 = img_3.to(device)

        res = model(img_1, img_2)
        src_1.append(img_1)
        src_2.append(img_2)
        src_3.append(img_3)
        prs.append(res)
    pass


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
        default="./data/Havard-Medical-Image-Fusion-Datasets",
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

        train_img_paths = []
        train_mask_paths = []
        train_img_paths = glob(
            "{}/{}/{}/*".format(args.train_path, _ds, dataset_path[0])
        )
        train_mask_paths = glob(
            "{}/{}/{}/*".format(args.train_path, _ds, dataset_path[1])
        )
        train_img_paths.sort()
        train_mask_paths.sort()

        # transform = A.Compose(
        #     [
        #         A.Resize(height=256, width=256),
        #     ]
        # )

        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
            ]
        )
        dataset = Dataset(
            train_img_paths, train_mask_paths, transform=transform, type=dataset_path[0]
        )
        total_dataset = len(dataset)
        training_ratio = int(total_dataset * 0.9)
        test_ratio = total_dataset - training_ratio
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [training_ratio, test_ratio]
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batchsize,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batchsize,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        print(len(train_loader))
        _total_step = len(train_loader)

        model = FusionModel().cuda()

        # ---- flops and params ----
        params = model.parameters()
        optimizer = torch.optim.AdamW(
            params, args.init_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(train_loader) * args.num_epochs,
            eta_min=args.init_lr / 1000,
        )

        start_epoch = 1

        best_result = 0
        loss_record = AvgMeter()

        l1_loss = torch.nn.L1Loss()
        ssim_loss = pytorch_msssim.ssim

        print("#" * 20, "Start Training", "#" * 20)
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            with torch.autograd.set_detect_anomaly(True):
                for i, pack in enumerate(train_loader, start=1):
                    if epoch <= 1:
                        optimizer.param_groups[0]["lr"] = (
                            (epoch * i) / (1.0 * _total_step) * args.init_lr
                        )
                    else:
                        lr_scheduler.step()

                    optimizer.zero_grad()
                    # ---- data prepare ----
                    img_1, img_2, img_Y = pack
                    img_1 = img_1.to(device)
                    img_2 = img_2.to(device)
                    
                    # ---- forward ----
                    logits = model(img_1, img_2)

                    # _l1_loss = l1_loss(logits, img_2)
                    _ssim_loss = ssim_loss(logits, img_1, img_2)
                    loss = _ssim_loss
                    loss.backward()

                    # ---- metrics ----
                    dice_score = SCD(img_1, img_2, logits)
                    # ---- backward ----
                    # clip_gradient(optimizer, args.clip)
                    optimizer.step()
                    # ---- recording loss ----
                    loss_record.update(loss.data, args.batchsize)
                    # iou.update(iou_score.data, args.batchsize)

                # ---- train visualization ----
                print(
                    "{} Training Epoch [{:03d}/{:03d}], "
                    "[loss: {:0.4f}]".format(
                        datetime.now(),
                        epoch,
                        args.num_epochs,
                        loss_record.show(),
                    )
                )
                
        ckpt_path = save_path + "last.pth"
        print("[Saving Checkpoint:]", ckpt_path)
        torch.save(model.state_dict(), ckpt_path)
