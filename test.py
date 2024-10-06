import argparse
import os
from glob import glob

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import FusionModel
from torchvision.utils import save_image
from eval import psnr, ssim, mutual_information
from evaluation_metrics import fsim, nmi, en


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
        source_1_name = source_1_path.split("/")[-1]

        if self.type == "CT":
            source_1 = cv2.imread(source_1_path, cv2.IMREAD_GRAYSCALE)
            source_2 = cv2.imread(source_2_path, cv2.IMREAD_GRAYSCALE)

            if self.transform is not None:
                source_1 = self.transform(source_1)
                source_2 = self.transform(source_2)

            return (
                np.asarray(source_1),
                np.asarray(source_2),
                np.asarray(source_2),
                source_1_name,
            )
        else:
            img1 = cv2.imread(source_1_path)
            img2 = cv2.imread(source_2_path, cv2.IMREAD_GRAYSCALE)

            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)  # CT/PET/SPECT 256,256,3

            img1_Y = img1[:, :, 0:1]
            img1_CrCb = img1[:, :, 1:3].transpose(2, 0, 1)

            if self.transform:
                img1_Y = self.transform(img1_Y)
                img2 = self.transform(img2)

            return (
                np.asarray(img1_Y),
                np.asarray(img2),
                np.asarray(img1_CrCb),
                source_1_name,
            )


def get_scores(src_1, src_2, prs):
    psnrs = []
    ssims = []
    nmis = []
    mis = []
    fsims = []
    ens = []

    for gt1, gt2, pr in zip(src_1, src_2, prs):
        gt1 = gt1.squeeze(0).squeeze(0).cpu().clamp(min=0, max=1)
        gt2 = gt2.squeeze(0).squeeze(0).cpu().clamp(min=0, max=1)
        pr = pr.squeeze(0).squeeze(0).detach().cpu().clamp(min=0, max=1)

        psnr_val1 = psnr(pr, gt1)
        psnr_val2 = psnr(pr, gt2)
        psnr_val = (psnr_val1 + psnr_val2) / 2
        psnrs.append(psnr_val)

        ssim_val1 = ssim(pr, gt1)
        ssim_val2 = ssim(pr, gt2)
        ssim_val = (ssim_val1 + ssim_val2) / 2
        ssims.append(ssim_val)

        nmi_val1 = nmi(pr, gt1)
        nmi_val2 = nmi(pr, gt2)
        nmi_val = (nmi_val1 + nmi_val2) / 2
        nmis.append(nmi_val)

        mi_val1 = mutual_information(pr, gt1)
        mi_val2 = mutual_information(pr, gt2)
        mi_val = (mi_val1 + mi_val2) / 2
        mis.append(mi_val)

        fsim_val1 = fsim(pr, gt1)
        fsim_val2 = fsim(pr, gt2)
        fsim_val = (fsim_val1 + fsim_val2) / 2
        fsims.append(fsim_val)

        en_val = en(pr)
        ens.append(en_val)

    _ssims = sum(ssims) / len(ssims)
    print("psnrs")
    print(sum(psnrs) / len(psnrs))
    print("ssims")
    print(sum(ssims) / len(ssims))
    print("nmis")
    print(sum(nmis) / len(nmis))
    print("mis")
    print(sum(mis) / len(mis))
    print("fsims")
    print(sum(fsims) / len(fsims))
    print("entropy")
    print(sum(ens) / len(ens))

    return _ssims


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init_trainsize", type=int, default=256, help="training dataset size"
    )
    parser.add_argument(
        "--clip", type=float, default=0.5, help="gradient clipping margin"
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="./data/Havard-Medical-Image-Fusion-Datasets/MyDatasets",
        help="path to train dataset",
    )
    parser.add_argument("--train_save", type=str, default="ours")
    args = parser.parse_args()

    device = torch.device("cuda")

    ds = ["PET-MRI", "SPECT-MRI"]
    for _ds in ds:
        save_path = "results/{}/".format(_ds)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        else:
            print("Save path existed")

        dataset_path = _ds.split("-")

        train_img_paths = []
        train_mask_paths = []
        train_img_paths = glob(
            "{}/{}/test/{}/*".format(args.test_path, _ds, dataset_path[0])
        )
        train_mask_paths = glob(
            "{}/{}/test/{}/*".format(args.test_path, _ds, dataset_path[1])
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

        dataset = Dataset(
            train_img_paths, train_mask_paths, transform=transform, type=dataset_path[0]
        )

        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=True
        )

        saved_model = f"snapshots/ours/{_ds}/best.pth"
        model = FusionModel().to(device)
        state_dict = torch.load(saved_model, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

        torch.cuda.empty_cache()
        model.eval()

        src_1 = []
        src_2 = []
        src_3 = []
        prs = []
        for i, pack in enumerate(test_loader, start=1):
            img_1, img_2, img_3, img_name = pack
            img_1 = img_1.to(device)
            img_2 = img_2.to(device)

            res = model(img_1, img_2)
            prs.append(res)

            res = res.data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8) * 255
            fused_img = res

            src_1.append(img_1)
            src_2.append(img_2)
            src_3.append(img_3)
            # prs.append(res.astype(np.uint8))

            # res = res.data.cpu().numpy()
            # res = (res - res.min()) / (res.max() - res.min() + 1e-8) * 255
            # img = res
            # fused_img = np.concatenate((img, img_3), axis=1).squeeze()
            # fused_img = np.transpose(fused_img, (1, 2, 0))
            # fused_img = fused_img.astype(np.uint8)
            # fused_img = cv2.cvtColor(fused_img, cv2.COLOR_YCrCb2BGR)
            # print(f"{save_path}/{img_name[0]}")
            cv2.imwrite(f"{save_path}/{img_name[0]}", fused_img)

        get_scores(src_1, src_2, prs)
