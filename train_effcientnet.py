import os
import time
import numpy as np
import logging
import argparse
import torch
import torch.nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
import cv2
import yaml
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from torchvision import transforms
import torch
import os
import cv2
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from dataset.transform import xception_default_data_transforms as xception_transforms
from data_load_helper import data_load_race
from validation_helper import test_race
from save_and_load_helper import save, load
from train_helper import train_effi
from efficientnet_pytorch import EfficientNet

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./config/yaml/efficient.yaml')
    parser.add_argument("--progress", type=float)

    args = parser.parse_args()

    return args



class OrthogonalProjection(torch.nn.Module):
    """
    A linear projection layer with orthogonality regularization (SVDNet-style).
    """

    def __init__(self, input_dim, output_dim, rank=None):
        super().__init__()
        self.rank = rank or output_dim
        self.U = torch.nn.Parameter(torch.eye(output_dim))  # [output_dim, output_dim]
        self.S = torch.nn.Parameter(torch.ones(output_dim))  # [output_dim]
        self.V = torch.nn.Parameter(torch.randn(output_dim, input_dim))  # [output_dim, input_dim]

    def forward(self, x):
        S_mat = torch.diag(self.S)
        W = self.U @ S_mat @ self.V  # W: [output_dim, input_dim]
        return F.linear(x, W)

    def orthogonality_loss(self):
        Iu = torch.eye(self.U.shape[1], device=self.U.device)
        Iv = torch.eye(self.V.shape[0], device=self.V.device)
        loss_u = F.mse_loss(self.U.T @ self.U, Iu)
        loss_v = F.mse_loss(self.V @ self.V.T, Iv)
        return loss_u + loss_v

    def singular_value_loss(self, mode='sparse', r=None):
        if mode == 'sparse':
            return self.S.abs().sum()
        elif mode == 'tail' and r is not None:
            return (self.S[r:] ** 2).sum()
        else:
            return torch.tensor(0.0, device=self.S.device)


class SVDRegularizedEfficientNet(torch.nn.Module):
    def __init__(self, model_name='efficientnet-b4', proj_dim=256, num_classes=2):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained(model_name,
                                                     weights_path='/home/harryxa/Fair/CADDM/backbones/efficientnet_pytorch/efficientnet-b4-6ed6700e.pth')
        self.proj = OrthogonalProjection(input_dim=self.backbone._fc.in_features, output_dim=proj_dim)
        self.backbone._fc = torch.nn.Identity()  # remove original fc layer
        self.classifier = torch.nn.Linear(proj_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        features = self.proj(features)
        logits = self.classifier(features)
        return logits, features

    def orthogonality_regularization(self):
        return self.proj.orthogonality_loss()


    def singular_value_regularization(self, mode='sparse', r=None):
        return self.proj.singular_value_loss(mode=mode, r=r)

def main(config=None):

    args = get_args()
    if config is None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = config
    logger.info(config)
    run = wandb.init(project=config['general']['exp_name'], name=config['general']['exp_id'])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['general']['gpu_id'])

    torch.set_num_threads(4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    torch.backends.cudnn.benchmark = True

    train_dataset_list, val_dataset_list, test_dataset_list, train_loader_list, val_loader_list, test_loader_list = data_load_race(logger, config, xception_transforms)

    train_dataset = train_dataset_list['train_dataset']
    train_loader = train_loader_list['train_loader']


    model = SVDRegularizedEfficientNet(model_name='efficientnet-b4')

    model.to(device)
    # model = EfficientNet.from_name('efficientnet-b4')

    if config['general']['mode'] == "train":
        # model = torch.nn.DataParallel(model).cuda()
        logger.info(f"train_dataset.size: {len(train_dataset)}")

        criterion = torch.nn.CrossEntropyLoss(reduction=config['data']['reduction'])
        # criterion = torch.nn.CrossEntropyLoss()
        criterion_cos = torch.nn.CosineEmbeddingLoss()
        best_val_auc = 0  # best validation auc of roc curve

        for param in model.parameters():
            param.requires_grad = True
        optimizer_net = torch.optim.Adam(model.parameters(), lr=2e-4)
        for i in range(config['network']['epochs_net']):
            train_loss = train_effi(config, model, optimizer_net, criterion, criterion_cos, train_loader, device=device, lambda_ortho=config['network']['lambda_ortho'])
            logger.info(f"Network Epoch：{i + 1}/{config['network']['epochs_net']}. train loss: {train_loss}.")

            auc_list = test_race(logger, config, model, test_dataset_list, test_loader_list, device, epoch=i)

            val_auc = auc_list['auc']
            val_auc, best_val_auc = save(config, model, optimizer_net, val_auc, best_val_auc, i)

    else:
        model = load(logger, model, config)
        model.to(device)

    val_auc_list = test_race(logger, config, model, test_dataset_list, test_loader_list, device, epoch=0)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    main()
