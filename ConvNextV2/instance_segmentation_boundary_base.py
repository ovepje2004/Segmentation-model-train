import os
import json
import torch
import torchvision
import timm
from PIL import Image
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.nn.functional as nnF
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import SyncBatchNorm
import pynvml
import torch.multiprocessing as mp
from collections import OrderedDict
import gc

class Config:
    def __init__(self):
        self.dataloader_num_workers = 0
        self.batch_size = 2
        self.image_size = 1024
        self.base_lr = 0.0001
        self.num_epochs = 100
        self.eval_period = 1
        self.num_classes = 13
        self.model_path = './output/maskrcnn_convnext_dice_epoch47.pth'
        self.checkpoint_period = 3000  

config = Config()

class CocoDataset(Dataset):
    def __init__(self, img_folder, ann_file, transform=None):
        self.coco = COCO(ann_file)
        self.img_folder = img_folder
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_folder, img_info['file_name'])

        if not os.path.exists(path):
            print(f"Image not found: {path}. Skipping.")
            return None

        img = Image.open(path).convert("RGB")

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowds = []

        for ann in anns:
            xmin, ymin, width, height = ann['bbox']
            xmax = xmin + width
            ymax = ymin + height
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann['category_id'])
            masks.append(coco.annToMask(ann))
            areas.append(ann['area'])
            iscrowds.append(ann.get('iscrowd', 0))

        if len(boxes) == 0:  
            print(f"No annotations found for image {img_id}. Skipping.")
            return None

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowds = torch.as_tensor(iscrowds, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.tensor([img_id])
        target["area"] = areas
        target["iscrowd"] = iscrowds

        if self.transform:
            img, target["masks"] = self.transform(img, target["masks"])

        return img, target

    def __len__(self):
        return len(self.ids)

def resize(img, masks, size):
    img = F.resize(img, size)
    masks = F.resize(masks.unsqueeze(0), size).squeeze(0)
    return img, masks

def to_tensor(img, masks):
    return F.to_tensor(img), torch.as_tensor(masks, dtype=torch.uint8)

def get_transform(train, image_size):
    transforms = []
    if train:
        transforms.extend([
            lambda img, masks: resize(img, masks, (image_size, image_size)),
            lambda img, masks: (torchvision.transforms.RandomHorizontalFlip(0.5)(img), masks),
            lambda img, masks: (torchvision.transforms.RandomRotation(10)(img), masks),
            lambda img, masks: (torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(img), masks)
        ])
    transforms.append(lambda img, masks: to_tensor(img, masks))
    return lambda img, masks: transforms[-1](*[t(img, masks) for t in transforms[:-1]][-1])

class ConvNeXtBackbone(nn.Module):
    def __init__(self):
        super(ConvNeXtBackbone, self).__init__()
        self.backbone = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', features_only=True, pretrained=False)
        self.out_channels = self.backbone.feature_info.channels()[-1]

    def forward(self, x):
        xs = self.backbone(x)
        return xs[-1]

def get_instance_segmentation_model(num_classes):
    backbone = ConvNeXtBackbone()
    backbone = SyncBatchNorm.convert_sync_batchnorm(backbone)
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], output_size=7, sampling_ratio=2
    )
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], output_size=14, sampling_ratio=2
    )
    model = MaskRCNN(backbone, num_classes=num_classes,
                     rpn_anchor_generator=anchor_generator,
                     box_roi_pool=roi_pooler,
                     mask_roi_pool=mask_roi_pooler)
    return model

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return tuple(zip(*batch))

def visualize_sample(images, targets, predictions, epoch, iteration):
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))

    image = images[0].permute(1, 2, 0).cpu().numpy()
    ax[0].imshow(image)
    ax[0].set_title("Input Image")

    true_mask = targets[0]['masks'].cpu().numpy().sum(axis=0)
    ax[1].imshow(true_mask, cmap='viridis')
    ax[1].set_title("True Mask")

    pred_mask = predictions[0]['masks'].detach().cpu().numpy().sum(axis=0)[0]
    ax[2].imshow(pred_mask, cmap='viridis')
    ax[2].set_title("Predicted Mask")

    overlay = image.copy()
    for i in range(predictions[0]['masks'].shape[0]):
        mask = predictions[0]['masks'][i, 0].cpu().numpy()
        color = np.array(mcolors.to_rgb(mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS.keys())[i % 10]]))
        overlay[mask > 0.5] = overlay[mask > 0.5] * 0.5 + color * 0.5

    ax[3].imshow(overlay)
    ax[3].set_title("Predicted Mask Overlay")

    plt.suptitle(f'Epoch {epoch}, Iteration {iteration}')
    plt.savefig(f'./output/visualization_epoch{epoch}_iter{iteration}.png')
    plt.show()
    plt.close()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = inputs.sigmoid()
        targets = targets.float()
        BCE_loss = nnF.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class DiceLossWithEmptyPenalty(nn.Module):
    def __init__(self, smooth=1, empty_penalty_weight=0):
        super(DiceLossWithEmptyPenalty, self).__init__()
        self.smooth = smooth
        self.empty_penalty_weight = empty_penalty_weight

    def forward(self, inputs, targets):
        inputs = inputs.sigmoid()
        targets = targets.float()

        if inputs.shape[-2:] != targets.shape[-2:]:
            targets = nnF.interpolate(targets.unsqueeze(1).float(), size=inputs.shape[-2:], mode="nearest").squeeze(1)

        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (inputs * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (inputs.sum(dim=1) + targets.sum(dim=1) + self.smooth)

        empty_space_penalty = ((1 - inputs) * targets).sum(dim=1)
        empty_space_penalty = self.empty_penalty_weight * (empty_space_penalty / targets.sum(dim=1))

        return 1 - dice.mean() + empty_space_penalty.mean()

class CIoULoss(nn.Module):
    def __init__(self):
        super(CIoULoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = inputs.sigmoid()
        targets = targets.float()

        pred_center_x = inputs[:, 0]
        pred_center_y = inputs[:, 1]
        pred_w = inputs[:, 2]
        pred_h = inputs[:, 3]

        target_center_x = targets[:, 0]
        target_center_y = targets[:, 1]
        target_w = targets[:, 2]
        target_h = targets[:, 3]

        pred_x1 = pred_center_x - pred_w / 2
        pred_y1 = pred_center_y - pred_h / 2
        pred_x2 = pred_center_x + pred_w / 2
        pred_y2 = pred_center_y + pred_h / 2

        target_x1 = target_center_x - target_w / 2
        target_y1 = target_center_y - target_h / 2
        target_x2 = target_center_x + target_w / 2
        target_y2 = target_center_y + target_h / 2

        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        pred_area = pred_w * pred_h
        target_area = target_w * target_h
        union_area = pred_area + target_area - inter_area

        iou = inter_area / (union_area + 1e-6)

        center_distance = (pred_center_x - target_center_x).pow(2) + (pred_center_y - target_center_y).pow(2)

        enc_x1 = torch.min(pred_x1, target_x1)
        enc_y1 = torch.min(pred_y1, target_y1)
        enc_x2 = torch.max(pred_x2, target_x2)
        enc_y2 = torch.max(pred_y2, target_y2)

        enc_diagonal = (enc_x2 - enc_x1).pow(2) + (enc_y2 - enc_y1).pow(2)

        diou = iou - (center_distance / (enc_diagonal + 1e-6))

        v = (4 / (3.141592 ** 2)) * torch.pow(torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h), 2)
        alpha = v / (1 - iou + v + 1e-6)

        ciou = diou - alpha * v

        return 1 - ciou.mean()

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = inputs.sigmoid()
        targets = targets.float()

        if inputs.shape[-2:] != targets.shape[-2:]:
            targets = nnF.interpolate(targets.unsqueeze(1).float(), size=inputs.shape[-2:], mode="nearest").squeeze(1)

        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (inputs * targets).sum(dim=1)
        union = inputs.sum(dim=1) + targets.sum(dim=1) - intersection
        iou = (intersection + 1) / (union + 1)

        return 1 - iou.mean()

def hausdorff_distance_loss(pred, target):
    pred = pred.float()
    target = target.float()
    pred_boundary = pred - nnF.max_pool2d(pred, kernel_size=3, stride=1, padding=1)
    target_boundary = target - nnF.max_pool2d(target, kernel_size=3, stride=1, padding=1)
    
    distance = torch.cdist(pred_boundary.flatten(1), target_boundary.flatten(1), p=2)
    loss = distance.mean()
    return loss

def shape_aware_loss(pred, target):
    pred_center_mass = torch.mean(pred, dim=(1, 2, 3))
    target_center_mass = torch.mean(target, dim=(1, 2, 3))
    
    shape_loss = torch.mean((pred_center_mass - target_center_mass) ** 2)
    return shape_loss

# 손실 함수 인스턴스화
ciou_loss_fn = CIoULoss()
dice_loss_fn = DiceLossWithEmptyPenalty()
focal_loss_fn = FocalLoss()
iou_loss_fn = IoULoss()

import pynvml

def get_gpu_memory_usage(rank):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(rank)  
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return info.used/(1024**2)

def init_distributed_training(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5001'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def main(rank, world_size):
    init_distributed_training(rank, world_size)
    config = Config()

    train_dataset = CocoDataset(
        img_folder='./dataset/train/images', 
        ann_file='./dataset/train/train_annotations.coco.json', 
        transform=get_transform(train=True, image_size=config.image_size)
    )
    val_dataset = CocoDataset(
        img_folder='./dataset/valid/images', 
        ann_file='./dataset/valid/valid_annotations.coco.json', 
        transform=get_transform(train=False, image_size=config.image_size)
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, collate_fn=collate_fn, num_workers=config.dataloader_num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=val_sampler, collate_fn=collate_fn, num_workers=config.dataloader_num_workers)

    device = torch.device('cuda:{}'.format(rank))

    model = get_instance_segmentation_model(config.num_classes).to(device)
    
    if os.path.exists(config.model_path):
        checkpoint = torch.load(config.model_path, map_location=device)

        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace('module.', '')  
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        print(f"Loaded model from {config.model_path}")

    model = DDP(model, device_ids=[rank])

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config.base_lr, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    start_epoch = 47
    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config.num_epochs}", disable=rank != 0) as pbar:
            for i, batch in enumerate(train_dataloader):
                if batch is None:
                    print(f"Skipping batch {i} due to None value.")
                    continue

                images, targets = batch
                try:
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    if any(len(target["boxes"]) == 0 for target in targets):
                        continue

                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    model.eval()
                    with torch.no_grad():
                        predictions = model(images)
                    model.train()

                    for j in range(len(predictions)):
                        mask_predictions = predictions[j]['masks']
                        mask_targets = targets[j]['masks']

                        if len(mask_predictions) != len(mask_targets):
                            continue

                        for k in range(len(mask_predictions)):
                            mask_prediction = mask_predictions[k]
                            mask_target = mask_targets[k]

                            if mask_target.dim() == 2:
                                mask_target = mask_target.unsqueeze(0).unsqueeze(0)
                            elif mask_target.dim() == 3:
                                mask_target = mask_target.unsqueeze(1)

                            if mask_prediction.shape[-2:] != mask_target.shape[-2:]:
                                mask_target = nnF.interpolate(mask_target, size=mask_prediction.shape[-2:], mode="nearest")

                            mask_prediction = mask_prediction.squeeze(1)
                            mask_target = mask_target.squeeze(1)

                            dice_loss = dice_loss_fn(mask_prediction, mask_target)
                            focal_loss = focal_loss_fn(mask_prediction, mask_target)
                            ciou_loss = ciou_loss_fn(mask_prediction, mask_target)
                            iou_loss = iou_loss_fn(mask_prediction, mask_target)
                            
                            hausdorff_loss = hausdorff_distance_loss(mask_prediction, mask_target)
                            shape_loss = shape_aware_loss(mask_prediction, mask_target)

                            total_loss = dice_loss + focal_loss + 0.45 * ciou_loss + iou_loss + 0.5 * hausdorff_loss + 0.5 * shape_loss
                            losses += total_loss

                    optimizer.zero_grad()
                    losses.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()

                    if i % 10 == 0:
                        pbar.set_postfix(loss=losses.item())
                    pbar.update(1)

                    if i % 3 == 0:
                        used_memory = get_gpu_memory_usage(rank)
                        if used_memory >= 22000.00:
                            torch.cuda.empty_cache()

                    if i % config.checkpoint_period == 0:
                        model.eval()
                        with torch.no_grad():
                            visualize_sample(images, targets, predictions, epoch+1, i)
                        model.train()
                        torch.cuda.empty_cache()

                    del images, targets, predictions, loss_dict, losses
                except Exception as e:
                    print(f"Error processing batch {i}: {e}")

        lr_scheduler.step()

        if epoch % config.eval_period == 0 and rank == 0:
            model.eval()
            gc.collect()
            torch.save(model.state_dict(), f'./output/maskrcnn_convnext_dice_epoch{epoch+1}.pth')
    
    if rank == 0:
        torch.save(model.state_dict(), './output/maskrcnn_convnext_dice_final.pth')

    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    try:
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True, start_method='fork')
    except Exception as e:
        print(f"Training failed: {e}")
        cleanup()
