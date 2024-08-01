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
import pynvml
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial

class Args:
    epochs = 50
    batch_size = 2
    root = './dataset'
    model_path = './output/maskrcnn_convnext_dice_epoch19.pth'
    num_workers = 0
    num_classes = 13
    eval_period = 1

def init_distributed_training(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5001'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

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

def resize_to_mask(img, masks):
    # 마스크 크기에 맞춰 이미지 크기를 조정
    mask_size = masks.shape[-2:]
    img = F.resize(img, mask_size)
    return img, masks

def to_tensor_transform(img, masks):
    return F.to_tensor(img), torch.as_tensor(masks, dtype=torch.uint8)

def normalize_transform(img, masks):
    img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return img, masks

def random_horizontal_flip_transform(img, masks):
    if torch.rand(1) < 0.5:
        img = F.hflip(img)
        masks = F.hflip(masks)
    return img, masks

def random_rotation_transform(img, masks):
    angle = torch.randint(-10, 10, (1,)).item()
    img = F.rotate(img, angle)
    masks = F.rotate(masks, angle)
    return img, masks

def color_jitter_transform(img, masks):
    transform = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
    img = transform(img)
    return img, masks

def apply_train_transforms(img, masks):
    transforms = [
        resize_to_mask,
        random_horizontal_flip_transform,
        random_rotation_transform,
        color_jitter_transform,
        to_tensor_transform,
        normalize_transform
    ]
    for transform in transforms:
        img, masks = transform(img, masks)
    return img, masks

def apply_val_transforms(img, masks):
    transforms = [
        resize_to_mask,
        to_tensor_transform,
        normalize_transform
    ]
    for transform in transforms:
        img, masks = transform(img, masks)
    return img, masks

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

    # Display the input image
    image = images[0].permute(1, 2, 0).cpu().numpy()
    ax[0].imshow(image)
    ax[0].set_title("Input Image")

    # Display the true mask
    true_mask = targets[0]['masks'].cpu().numpy().sum(axis=0)
    ax[1].imshow(true_mask, cmap='viridis')
    ax[1].set_title("True Mask")

    # Display the predicted mask
    pred_mask = predictions[0]['masks'].detach().cpu().numpy().sum(axis=0)[0]
    ax[2].imshow(pred_mask, cmap='viridis')
    ax[2].set_title("Predicted Mask")

    # Overlay the predicted mask on the input image
    overlay = image.copy()
    for i in range(predictions[0]['masks'].shape[0]):
        mask = predictions[0]['masks'][i, 0].cpu().numpy()
        color = np.array(mcolors.to_rgb(mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS.keys())[i % 10]]))
        overlay[mask > 0.5] = overlay[mask > 0.5] * 0.5 + color * 0.5

    ax[3].imshow(overlay)
    ax[3].set_title("Predicted Mask Overlay")

    plt.savefig(f'./output/visualization_epoch{epoch}_iter{iteration}.png')
    plt.show()
    plt.close()

def main(rank, world_size):
    opts = Args()
    init_distributed_training(rank, world_size)
    setup_for_distributed(rank == 0)
    local_gpu_id = rank

    train_dataset = CocoDataset(
        img_folder=os.path.join(opts.root, 'train/images'),
        ann_file=os.path.join(opts.root, 'train/train_annotations.coco.json'),
        transform=apply_train_transforms
    )
    val_dataset = CocoDataset(
        img_folder=os.path.join(opts.root, 'valid/images'),
        ann_file=os.path.join(opts.root, 'valid/valid_annotations.coco.json'),
        transform=apply_val_transforms
    )

    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(dataset=val_dataset, shuffle=False, num_replicas=world_size, rank=rank)
    
    train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, sampler=train_sampler, collate_fn=collate_fn, num_workers=opts.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=opts.batch_size, sampler=val_sampler, collate_fn=collate_fn, num_workers=opts.num_workers)

    model = get_instance_segmentation_model(opts.num_classes).to(local_gpu_id)
    ddp_model = DDP(model, device_ids=[local_gpu_id])

    if os.path.exists(opts.model_path):
        ddp_model.load_state_dict(torch.load(opts.model_path, map_location='cuda:{}'.format(local_gpu_id)))
        print(f"Loaded model from {opts.model_path}")

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            inputs = inputs.sigmoid()
            targets = targets.float()  # Ensure targets are float
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

    # Loss function instantiations
    iou_loss_fn = IoULoss()
    dice_loss_fn = DiceLossWithEmptyPenalty()
    focal_loss_fn = FocalLoss()

    def get_gpu_memory_usage():
        # NVML 사용
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(rank)  # GPU rank
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"  Used memory: {info.used / (1024**2):.2f} MB")
        pynvml.nvmlShutdown()

        return info.used/(1024**2)

    start_epoch = 19
    for epoch in range(start_epoch, opts.epochs):
        ddp_model.train()
        train_sampler.set_epoch(epoch)
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{opts.epochs}", disable=rank != 0) as pbar:
            for i, batch in enumerate(train_dataloader):
                if batch is None:
                    if rank == 0:
                        print(f"Skipping batch {i} due to None value.")
                    continue

                images, targets = batch
                try:
                    images = list(image.to(local_gpu_id) for image in images)
                    targets = [{k: v.to(local_gpu_id) for k, v in t.items()} for t in targets]

                    if any(len(target["boxes"]) == 0 for target in targets):
                        continue

                    loss_dict = ddp_model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    ddp_model.eval()
                    with torch.no_grad():
                        predictions = ddp_model(images)
                    ddp_model.train()

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
                            iou_loss = iou_loss_fn(mask_prediction, mask_target)

                            total_loss = dice_loss + focal_loss + iou_loss
                            losses += total_loss

                    optimizer.zero_grad()
                    losses.backward()

                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)

                    optimizer.step()

                    if i % 10 == 0 and rank == 0:
                        pbar.set_postfix(loss=losses.item())
                    if rank == 0:
                        pbar.update(1)

                    if i % 3 == 0:
                        used_memory = get_gpu_memory_usage()
                        if used_memory >= 22500.00:
                            torch.cuda.empty_cache()

                    if i % 50 == 0 and rank == 0:
                        ddp_model.eval()
                        with torch.no_grad():
                            visualize_sample(images, targets, predictions, epoch+1, i)
                        ddp_model.train()
                    del images, targets, predictions, loss_dict, losses
                except Exception as e:
                    if rank == 0:
                        print(f"Error processing batch {i}: {e}")

        lr_scheduler.step()

        if epoch % opts.eval_period == 0 and rank == 0:
            ddp_model.eval()
            torch.save(ddp_model.state_dict(), f'./output/maskrcnn_convnext_dice_epoch{epoch+1}.pth')

    if rank == 0:
        torch.save(ddp_model.state_dict(), './output/maskrcnn_convnext_dice_final.pth')
    dist.barrier()
    cleanup()

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main,
             args=(world_size,),
             nprocs=world_size,
             join=True)
