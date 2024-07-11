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
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as nnF
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class Config:
    def __init__(self):
        self.dataloader_num_workers = 4
        self.batch_size = 2
        self.image_size = 512
        self.base_lr = 0.0001  # 학습률을 낮춤
        self.num_epochs = 50
        self.eval_period = 1
        self.num_classes = 13

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

        if len(boxes) == 0:  # Skip images without annotations
            return self.__getitem__((index + 1) % len(self.ids))

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
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.ids)

class ConvNeXtBackbone(torch.nn.Module):
    def __init__(self):
        super(ConvNeXtBackbone, self).__init__()
        self.backbone = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', features_only=True, pretrained=True)
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
    return tuple(zip(*batch))

def get_transform(train, image_size):
    transforms = []
    if train:
        transforms.append(torchvision.transforms.Resize((image_size, image_size)))
    transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(transforms)

def visualize_sample(images, targets, predictions, epoch, iteration):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Display the input image
    image = images[0].cpu().permute(1, 2, 0).numpy()
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

    plt.suptitle(f'Epoch {epoch}, Iteration {iteration}')
    plt.savefig(f'./output/visualization_epoch{epoch}_iter{iteration}.png')
    plt.show()
    plt.close()

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

train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=config.dataloader_num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=config.dataloader_num_workers)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = get_instance_segmentation_model(config.num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=config.base_lr, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

class DiceLossWithEmptyPenalty(nn.Module):
    def __init__(self, smooth=1, empty_penalty_weight=8.0):
        super(DiceLossWithEmptyPenalty, self).__init__()
        self.smooth = smooth
        self.empty_penalty_weight = empty_penalty_weight

    def forward(self, inputs, targets):
        inputs = inputs.sigmoid()
        # Ensure the same shape
        if inputs.shape[-2:] != targets.shape[-2:]:
            targets = nnF.interpolate(targets.unsqueeze(1).float(), size=inputs.shape[-2:], mode="nearest").squeeze(1)

        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (inputs * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (inputs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        
        # 빈 공간 패널티 추가
        empty_space_penalty = ((1 - inputs) * targets).sum(dim=1)
        empty_space_penalty = self.empty_penalty_weight * (empty_space_penalty / targets.sum(dim=1))

        return 1 - dice.mean() + empty_space_penalty.mean()

dice_loss_fn = DiceLossWithEmptyPenalty()

for epoch in range(config.num_epochs):
    model.train()
    with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config.num_epochs}") as pbar:
        for i, (images, targets) in enumerate(train_dataloader):
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
                    losses += dice_loss

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if i % 10 == 0:
                pbar.set_postfix(loss=losses.item())
            pbar.update(1)

            if i % 100 == 0:
                model.eval()
                with torch.no_grad():
                    visualize_sample(images, targets, predictions, epoch+1, i)
                model.train()

            del images, targets, predictions, loss_dict, losses
            torch.cuda.empty_cache()

    lr_scheduler.step()

    if epoch % config.eval_period == 0:
        model.eval()
        torch.save(model.state_dict(), f'./output/maskrcnn_convnext_dice_epoch{epoch+1}.pth')
        with torch.no_grad():
            for val_images, val_targets in val_dataloader:
                val_images = list(image.to(device) for image in val_images)
                val_targets = [{k: v.to(device) for k, v in t.items()} for t in val_targets]

                loss_dict = model(val_images, val_targets)
                losses = sum(loss for loss in loss_dict.values())
                print(f"Validation Loss after epoch {epoch}: {losses.item()}")

torch.save(model.state_dict(), './output/maskrcnn_convnext_dice_final.pth')
