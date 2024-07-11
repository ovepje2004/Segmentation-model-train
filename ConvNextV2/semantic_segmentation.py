import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import torch.nn as nn
import torch.optim as optim
from timm import create_model
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class Config:
    def __init__(self):
        self.dataloader_num_workers = 4
        self.batch_size = 2
        self.image_size = 1024
        self.base_lr = 0.003
        self.num_epochs = 100
        self.eval_period = 5
        self.num_classes = 14

cfg = Config()

class COCOSegmentationDataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        if len(anns) == 0:
            return self.__getitem__((index + 1) % len(self.ids))

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        mask = Image.new('L', (img.width, img.height), 0)
        for ann in anns:
            if 'segmentation' in ann and ann['segmentation']:
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((len(seg) // 2, 2))
                    ImageDraw.Draw(mask).polygon([tuple(p) for p in poly], outline=1, fill=1)

        if self.transform is not None:
            img = self.transform(img)
            mask = mask.resize((cfg.image_size, cfg.image_size), Image.NEAREST)
            mask = torch.from_numpy(np.array(mask)).long()

        return img, mask

    def __len__(self):
        return len(self.ids)

transform = transforms.Compose([
    transforms.Resize((cfg.image_size, cfg.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = COCOSegmentationDataset(root='./dataset/train/images', annFile='./dataset/train/_annotations.coco.json', transform=transform)
val_dataset = COCOSegmentationDataset(root='./dataset/valid/images', annFile='./dataset/valid/_annotations.coco.json', transform=transform)

print(f"Train dataset length: {len(train_dataset)}")
print(f"Validation dataset length: {len(val_dataset)}")

def create_dataloaders(train_dataset, val_dataset, batch_size, num_workers):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, cfg.batch_size, cfg.dataloader_num_workers)

class ConvNeXtSegmentationModel(nn.Module):
    def __init__(self, num_classes=cfg.num_classes):
        super(ConvNeXtSegmentationModel, self).__init__()
        self.model = create_model('convnextv2_base', pretrained=True)
        self.segmentation_head = nn.Conv2d(self.model.num_features, num_classes, kernel_size=1)
        
    def forward(self, x):
        features = self.model.forward_features(x)
        x = self.segmentation_head(features)
        x = nn.functional.interpolate(x, size=(cfg.image_size, cfg.image_size), mode='bilinear', align_corners=False)
        return x

model = ConvNeXtSegmentationModel(num_classes=cfg.num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg.base_lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def visualize_sample(data_loader):
    model.eval()
    with torch.no_grad():
        for inputs, masks in data_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(inputs[0].cpu().permute(1, 2, 0))
            plt.title('Input Image')

            plt.subplot(1, 3, 2)
            plt.imshow(masks[0].cpu())
            plt.title('True Mask')

            plt.subplot(1, 3, 3)
            plt.imshow(predicted[0].cpu())
            plt.title('Predicted Mask')

            plt.show()
            break

# Visualize a sample from the validation set
visualize_sample(val_loader)

for epoch in range(cfg.num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}"):
        inputs, masks = inputs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    print(f"Epoch {epoch+1}/{cfg.num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    if (epoch + 1) % cfg.eval_period == 0:
        torch.save(model.state_dict(), f'./output/convnextv2_coco_epoch{epoch+1}.pth')

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, masks in val_loader:
                inputs, masks = inputs.to(device), masks.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += masks.numel()
                correct += (predicted == masks).sum().item()

        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), './output/convnextv2_coco_final.pth')
