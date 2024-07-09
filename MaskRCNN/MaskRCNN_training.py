import os
import torch
import numpy as np
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

# 데이터셋 등록
register_coco_instances("my_dataset_train", {}, "./data/dataset/train/images/_annotations.coco.json",
                        "./data/dataset/train/images/")
register_coco_instances("my_dataset_val", {}, "./data/dataset/valid/images/_annotations.coco.json",
                        "./data/dataset/valid/images/")

def custom_mapper(dataset_dict):
    # dataset_dict 복사
    dataset_dict = dataset_dict.copy()

    # 이미지 읽기
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    # 변환 적용
    transform_list = [
        T.Resize((512, 512)),  # 이미지 크기를 512x512로 리사이즈
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomBrightness(0.8, 1.2),
        T.RandomContrast(0.8, 1.2),
        T.RandomRotation(angle=[-10, 10])
    ]
    augmentation = T.AugmentationList(transform_list)
    image, transforms = T.apply_transform_gens(transform_list, image)

    # 변환된 이미지를 dataset_dict에 추가
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    # 주석 변환
    annos = [
        utils.transform_instance_annotations(annotation, transforms, image.shape[:2])
        for annotation in dataset_dict.pop("annotations")
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

cfg = get_cfg()
cfg.merge_from_file("C:/Users/US-DL-002/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 14
cfg.SOLVER.BASE_LR = 0.003
cfg.SOLVER.MAX_ITER = 100000
cfg.TEST.EVAL_PERIOD = 1000
# cfg.SOLVER.CHECKPOINT_PERIOD = 1000
# cfg.SOLVER.STEPS = [1000,2000,3000,4000,5000,6000,7000]
# cfg.SOLVER.GAMMA = 0.5
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 14

cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
