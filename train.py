from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
import numpy as np
import json
import torch
from tqdm import tqdm
import wandb
import evaluate
from torch import optim, nn
from datasets import load_dataset
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from torch.utils.data import DataLoader, random_split
import albumentations as A
import os
from PIL import Image
from datasets import Dataset
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@dataclass
class TrainingConfig:
    """
    Training configuration for the model.

    Args:
        batch_size: Batch size for training.
        epochs: Number of epochs to train for.
        learning_rate: Learning rate for the optimizer.
        background_weight: Weight for the background class.
        other_classes_weight: Weight for the other classes.
        lr_decay_rate: Learning rate decay rate.
        seed: Random seed for reproducibility.
        model_name: Name of the model to use.
        project_name: Name of the project for wandb.
        device: Device to use for training.
    """
    batch_size: int = 1
    epochs: int = 1
    learning_rate: float = 1e-4
    background_weight: float = 1.0
    other_classes_weight: float = 3.0
    lr_decay_rate: float = 0.9998
    seed: int = 42
    model_name: str = "nvidia/mit-b2"
    project_name: str = "Clothes segmentation"
    device: Optional[str] = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    def as_dict(self) -> Dict[str, Any]:
        return vars(self)


config = TrainingConfig(
    batch_size=8,
    epochs=6,
    learning_rate=1e-4,
    background_weight=1.0,
    other_classes_weight=3.0,
    lr_decay_rate=0.9998,
    model_name="nvidia/mit-b2",
    project_name="object segmentation",
)
wandb_config = config.as_dict()

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def read_nouns(nouns_file):
    with open(nouns_file, 'r') as file:
        nouns = [line.strip() for line in file.readlines()]
    return nouns

def read_coords(coords_folder, image_filenames):
    coords = []
    for filename in image_filenames:
        json_filename = os.path.splitext(filename)[0] + '.json'
        json_path = os.path.join(coords_folder, json_filename)
        with open(json_path, 'r') as file:
            data = json.load(file)
            position = data.get('position', {})
            direction = data.get('direction', {})
            coord = [
                position.get('x', 0),
                position.get('y', 0),
                position.get('z', 0),
                direction.get('x', 0),
                direction.get('y', 0),
                direction.get('z', 0)
            ]
            coords.append(coord)
    return coords


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

class CustomImageNounDataset(Dataset):
    def __init__(self, image_folder_path, nouns_file, coords_folder, transform=None):
        self.image_filenames = os.listdir(image_folder_path)
        self.image_paths = [os.path.join(image_folder_path, filename) for filename in self.image_filenames]
        self.nouns = read_nouns(nouns_file)
        self.coords = read_coords(coords_folder, self.image_filenames)
        self.transform = transforms.ToTensor() if transform is None else transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        noun = self.nouns[idx]
        coord = self.coords[idx]
        image = self.transform(image)
        return {'image': image, 'noun': noun, 'coord': coord}

image_folder_path = '/home/zhicao/output7/source'
nouns_file = '/home/zhicao/encoder/all_nouns.txt'
coords_folder = '/home/zhicao/output7/eye_source'

def custom_collate_fn(batch):
    images = [item['image'] for item in batch]
    images_batch = torch.stack(images, dim=0)
    nouns = [item['noun'] for item in batch]
    coords = [item['coord'] for item in batch]
    return {'image': images_batch, 'noun': nouns, 'coord': coords}

dataset = CustomImageNounDataset(image_folder_path, nouns_file, coords_folder, transform=None)
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
test_size = dataset_size - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
id2label: Dict[str, str] = {
    "0": "ink_cartridge",
    "1": "dslr",
    "2": "glass",
    "3": "leg",
    "4": "copy_gate",
    "5": "lid",
    "6": "paper_tray",
    "7": "lever",
    "8": "washer",
    "9": "handheld_grip",
    "10": "half_frame",
    "11": "screwdriver",
    "12": "battery",
    "13": "allen_wrench",
    "14": "drip_tray",
    "15": "cover",
    "16": "hex_nut",
    "17": "bracket",
    "18": "joy_con_controller",
    "19": "handle",
    "20": "wheel",
    "21": "joy_con_strap",
    "22": "tray",
    "23": "sd_card",
    "24": "screw",
    "25": "hexagonal_wrench",
    "26": "paper_stack",
    "27": "computer_tower",
    "28": "capsule_container",
    "29": "circle",
    "30": "button",
    "31": "power_button",
    "32": "hex_socket_head",
    "33": "stool",
    "34": "utility_cart",
    "35": "lens_cover",
    "36": "nightstand",
    "37": "gopro"
}
label2id: Dict[str, str] = {v: k for k, v in id2label.items()}
num_labels: int = len(id2label)

class CustomSegmentationModelWithCLIP(nn.Module):
    def __init__(self, config, clip_model, clip_processor):
        super(CustomSegmentationModelWithCLIP, self).__init__()
        self.processor = SegformerImageProcessor.from_pretrained(config.model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            config.model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        ).to(config.device)
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.fusion_layer = nn.Conv2d(1024, 512, kernel_size=1)

    def forward(self, images, nouns):
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(config.device)

        text_inputs = self.clip_processor(text=nouns, return_tensors="pt", padding=True, truncation=True).to(config.device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**text_inputs)

        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits
        batch_size = logits.size(0)

        # 初始化 selected_logits，使其形状与 text_features 一致
        # 假设我们想要保留空间维度信息，可以考虑使用 adaptive pooling
        selected_logits = torch.zeros(batch_size, text_features.size(1), logits.size(2), logits.size(3), device=config.device)

        for i in range(batch_size):
            class_id = int(label2id[nouns[i]])
            selected_logit = logits[i, class_id, :, :]
            selected_logits[i] = selected_logit 

        text_features_expanded = text_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, selected_logits.size(2), selected_logits.size(3))
        combined_features = torch.cat([selected_logits, text_features_expanded], dim=1)
        fused_features = self.fusion_layer(combined_features)

        return fused_features



def project_to_2d(coor, camera_params):
    x = coor[0]
    y = coor[1]
    z = coor[2]
    dx = coor[3]
    dy = coor[4]
    dz = coor[5]
    x2d = camera_params['focal_length'] * (dx / dz) + camera_params['center_x']
    y2d = camera_params['focal_length'] * (dy / dz) + camera_params['center_y']
    return x2d, y2d


def get_object_dimensions(logits, label_index, threshold=0.5):
    probs = torch.softmax(logits, dim=1)
    class_probs = probs[:, label_index, :, :]
    object_dimensions = []
    for i in range(class_probs.shape[0]):
        object_mask = class_probs[i] > threshold
        if object_mask.sum() == 0:
            object_dimensions.append(None)
            continue
        object_indices = object_mask.nonzero(as_tuple=False)
        y_indices, x_indices = object_indices[:, 0], object_indices[:, 1]
        object_height = y_indices.max() - y_indices.min() + 1
        object_width = x_indices.max() - x_indices.min() + 1
        object_dimensions.append((object_width.item(), object_height.item()))
    return object_dimensions


def draw_bbox_around_center(center, object_size, image_size):
    center_x, center_y = center
    object_width, object_height = object_size
    image_height, image_width = image_size
    
    x_min = max(center_x - object_width // 2, 0)
    x_max = min(center_x + object_width // 2, image_width)
    y_min = max(center_y - object_height // 2, 0)
    y_max = min(center_y + object_height // 2, image_height)
    
    return x_min, y_min, x_max, y_max

def iou_of_bboxes(bbox1, bbox2):
    inter_x_min = max(bbox1[0], bbox2[0])
    inter_y_min = max(bbox1[1], bbox2[1])
    inter_x_max = min(bbox1[2], bbox2[2])
    inter_y_max = min(bbox1[3], bbox2[3])
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    
    union_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) + \
                 (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) - inter_area
    
    iou = inter_area / union_area if union_area != 0 else 0
    return iou


def custom_loss(logits, coords, camera_params, label_index, image_size, device):
    batch_size = logits.size(0)
    total_loss = logits.new_zeros(1, requires_grad=True)
    for i in range(batch_size):
        position_2d = project_to_2d(coords[i], camera_params)
        print(position_2d)
        object_dims = get_object_dimensions(logits[i:i+1], label_index[i])

        if object_dims[0] is not None:
            bbox_predicted = draw_bbox_around_center(position_2d, object_dims[0], image_size)

            mask = torch.zeros(image_size, dtype=torch.float32, device=device)
            mask[bbox_predicted[1]:bbox_predicted[3], bbox_predicted[0]:bbox_predicted[2]] = 1.0

            iou = iou_of_bboxes(
                draw_bbox_around_center(position_2d, object_dims[0], image_size),
                bbox_predicted
            )
            total_loss += (1 - iou)
    return total_loss / batch_size



# ////////////////////////////////////////////////////////////////////////////////////////

def train(model, train_loader, epochs, device, camera_params, image_size):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            images = batch['image'].to(device)
            nouns = batch['noun']
            coords = batch['coord']
            
            label_indices = torch.tensor([int(label2id[noun]) for noun in nouns], dtype=torch.long, device=device)
            optimizer.zero_grad()

            logits = model(images, nouns)

            loss = custom_loss(logits, coords, camera_params, label_indices, image_size, device)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")


config = TrainingConfig()
model = CustomSegmentationModelWithCLIP(config, clip_model, clip_processor)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

camera_params = {'focal_length': 1000, 'center_x': 640/2, 'center_y': 360}
image_size = (512, 512)
train(model, train_loader, config.epochs, config.device, camera_params, image_size)