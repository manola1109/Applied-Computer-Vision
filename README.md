# Object Detection with Faster R-CNN

This repository contains a PyTorch implementation of an object detection model using Faster R-CNN with ResNet-50 backbone, trained on the Pascal VOC dataset.

## Project Overview

This project demonstrates how to:
- Load and preprocess the Pascal VOC dataset
- Train a Faster R-CNN object detection model
- Evaluate model performance using precision, recall, F1 score, and mAP
- Make predictions on new images

## Dataset

The Pascal VOC dataset contains images with bounding box annotations for 20 object categories:
- Person: person
- Animal: bird, cat, cow, dog, horse, sheep
- Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
- Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor

The dataset is structured with:
- Training images: 16,551 objects across 5,717 images
- Validation images: 4,952 objects across 1,444 images

## Model Architecture

- **Base Model**: Faster R-CNN
- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
- **Pretrained**: ImageNet weights

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/object-detection-project.git
cd object-detection-project
```

2. Install dependencies:
```bash
pip install torch torchvision
pip install opencv-python matplotlib tqdm
pip install torchmetrics[detection]
```

## Usage

### Dataset Preparation

The dataset should be organized as follows:
```
dataset_pascalVOCDetection-XXXXXX-XXXXXX/
├── train.csv
├── val.csv
├── train_images/
│   ├── image1.jpg
│   └── ...
└── val_images/
    ├── image1.jpg
    └── ...
```

Each CSV file should contain columns for image filename, class, and bounding box coordinates (xmin, ymin, xmax, ymax).

### Training

To train the model:

```python
# Run the notebook cells in sequence or use the provided training script
python train.py --data_path /path/to/dataset --epochs 10 --batch_size 4
```

### Inference

To detect objects in new images:

```python
# Load the model
model = create_model(num_classes)
model.load_state_dict(torch.load('best_pascal_voc_model.pth'))
model.to(device)
model.eval()

# Detect objects
results = detect_objects_in_image('path/to/image.jpg', model, class_names)
```

## Evaluation Metrics

The model is evaluated using:
- Precision
- Recall
- F1 Score
- mAP (mean Average Precision)

## Results

The model achieves:
- mAP@0.5: X.XX
- Precision: X.XX
- Recall: X.XX
- F1 Score: X.XX

## Example Detections

![Example Detection](example_detection.jpg)

## Future Improvements

- Data augmentation to improve model robustness
- Test with different backbones (ResNet-101, EfficientNet)
- Implement real-time object detection
- Fine-tune hyperparameters to improve performance

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Pascal VOC dataset
- PyTorch and torchvision teams
- [Original Faster R-CNN paper](https://arxiv.org/abs/1506.01497)
