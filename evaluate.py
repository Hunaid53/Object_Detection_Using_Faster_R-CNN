import torch
import torchvision
from torchvision.transforms import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

test_images = [
    r"D:\College {BVM}\{#2nd Semester#}\Mini Project\Project\Object-Detection-using-Faster-R-CNN\images\test\Persons.jpeg",
    r"D:\College {BVM}\{#2nd Semester#}\Mini Project\Project\Object-Detection-using-Faster-R-CNN\images\test\Home.jpg"
]

test_annotations = [
    {
    'boxes': torch.tensor([
        [346, 83, 495, 373],
        [249, 96, 364, 374],
        [1, 94, 150, 369],
        [137, 94, 260, 376],
        [101, 133, 128, 175]
    ]),
    'labels': torch.tensor([1, 1, 1, 1, 1])
    },
    {
    'boxes': torch.tensor([
        [3, 2398, 490, 3073],
        [3060, 1765, 3841, 3072],
        [1679, 1128, 2441, 1910],
        [651, 928, 1516, 1820],
        [2401, 647, 3129, 1901],
        [1945, 169, 3135, 1890],
        [2122, 1704, 2322, 1913],
        [2513, 1707, 2724, 1914],
        [1978, 1296, 2485, 1925]
    ]),
    'labels': torch.tensor([64, 64, 64, 64, 64, 64, 64, 64, 64])
    }
]

def get_predictions(pred, threshold=0.8, objects=None):
    predicted_classes = [
        (COCO_INSTANCE_CATEGORY_NAMES[i], p, [(box[0], box[1]), (box[2], box[3])])
        for i, p, box in zip(
            list(pred[0]['labels'].numpy()),
            pred[0]['scores'].detach().numpy(),
            list(pred[0]['boxes'].detach().numpy())
        )
    ]
    predicted_classes = [stuff for stuff in predicted_classes if stuff[1] > threshold]
    if objects and predicted_classes:
        predicted_classes = [(name, p, box) for name, p, box in predicted_classes if name in objects]
    return predicted_classes

def draw_box(predicted_classes, image, rect_th=3, text_size=1, text_th=2):
    img = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
    for predicted_class in predicted_classes:
        label = predicted_class[0]
        probability = predicted_class[1]
        box = predicted_class[2]
        pt1 = (int(box[0][0]), int(box[0][1]))
        pt2 = (int(box[1][0]), int(box[1][1]))
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), rect_th)
        cv2.putText(img, f"{label}: {round(probability, 2)}", pt1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def detect_objects_in_image(image_path, model, threshold=0.8, show=True):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(image)
    with torch.no_grad():
        pred = model([img])
    pred_classes = get_predictions(pred, threshold)
    if show:
        draw_box(pred_classes, img)
    return pred[0]

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluate_model(model, test_images, test_annotations, iou_threshold=0.5, score_threshold=0.8):
    model.eval()
    transform = transforms.Compose([transforms.ToTensor()])
    correct = 0
    total = 0

    for img_path, gt in zip(test_images, test_annotations):
        image = Image.open(img_path).convert("RGB")
        img = transform(image)
        with torch.no_grad():
            pred = model([img])

        pred_boxes = pred[0]['boxes'][pred[0]['scores'] > score_threshold].cpu()
        pred_labels = pred[0]['labels'][pred[0]['scores'] > score_threshold].cpu()
        gt_boxes = gt['boxes']
        gt_labels = gt['labels']

        matched = set()
        for i, gt_box in enumerate(gt_boxes):
            for j, pred_box in enumerate(pred_boxes):
                iou = calculate_iou(gt_box.numpy(), pred_box.numpy())
                if iou >= iou_threshold and gt_labels[i] == pred_labels[j] and j not in matched:
                    correct += 1
                    matched.add(j)
                    break
            total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Detection Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    model.load_state_dict(torch.load('object_detection_model.pth', map_location='cpu'))
    model.eval()

    for img_path in test_images:
        print(f"\nPredictions for {img_path}:")
        detect_objects_in_image(img_path, model, threshold=0.5, show=True)

    print("\nEvaluating accuracy on test images...")
    evaluate_model(model, test_images, test_annotations, iou_threshold=0.5, score_threshold=0.5)
