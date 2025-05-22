import torch
import torchvision
from torchvision.transforms import transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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

def get_predictions(pred, threshold=0.8, objects=None):

    pred_boxes = pred[0]['boxes'].detach().cpu().numpy()
    pred_scores = pred[0]['scores'].detach().cpu().numpy()
    pred_labels = pred[0]['labels'].detach().cpu().numpy()
    results = []
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        if score >= threshold:
            results.append({
                'label': COCO_INSTANCE_CATEGORY_NAMES[label],
                'score': float(score),
                'box': [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
            })
    return results

def draw_box(predicted_classes, image, rect_th=3, text_size=1, text_th=2):
    img = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
    for pred in predicted_classes:
        label = pred['label']
        score = pred['score']
        box = pred['box']
        pt1 = (box[0], box[1])
        pt2 = (box[2], box[3])
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), rect_th)
        cv2.putText(img, f"{label}: {round(score, 2)}", pt1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def detect_objects_in_image(image_path, threshold=0.8):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    model.load_state_dict(torch.load('object_detection_model.pth', map_location='cpu'))
    model.eval()


    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(image)

    with torch.no_grad():
        pred = model([img])

    pred_classes = get_predictions(pred, threshold=threshold)

    print(f"\nDetected objects in {image_path}:")
    for idx, pred in enumerate(pred_classes):
        print(f"Object {idx+1}: {pred['label']} | Score: {pred['score']:.2f} | Box: {pred['box']}")

    draw_box(pred_classes, img)

    return pred_classes

if __name__ == "__main__":
    image_paths = [
        r"D:\College {BVM}\{#2nd Semester#}\Mini Project\Project\Object-Detection-using-Faster-R-CNN\images\test\Persons.jpeg",
        r"D:\College {BVM}\{#2nd Semester#}\Mini Project\Project\Object-Detection-using-Faster-R-CNN\images\test\Home.jpg"
    ]
    for image_path in image_paths:
        detect_objects_in_image(image_path, threshold=0.5)
