import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

def train_object_detection_model():

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    for name, param in model.named_parameters():
        param.requires_grad = False


    torch.save(model.state_dict(), 'object_detection_model.pth')

if __name__ == "__main__":
    train_object_detection_model()
