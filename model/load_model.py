import torch
import torchvision.models as models

def load_model(path='model/resnet_genre.pth', num_classes=10):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model
