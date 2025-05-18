import torch
import torch.nn as nn
import torchvision.models as models

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes_clarity, num_classes_color, num_classes_cut):
        super(MultiTaskModel, self).__init__()
        self.backbone = models.efficientnet_b3(pretrained=True)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.fc_clarity = nn.Linear(in_features, num_classes_clarity)
        self.fc_color = nn.Linear(in_features, num_classes_color)
        self.fc_cut = nn.Linear(in_features, num_classes_cut)
        self.fc_carat = nn.Linear(in_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        return {
            "clarity": self.fc_clarity(features),
            "color": self.fc_color(features),
            "cut": self.fc_cut(features),
            "carat": self.fc_carat(features)
        }
