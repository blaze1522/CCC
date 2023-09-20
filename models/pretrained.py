import torch
import torch.nn as nn


from models.registery import AdaptiveModel, register
from robustbench.model_zoo.architectures.utils_architectures import normalize_model, ImageNormalizer

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

@register("pretrained")
class Pretrained(AdaptiveModel):
    def __init__(self, model):
        super().__init__(model)
        self.model = model
        self.model.eval()
        self.encoder, self.classifier = split_up_model(self.model) 

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
            outputs = self.classifier(features)
            # outputs = self.model(x)
        return outputs

def split_up_model(model):
    """
    Split up the model into an encoder and a classifier.
    :param model: model to be split up
    :return: encoder and classifier
    """
    normalization = ImageNormalizer(mean=MEAN, std=STD).cuda()
    encoder = nn.Sequential(normalization, nn.Sequential(*list(model.module.children())[:-1]), nn.Flatten())
    classifier = model.module.fc

    return encoder, classifier