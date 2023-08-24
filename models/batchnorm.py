import torch

from models.registery import AdaptiveModel, register
# from models import register
from models.functional import erase_bn_stats


@register("batchnorm")
class BNAdapt(AdaptiveModel):
    def __init__(self, model):
        super().__init__(model)
        model = erase_bn_stats(model)
        self.model = model
        self.model = model.train()

    def forward(self, x):
        with torch.no_grad():
            outputs = self.model(x)
        return outputs
