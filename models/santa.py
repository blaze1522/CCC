import os
from copy import deepcopy
# import tqdm
import PIL
import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from numpy import random
from torchvision.transforms import ColorJitter, Compose, Lambda

from models.registery import AdaptiveModel, register
from models.functional import collect_params, configure_model

from robustbench.model_zoo.architectures.utils_architectures import normalize_model, ImageNormalizer

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# from methods.base import TTAMethod
# from models.model import split_up_model
# from augmentations.transforms_cotta import get_tta_transforms
# from time import time

@register("santa")
class SANTA(AdaptiveModel):
    def __init__(self, model, lr=0.01, momentum=0.9):
        super().__init__(model.cuda())

        self.lr = lr # 0.01
        self.momentum = momentum # 0.9
        
        model = configure_model(model)
        params, param_names = collect_params(model)
        
        self.optimizer = torch.optim.SGD(
            params,
            lr=self.lr,
            momentum=self.momentum,
            dampening=0.0,
            weight_decay=0.0,
            nesterov=True,
        )
        
        self.num_classes = 1000

        self.contrast_mode = 'all'
        self.temperature = 0.1
        self.base_temperature = self.temperature
        self.projection_dim = 128
        self.lambda_ce_trg = 1
        self.lambda_cont = 1

        self.tta_transform = get_tta_transforms()
        
        # Setup EMA model
        self.model_ema = self.copy_model(self.model)
        for param in self.model_ema.parameters():
            param.detach_()

        # split up the model
        self.feature_extractor, self.classifier = split_up_model(model)

        # define the prototype paths
        fname = "ckpt/prototypes/protos_imagenet_c.pth"
        # fname = "/mnt/c37d5099-ac2c-4992-847b-b73906e77df8/goirik/CCC/ckpt/prototypes/protos_imagenet_c.pth"

        # get source prototypes
        if os.path.exists(fname):
            print("Loading class-wise source prototypes...")
            self.prototypes_src = torch.load(fname)
        else:
            os.makedirs(proto_dir_path, exist_ok=True)
            features_src = torch.tensor([])
            labels_src = torch.tensor([])
            print("Extracting source prototypes...")
            with torch.no_grad():
                for x, y in tqdm.tqdm(self.src_loader):
                    tmp_features = self.feature_extractor(x.cuda())
                    features_src = torch.cat([features_src, tmp_features.cpu()], dim=0)
                    labels_src = torch.cat([labels_src, y], dim=0)
                    if len(features_src) > 100000:
                        break

            # create class-wise source prototypes
            self.prototypes_src = torch.tensor([])
            for i in range(self.num_classes):
                mask = labels_src == i
                self.prototypes_src = torch.cat([self.prototypes_src, features_src[mask].mean(dim=0, keepdim=True)], dim=0)

            torch.save(self.prototypes_src, fname)

        self.prototypes_src = self.prototypes_src.cuda().unsqueeze(1)
        self.prototype_labels_src = torch.arange(start=0, end=self.num_classes, step=1).cuda().long()

        # setup projector
        num_channels = self.prototypes_src.shape[-1]
        self.projector = nn.Sequential(nn.Linear(num_channels, self.projection_dim), nn.ReLU(),
                                        nn.Linear(self.projection_dim, self.projection_dim)).cuda()
        self.optimizer.add_param_group({'params': self.projector.parameters(), 'lr': self.optimizer.param_groups[0]["lr"]})

        # note: if the self.model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.model, self.model_ema, self.projector]

    # Integrated from: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    def contrastive_loss(self, features, labels=None, mask=None):
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).cuda()
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().cuda()
        else:
            mask = mask.float().cuda()

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = self.projector(contrast_feature)
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward(self, x):
        imgs_test = x

        self.optimizer.zero_grad()

        # forward original test data
        features_test = self.feature_extractor(imgs_test)
        outputs_test = self.classifier(features_test)

        # forward augmented test data
        features_aug_test = self.feature_extractor(self.tta_transform((imgs_test)))
        outputs_aug_test = self.classifier(features_aug_test)

        # forward original test data through the ema model
        outputs_ema = self.model_ema(imgs_test)

        with torch.no_grad():
            # dist[:, i] contains the distance from every source sample to one test sample
            dist = F.cosine_similarity(self.prototypes_src.repeat(1, features_test.shape[0], 1),
                                       features_test.unsqueeze(0).repeat(self.prototypes_src.shape[0], 1, 1), dim=-1)

            # for every test feature, get the nearest source prototype and derive the label
            _, indices = dist.topk(1, largest=True, dim=0)
            indices = indices.squeeze(0)

        # Note: Shape: [200, 3, 640]
        features = torch.cat([self.prototypes_src[indices],
                              features_test.unsqueeze(1),
                              features_aug_test.unsqueeze(1)], 
                              dim=1) # Note: Every sample has three latents associated with is original, augmentation and prototype


        loss_contrastive = self.contrastive_loss(features=features, labels=None)

        loss_entropy = symmetric_entropy(x=outputs_test, x_aug=outputs_aug_test, x_ema=outputs_ema).mean(0)
        loss_trg =  self.lambda_cont * loss_contrastive + self.lambda_ce_trg * loss_entropy
        
        loss_trg.backward()
        self.optimizer.step()
        
        return outputs_test

    @staticmethod
    def configure_model(model):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        model.train()
        # disable grad, to (re-)enable only what tent updates
        model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm):
                m.requires_grad_(True)

        return model
    @staticmethod
    def check_model(model):
        """Check model for compatability with tent."""
        is_training = model.training
        assert is_training, "tent needs train mode: call model.train()"
        param_grads = [p.requires_grad for p in model.parameters()]
        has_any_params = any(param_grads)
        has_all_params = all(param_grads)
        assert has_any_params, "tent needs params to update: " \
                            "check which require grad"
        assert not has_all_params, "tent should not update all params: " \
                                "check which require grad"
        has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
        assert has_bn, "tent needs normalization for its optimization"
    @staticmethod
    def copy_model(model):
        coppied_model = deepcopy(model)
        return coppied_model

@torch.jit.script
def symmetric_entropy(x, x_aug, x_ema):# -> torch.Tensor:
    return  - 0.5 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1) \
            - 0.5 * (x_aug.softmax(1) * x_ema.log_softmax(1)).sum(1)

@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)

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

class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        noise = noise.to(img.device)
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class Clip(torch.nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        return torch.clip(img, self.min_val, self.max_val)

    def __repr__(self):
        return self.__class__.__name__ + "(min_val={0}, max_val={1})".format(
            self.min_val, self.max_val
        )


class ColorJitterPro(ColorJitter):
    """Randomly change the brightness, contrast, saturation, and gamma correction of an image."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, gamma=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.gamma = self._check_input(gamma, "gamma")

    @staticmethod
    @torch.jit.unused
    def get_params(brightness, contrast, saturation, hue, gamma):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(
                Lambda(lambda img: F.adjust_brightness(img, brightness_factor))
            )

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(
                Lambda(lambda img: F.adjust_contrast(img, contrast_factor))
            )

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(
                Lambda(lambda img: F.adjust_saturation(img, saturation_factor))
            )

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        if gamma is not None:
            gamma_factor = random.uniform(gamma[0], gamma[1])
            transforms.append(Lambda(lambda img: F.adjust_gamma(img, gamma_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx = torch.randperm(5)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = (
                    torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                )
                img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = (
                    torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                )
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = (
                    torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                )
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F.adjust_hue(img, hue_factor)

            if fn_id == 4 and self.gamma is not None:
                gamma = self.gamma
                gamma_factor = torch.tensor(1.0).uniform_(gamma[0], gamma[1]).item()
                img = img.clamp(
                    1e-8, 1.0
                )  # to fix Nan values in gradients, which happens when applying gamma
                # after contrast
                img = F.adjust_gamma(img, gamma_factor)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "brightness={0}".format(self.brightness)
        format_string += ", contrast={0}".format(self.contrast)
        format_string += ", saturation={0}".format(self.saturation)
        format_string += ", hue={0})".format(self.hue)
        format_string += ", gamma={0})".format(self.gamma)
        return format_string


def get_tta_transforms(gaussian_std: float = 0.005, soft=False, clip_inputs=False):
    img_shape = (224, 224, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose(
        [
            Clip(0.0, 1.0),
            ColorJitterPro(
                brightness=[0.8, 1.2] if soft else [0.6, 1.4],
                contrast=[0.85, 1.15] if soft else [0.7, 1.3],
                saturation=[0.75, 1.25] if soft else [0.5, 1.5],
                hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
                gamma=[0.85, 1.15] if soft else [0.7, 1.3],
            ),
            transforms.Pad(padding=int(n_pixels / 2), padding_mode="edge"),
            transforms.RandomAffine(
                degrees=[-8, 8] if soft else [-15, 15],
                translate=(1 / 16, 1 / 16),
                scale=(0.95, 1.05) if soft else (0.9, 1.1),
                shear=None,
                interpolation=PIL.Image.BILINEAR,
                fill=None,
            ),
            transforms.GaussianBlur(
                kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]
            ),
            transforms.CenterCrop(size=n_pixels),
            transforms.RandomHorizontalFlip(p=p_hflip),
            GaussianNoise(0, gaussian_std),
            Clip(clip_min, clip_max),
        ]
    )
    return tta_transforms