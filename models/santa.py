import os
import tqdm
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F

from methods.base import TTAMethod
from models.model import split_up_model
from augmentations.transforms_cotta import get_tta_transforms
from time import time


logger = logging.getLogger(__name__)


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

class RCONT(TTAMethod):
    def __init__(self, model, optimizer, steps, episodic, window_length, dataset_name, arch_name, num_classes, ckpt_dir, ckpt_path,
                 contrast_mode, temperature, projection_dim, lambda_ce_trg, lambda_cont, m_teacher_momentum):
        super().__init__(model.cuda(), optimizer, steps, episodic, window_length)

        self.dataset_name = dataset_name
        self.num_classes = num_classes

        self.contrast_mode = contrast_mode
        self.temperature = temperature
        self.base_temperature = self.temperature
        self.projection_dim = projection_dim
        self.lambda_ce_trg = lambda_ce_trg
        self.lambda_cont = lambda_cont
        self.m_teacher_momentum = m_teacher_momentum

        self.tta_transform = get_tta_transforms(self.dataset_name)

        # Setup EMA model
        self.model_ema = self.copy_model(self.model)
        for param in self.model_ema.parameters():
            param.detach_()

        # self.model_ema = anchor_model

        # split up the model
        self.feature_extractor, self.classifier = split_up_model(model, arch_name, self.dataset_name)

        # define the prototype paths
        proto_dir_path = os.path.join(ckpt_dir, "prototypes")
        if self.dataset_name == "domainnet126":
            fname = f"protos_{self.dataset_name}_{ckpt_path.split(os.sep)[-1].split('_')[1]}.pth"
        else:
            # fname = f"protos_{self.dataset_name}_{arch_name}.pth"
            fname = f"protos_{self.dataset_name}.pth"
        fname = os.path.join(proto_dir_path, fname)

        # get source prototypes
        if os.path.exists(fname):
            logger.info("Loading class-wise source prototypes...")
            self.prototypes_src = torch.load(fname)
        else:
            os.makedirs(proto_dir_path, exist_ok=True)
            features_src = torch.tensor([])
            labels_src = torch.tensor([])
            logger.info("Extracting source prototypes...")
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
        if self.dataset_name == "domainnet126":
            # do not use a projector since the network already clusters the features and reduces the dimensions
            self.projector = nn.Identity()
        else:
            num_channels = self.prototypes_src.shape[-1]
            self.projector = nn.Sequential(nn.Linear(num_channels, self.projection_dim), nn.ReLU(),
                                           nn.Linear(self.projection_dim, self.projection_dim)).cuda()
            self.optimizer.add_param_group({'params': self.projector.parameters(), 'lr': self.optimizer.param_groups[0]["lr"]})

        # note: if the self.model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.model, self.model_ema, self.projector]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

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
    def forward_and_adapt(self, x):
        t0 = time()
        imgs_test = x[0]

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
        
        t1 = time()
        loss_trg.backward()
        self.optimizer.step()
        t2 = time()

        # print("Time for Forward Pass:", t1-t0)
        # print("Time for Backward Pass:", t2-t1)

        # self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.m_teacher_momentum)

        return outputs_test

    @torch.no_grad()
    def forward_sliding_window(self, x):
        """
        Create the prediction for single sample test-time adaptation with a sliding window
        :param x: The buffered data created with a sliding window
        :return: Model predictions
        """
        imgs_test = x[0]
        outputs_test = self.model(imgs_test)
        outputs_ema = self.model_ema(imgs_test)
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

# - 0.25 * (x_ema.softmax(1) * x.log_softmax(1)).sum(1)
# - 0.25 * (x_ema.softmax(1) * x_aug.log_softmax(1)).sum(1)

@torch.jit.script
def symmetric_entropy(x, x_aug, x_ema):# -> torch.Tensor:
    return  - 0.5 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1) \
            - 0.5 * (x_aug.softmax(1) * x_ema.log_softmax(1)).sum(1)

@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)