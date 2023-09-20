import os
import torch 
import torch.nn.functional as F
import wandb

wandb.init(project="ccc", entity="sfda", save_code=True)


ROOT = 'data/src_features'
feature_fnames = sorted(os.listdir(ROOT))
print("No of batches", len(feature_fnames))
sims = []

for i in range(1, len(feature_fnames)):
    prev = torch.load('/'.join((ROOT,feature_fnames[i-1]))).mean(dim=0)
    curr = torch.load('/'.join((ROOT,feature_fnames[i]))).mean(dim=0)
    sim = F.cosine_similarity(prev, curr, dim=0)
    wandb.log({"sim":sim})
    
