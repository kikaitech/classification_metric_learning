import sys
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import faiss
from pretrainedmodels.utils import ToRange255
from pretrainedmodels.utils import ToSpaceBGR

from data.wineeye import WineEye
import metric_learning.modules.featurizer as featurizer
from metric_learning.extract_features import extract_feature

gpu_device = torch.device('cuda')

model_factory = getattr(featurizer, 'resnet50')
model = model_factory(512)
model.to(device=gpu_device)
model.load_state_dict(torch.load('/data1/output/WineEye/512/resnet50_30/epoch_5.pth'))
model.eval()

eval_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    ToSpaceBGR(model.input_space == 'BGR'),
    ToRange255(max(model.input_range) == 255),
    transforms.Normalize(mean=model.mean, std=model.std)
])

eval_dataset = WineEye('/data1/data/wineeye', train=False, transform=eval_transform)
# eval_loader = DataLoader(eval_dataset,
#                         drop_last=False,
#                         shuffle=False,
#                         pin_memory=True,
#                         num_workers=16)
# embeddings, _ = extract_feature(model, eval_loader, gpu_device)
# np.savez_compressed('embeddings.npz', embeddings)

embeddings = np.load('embeddings.npz')['arr_0']
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print "Ready!"
im_path = sys.stdin.readline()
while im_path:
    im_path = im_path.strip() + '.jpg'
    im = Image.open(im_path).convert('RGB')
    im = eval_transform(im)
    im = im.to(device=gpu_device, non_blocking=True)
    embedding = model(im.unsqueeze_(0))
    dists, retrieved_result_indices = index.search(embedding.cpu().detach().numpy(), 3)
    for idx in retrieved_result_indices[0]:
        os.system('xdg-open "' + eval_dataset.image_paths[idx] + '"')
    im_path = sys.stdin.readline()
    
    