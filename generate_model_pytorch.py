import torchvision.models as models
import torch
model = models.vgg16(pretrained=True)
torch.save(model, 'vgg16_pytorch.pkl')

