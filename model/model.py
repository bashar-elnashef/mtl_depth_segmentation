import torch 
import torch.nn as nn
import model.decoder as decode
import model.encoder as encode
from torchvision.models import resnet50, resnet101, resnet18, resnet34, ResNet101_Weights

def hydranet(encoder=None, decoder=None, dec_cls=None):
    """Create a model instance from decoder and encoder"""
    encoder = getattr(encode, encoder)()
    encoder.load_state_dict(torch.load("data/mobilenetv2-e6e8dd43.pth"))

    encoder = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)

    num_classes = (dec_cls, 1)
    decoder = getattr(decode, decoder)([encoder.fc.in_features, encoder.fc.out_features], num_classes)

    return nn.DataParallel(nn.Sequential(encoder, decoder).cuda()), encoder, decoder



