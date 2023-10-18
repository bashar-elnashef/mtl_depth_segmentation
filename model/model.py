import torch 
import torch.nn as nn
import model.decoder as decode
import model.encoder as encode

# TODO: combine the encoder, coder and model scripts into one (model.py). General stuff should be moved to a helper script for NN's.  
def hydranet(encoder=None, decoder=None, dec_cls=None):
    """Create a model instance from decoder and encoder"""
    encoder = getattr(encode, encoder)()
    encoder.load_state_dict(torch.load("data/mobilenetv2-e6e8dd43.pth"))
    num_classes = (dec_cls, 1)
    decoder = getattr(decode, decoder)(encoder._out_c, num_classes)

    return nn.DataParallel(nn.Sequential(encoder, decoder).cuda()), encoder, decoder



