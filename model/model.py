import torch 
import torch.nn as nn
import model.decoder
import model.encoder

def hydranet(encoder=None, decoder=None, dec_cls=None):
    """Create a model instance from decoder and encoder"""
    _encoder = getattr(model.encoder, encoder)()
    _encoder.load_state_dict(torch.load("data/mobilenetv2-e6e8dd43.pth"))

    num_classes = (dec_cls, 1)
    _decoder = getattr(model.decoder, decoder)(_encoder._out_c, num_classes)

    return nn.DataParallel(nn.Sequential(_encoder, _decoder).cuda()), _encoder, _decoder



