import torch.utils.model_zoo as model_zoo
from .senet import senet_encoders
import numpy as np

def preprocess_input(x, mean=None, std=None, input_space='RGB', input_range=None, **kwargs):

    if input_space == 'BGR':
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    return x

encoders = {}
# encoders.update(vgg_encoders)
encoders.update(senet_encoders)



def get_encoder(name, encoder_weights=None):
    Encoder = encoders[name]['encoder']
    encoder = Encoder(**encoders[name]['params'])
    encoder.out_shapes = encoders[name]['out_shapes']

    if encoder_weights is not None:
        #print(model_zoo.load_url.items())
        settings = encoders[name]['pretrained_settings'][encoder_weights]
        encoder.load_state_dict(model_zoo.load_url(settings['url']))

    return encoder


def get_encoder_names():
    return list(encoders.keys())


def get_preprocessing_fn(encoder_name, pretrained='imagenet'):
    settings = encoders[encoder_name]['pretrained_settings']

    if pretrained not in settings.keys():
        raise ValueError('Avaliable pretrained options {}'.format(settings.keys()))

    input_space = settings[pretrained].get('input_space')
    input_range = settings[pretrained].get('input_range')
    mean = settings[pretrained].get('mean')
    std = settings[pretrained].get('std')

    def _preprocess_input(x, **kwargs):
        return preprocess_input(x, mean=mean, std=std, input_space=input_space, input_range=input_range, **kwargs)

    return _preprocess_input