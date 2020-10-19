from .Decoder.unet_decoder import UnetDecoder
from .Encoder import get_encoder
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class EncoderDecoder(Model):

    def __init__(self, encoder, decoder, criterion=nn.BCEWithLogitsLoss()):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion

    def forward(self, x, y=None):
        x = self.encoder(x)
        x = self.decoder(x)

        if self.training:
            y = y.view(y.shape[0], y.shape[2], y.shape[3])

            main_loss = self.criterion(x, y.long())
            return x.max(1)[1], main_loss
        else:
            return x

class Unet(EncoderDecoder):

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            classes=1,
            center=False,
            criterion = nn.BCEWithLogitsLoss()
    ):
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        decoder = UnetDecoder(
            encoder_channels=encoder.out_shapes,
            decoder_channels=decoder_channels,
            final_channels=classes,
            use_batchnorm=decoder_use_batchnorm,
            center=center,
        )

        super().__init__(encoder, decoder,criterion)

        self.name = 'u-{}'.format(encoder_name)