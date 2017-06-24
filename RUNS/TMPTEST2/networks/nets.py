import torch.nn as nn


# FRAMEWORK_RLNET: RLNet-v0
class RLNet(nn.Module):
    """
    NB Each component is responsible for itself on cuda or no cuda (some
    needs to play with some data for self inspection -- e.g. convolutional
    layers only know the dimension of the input at runtime.).

    Cuda status must be consistent among all components.
    """

    def __init__(self, enc, dec):
        """
        :param enc: Feature extractor. See Encoder.
        :type enc: Encoder
        :param dec: Task target predictor
        :type dec: Decoder
        """
        super(RLNet, self).__init__()
        assert enc.get_feature_num() == dec.input_num
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        y = self.enc(x)
        y = y.view(-1, self.enc.get_feature_num())
        y = self.dec(y)
        return y
