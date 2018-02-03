import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset


class Autoencoder(chainer.Chain):
    def __init__(self):
        super(Autoencoder, self).__init__(
            encoder=L.Linear(784, 64),
            decoder=L.Linear(64, 784))

    def __call__(self, x, mode="ENCDEC"):
        if mode == "ENCDEC":
            h = F.relu(self.encoder(x))
            return F.relu(self.decoder(h))
        elif mode == "ENC":
            return F.relu(self.encoder(x))
        else:
            return F.relu(self.decoder(x))
