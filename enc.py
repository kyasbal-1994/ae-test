import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from Autoencoder import Autoencoder
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

N_EPOCH = 0
def plot_mnist_data(samples, suffix,l):
    for index, (data, label) in enumerate(samples):
        plt.subplot(4, 4, index + 1)
        plt.axis('off')
        plt.imshow(data.reshape(28, 28))

    plt.savefig("./lerp/epoch_" + str(l) + "_" + suffix + '.png')

train, test = chainer.datasets.get_mnist()

train = train[0:1000]
test = test[47:63]
#plot_mnist_data(test, "base")

train = [i[0] for i in train]
train = tuple_dataset.TupleDataset(train, train)
train_iter = chainer.iterators.SerialIterator(train, 100)
enc = Autoencoder()
model = L.Classifier(enc, lossfun=F.mean_squared_error)
model.compute_accuracy = False
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer, device=-1)
trainer = training.Trainer(updater, (N_EPOCH, 'epoch'), out="result")

N_EPOCH = 990
chainer.serializers.load_npz(f'./result/snapshot_iter_{N_EPOCH}', trainer)
enc_data = []
for (data, label) in test:
    pred_data = model.predictor(
        np.array([data]).astype(np.float32), "ENC").data
    enc_data.append((pred_data,label))
for ti in range(0, 100):
    t = ti / 100.0
    print(t)
    predictedData = []
    for i in range(0,len(enc_data)):
        next = i + 1
        if next == len(enc_data):
            next = 0
        p = enc_data[i][0]
        n = enc_data[next][0]
        c = (n - p) * t + p # 2つの特徴ベクトルを線形補間
        pred_data = model.predictor(
            np.array([c]).astype(np.float32), "DEC").data
        predictedData.append((pred_data, str(enc_data[i][1]) + "-" + str(enc_data[next][1])))
    plot_mnist_data(predictedData,"LERPED",ti)
    
