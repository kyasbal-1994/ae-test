import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from Autoencoder import Autoencoder


train, test = chainer.datasets.get_mnist()
train = train[0:1000]
test = test[47:63]
#plot_mnist_data(test)
N_EPOCH = 100
train = [i[0] for i in train]
train = tuple_dataset.TupleDataset(train, train)
train_iter = chainer.iterators.SerialIterator(train, 100)

model = L.Classifier(Autoencoder(), lossfun=F.mean_squared_error)
model.compute_accuracy = False
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

updater = training.StandardUpdater(train_iter, optimizer, device=-1)
trainer = training.Trainer(updater, (N_EPOCH, 'epoch'), out="result")
trainer.extend(extensions.LogReport(), trigger=(1, 'epoch'))
trainer.extend(extensions.PrintReport(
    entries=['epoch', 'main/loss', 'main/accuracy', 'elapsed_time']), trigger=(1, 'epoch'))
trainer.extend(extensions.ProgressBar())
trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))
trainer.run()
