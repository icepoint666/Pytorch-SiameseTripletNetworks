from torchvision.datasets import MNIST
from torchvision import transforms

import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

cuda = torch.cuda.is_available()

from trainer import fit
from networks import EmbeddingNet, SiameseNet
from metrics import AccumulatedAccuracyMetric
from losses import ContrastiveLoss
from datasets import SiameseMNIST

mean, std = 0.1307, 0.3081

# prepare dataset
train_dataset = MNIST('../data/MNIST', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((mean,), (std,))
                             ]))
test_dataset = MNIST('../data/MNIST', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))
                            ]))
# MNIST:
# self.train_data[index] return the index(th) image of train data
# (type: torch.Tensor)
# (.size(): torch.Size([28,28]))
#
# self.train_labels[index] return the index(th) label of train data
# (type: torch.Tensor)
#
# test_dataset with the same way

# Output demo:
# print (type(train_dataset.train_data[5]))
# print (train_dataset.train_data[5].size())
# print (train_dataset.train_data[5])
# print (type(train_dataset.train_labels[5]))
# print (train_dataset.train_labels[5].size())
# print (train_dataset.train_labels[5])

siamese_train_dataset = SiameseMNIST(train_dataset)
siamese_test_dataset = SiameseMNIST(test_dataset)

# set up data loaders
batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

margin = 1
embedding_net = EmbeddingNet()
model = SiameseNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = ContrastiveLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 1
log_interval = 100

fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)