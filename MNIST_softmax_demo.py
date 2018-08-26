from torchvision.datasets import MNIST
from torchvision import transforms

import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

cuda = torch.cuda.is_available()

from trainer import fit
from networks import EmbeddingNet, ClassificationNet
from metrics import AccumulatedAccuracyMetric

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
# MNIST class:
# use DataLoader to invoke MNIST.__getitem__ method and MNIST can return (input, target) input is image, target is label.
# Returns:
#           tuple: (image, target) where target is index of the target class.


n_classes = 10

# set up data loaders
batch_size = 256
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

embedding_net = EmbeddingNet()
model = ClassificationNet(embedding_net, n_classes=n_classes)
if cuda:
    model.cuda()
loss_fn = torch.nn.NLLLoss()
lr = 1e-2
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 1
log_interval = 50

fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])