import torch
from torch import nn

batch_size = 1000
n_epochs = 5
dropout_rate = 0.1
learning_rate = 0.001
log_interval = 10
optimizerFunc = torch.optim.RMSprop
has_momentum = True
momentum = 0.1
lossFunc = nn.MSELoss()
saved_network_filename = "savednetwork.savednn"
