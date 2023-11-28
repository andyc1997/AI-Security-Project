import numpy as np
import random
import time
import torch
import logging
import sys

import nets_mnist
from parser import parser
from utils import *
from datetime import datetime
from torch import optim, autograd
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


# logger
file_handler = logging.FileHandler(filename='model_VAE_{ct}.log'.format(ct=time.strftime('%Y%m%d-%H%M')))
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=handlers)
# parser
parser = parser(model='VAE')
args = parser.parse_args()
logging.info(args.__str__())

# load data
ct = time.strftime('%Y%m%d-%H%M')
data = np.load(r''+args.path)
data_img = data.get(data.files[0]) / 255.0
N = data_img.shape[0]
p = data_img.reshape(N, -1).shape[1]
li_loss = []
logging.info('*'*30 + '\nModel: VAE\n' + '*'*30)

# fix seed
seed = args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# directory for saving
folder = './{dataset}/N-{N}/score/{model}'.format(dataset=args.dataset, N=N, model='VAE')
dirs = CreateDirs(ct=ct, folder=folder, model='VAE')
dirs.create()

# preprocessing
def preprocessing(X: np.ndarray):
    X = torch.Tensor(X).type(torch.float32)
    X = torch.Tensor(X).type(torch.float32)
    return X.view(X.size(0), args.n_channels, args.n_width, args.n_height)
data_img = preprocessing(data_img)
p_input = args.n_channels * args.n_width * args.n_height
X = DataLoader(TensorDataset(data_img), batch_size=args.batch_size, shuffle=True)

# device: gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f'Using {device} device')

# encoder-decoder networks
encoder = nets_mnist.Encoder(p_latent=args.h_dim).to(device)
decoder = nets_mnist.Decoder(p_latent=args.h_dim).to(device)

# Adam optimizer
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=args.lr, weight_decay=0)

# VAE loss
def vae_loss(phi, X):
    mu, log_sig2 = phi[:, :args.h_dim], phi[:, args.h_dim:]
    eps = torch.randn(phi.shape[0], args.h_dim).to(device)
    z = mu + eps * torch.exp(0.5*log_sig2) # reparametrization trick
    x_tilde = decoder(z)

    # calculate loss
    recon_loss = set_loss_func(args.custom_loss, args.reduction_type)
    f1 = torch.mean(torch.sum(0.5 * (- (log_sig2 + 1) + mu ** 2 + torch.exp(log_sig2)), dim=1))
    f2 = recon_loss(x_tilde.view(-1, p_input), X.view(-1, p_input))/X.shape[0]  # reconstruction
    loss = f1 + f2
    return loss

# Initialize
l_cost = np.inf
cost = np.inf
start = datetime.now()
loss_holder = []
epoch = 0

# train
while epoch < args.max_epoch and cost > args.t_cost:
    avg_loss = 0.0
    for i, x in enumerate(X, 0):
        with autograd.set_detect_anomaly(True):
            if i < np.floor(N / args.batch_size):
                try:
                    x = x[0].to(device)
                    phi = encoder(x)
                    loss = vae_loss(phi, x)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                except Exception as e:
                    logging.error(e, exc_info=True)
                    loss = torch.inf * torch.ones(1, device=device)

                avg_loss += loss.detach().cpu().numpy()
            else:
                break

    # epoch logging
    logging.info('epoch: {epoch}'.format(epoch=epoch) +
                 '\t avg. loss: {:10.4f}'.format(float(avg_loss)))
    loss_holder.append(avg_loss)
    cost = avg_loss
    epoch += 1


    # Remember lowest cost and save checkpoint
    is_best = cost < l_cost
    l_cost = min(cost, l_cost)
    dirs.save_checkpoint({
        'epochs': epoch + 1,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'l_cost': l_cost,
        'optimizer': optimizer.state_dict(),
    }, is_best)

logging.info('VAE finished Training. Lowest cost: {:10.4f}'.format(l_cost))

# load from existing savepoints
if os.path.exists(folder+'/cp/{}'.format(dirs.dircp)):
    sd_mdl = torch.load(folder+'/cp/{}'.format(dirs.dircp))
    encoder.load_state_dict(sd_mdl['encoder_state_dict'])
    decoder.load_state_dict(sd_mdl['decoder_state_dict'])

# final computation
encoder = encoder.cpu()
decoder = decoder.cpu()
device = 'cpu'
phi = encoder(torch.Tensor(data_img)).detach()
h = phi[:, :args.h_dim]

# save all model parameters
torch.save({'args': args,
            'encoder': encoder,
            'decoder': decoder,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'h': h.cpu(),
            'phi': phi.cpu(),
            'loss': loss_holder},
           folder+'/out/{}'.format(dirs.dirout))

# garbage recycle
import gc
del encoder
del decoder
torch.cuda.empty_cache()
gc.collect()

logging.info('Final computation finished.')


