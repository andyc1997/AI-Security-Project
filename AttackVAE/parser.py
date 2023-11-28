import argparse

def parser(model='VAE'):
    parser = argparse.ArgumentParser(prog='model:{model}'.format(model=model), description='Train {model} model.'.format(model=model))
    parser.add_argument('path', type=str, help='The path to the dataset.', metavar='')
    parser.add_argument('--dataset', type=str, help='Name for the dataset.')
    parser.add_argument('--p_feat', type=int, default=256, help='The dimension for feature space.', metavar='')
    parser.add_argument('--n_width', type=int, default=96, help='The dimension for width.', metavar='')
    parser.add_argument('--n_height', type=int, default=96, help='The dimension for height.', metavar='')
    parser.add_argument('--n_channels', type=int, default=1, help='The dimension for channels.', metavar='')
    parser.add_argument('--h_dim', type=int, default=5, help='The dimension for the latent space.', metavar='')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate.', metavar='')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.', metavar='')
    parser.add_argument('--max_epoch', type=int, default=1000, help='The maximum number of epochs.', metavar='')
    parser.add_argument('--l_cost', type=float, default=100., help='The cost start to save checkpoint.', metavar='')
    parser.add_argument('--t_cost', type=float, default=1., help='The cost to terminate the training process.', metavar='')
    parser.add_argument('--custom_loss', type=str, default='MSE', choices=['MSE', 'BCE'], help='The type of loss function.', metavar='')
    parser.add_argument('--reduction_type', type=str, default='sum', choices=['mean', 'sum'], help='The type of reduction for loss.', metavar='')
    parser.add_argument('--seed', type=int, help='Seed', metavar='')
    return parser