import os
import torch


class CreateDirs:
    """
    Creates directories for Checkpoints and saving trained models
    Source: https://github.com/MrPandey01/Stiefel_Restricted_Kernel_Machine/blob/main/code/utils.py
    """
    def __init__(self, ct, folder, model):
        self.ct = ct  # checkpoint time
        self.folder = folder
        self.dircp = 'checkpoint.pth_{}.tar'.format(self.ct)
        self.dirout = 'Mul_trained_{}_{}.tar'.format(model, self.ct)

    def create(self):
        # folder for checkpoints
        if not os.path.exists(self.folder+'/cp/'):
            os.makedirs(self.folder+'/cp/')
        # folder for model
        if not os.path.exists(self.folder+'/out/'):
            os.makedirs(self.folder+'/out/')

    def save_checkpoint(self, state, is_best):
        if is_best:
            torch.save(state, self.folder+'/cp/{}'.format(self.dircp))


def set_loss_func(loss_type: str = 'MSE', reduction_type: str = 'mean'):
    """
    Return a pytorch loss function.

    :param str loss_type: Type of the loss function for reconstruction, either MSE or BCE
    :param str reduction_type: Type of reduction, either mean or sum
    """
    if loss_type == 'MSE':
        return torch.nn.MSELoss(reduction=reduction_type)
    elif loss_type == 'BCE':
        return torch.nn.BCELoss(reduction=reduction_type)
    raise Exception(f'Invalid input for loss_type or reduction_type: {loss_type} | {reduction_type}')