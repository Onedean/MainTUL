import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class EarlyStopping:
    """[Early stops the training if validation loss doesn't improve after a given patience.]
    """

    def __init__(self, logger, dataset_name, patience=5, verbose=False, delta=0):
        """[Receive optional parameters]

        Args:
            patience (int, optional): [How long to wait after last time validation loss improved.]. Defaults to 7.
            verbose (bool, optional): [If True, prints a message for each validation loss improvement. ]. Defaults to False.
            delta (int, optional): [Minimum change in the monitored quantity to qualify as an improvement.]. Defaults to 0.
        """
        self.logger = logger
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset_name = dataset_name

    def __call__(self, val_loss, model):
        """[this is a Callback function]

        Args:
            val_loss ([float]): [The loss of receiving verification was changed to accuracy as the stop criterion in our experiment]
            model (Object): [model waiting to be saved]
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        """[Saves model when validation loss decrease.]

        Args:
            val_loss ([type]): [The loss value corresponding to the best checkpoint needs to be saved]
            model (Object): [Save the model corresponding to the best checkpoint]
        """
        if self.verbose:
            self.logger.info(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        torch.save(model.state_dict(), './project/temp/'+ self.dataset_name + '_checkpoint.pt')
        self.val_loss_min = val_loss


def accuracy_1(pred, targ):
    """[Used to calculate trajectory links acc@1]

    Args:
        pred ([torch.tensor]): [Predicted user probability distribution]
        targ ([type]): [The real label of the user corresponding to the trajectory]

    Returns:
        [float]: [acc@1]
    """
    pred = torch.max(torch.log_softmax(pred, dim=1), 1)[1]
    #ac = ((pred == targ).float()).sum().item() / targ.size()[0]
    ac = ((pred == targ)).float()
    return ac


def accuracy_5(pred, targ):
    """[Used to calculate trajectory links acc@5]

    Args:
        pred ([torch.tensor]): [Predicted user probability distribution]
        targ ([type]): [The real label of the user corresponding to the trajectory]

    Returns:
        [float]: [acc@5]
    """
    pred = torch.topk(torch.log_softmax(pred, dim=1), k=5, dim=1, largest=True, sorted=True)[1]
    #ac = (torch.tensor([t in p for p, t in zip(pred, targ)]).float()).sum().item() / targ.size()[0]
    ac = torch.tensor([t in p for p, t in zip(pred, targ)]).float()
    return ac


def loss_with_plot(avg_train_losses, avg_valid_losses, dataset_name):
    """[Function used to plot the loss curve and early stop line]

    Args:
        train_loss ([list]): [Loss list of training sets]
        val_loss ([list]): [Loss list of Validation sets]
    """
    # visualize the loss as the network trained
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(avg_train_losses)+1),
             avg_train_losses, label='Training Loss')
    plt.plot(range(1, len(avg_valid_losses)+1),
             avg_valid_losses, label='Validation Loss')

    # find position of lowest validation loss
    minposs = avg_valid_losses.index(min(avg_valid_losses))+1
    plt.axvline(minposs, linestyle='--', color='r',
                label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')

    # plt.ylim(0, 10) # consistent scale
    plt.xlim(0, len(avg_train_losses)+1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./project/log/' + dataset_name + 'early_stop_loss.png')