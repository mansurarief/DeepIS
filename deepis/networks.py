from typing import List

import numpy as np

from .exceptions import InputDimensionError
from .exceptions import WeightDimensionError

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device="cpu"
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class NeuralNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        '''Neural net architecture to learn rare/non-rare event boundaries

        Args:
            input_dim (int): input dimension
            hidden_dim (int): number of hidden nodes
            output_dim (int): output dimension
        '''
        super(NeuralNet, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Variable) -> Variable:
        '''Forward pass

        Args:
            x (Variable): input feature, size [N, D]

        Returns:
            _type_: logit score, size [N, C]
        '''
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def extract_params(self):
        W = {}
        i = 0
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                A = layer.state_dict()['weight'].cpu().numpy()
                A_dict = {str(i): {str(j): A[i][j] for j in range(A[i].shape[0])} for i in range(A.shape[0])}
                b = layer.state_dict()['bias'].cpu().numpy()
                b_dict = {str(i): b[i] for i in range(b.shape[0])}
                W[str(i)] = {'weight': A_dict, 'bias': b_dict}
                i += 1
        return W


def train_neural_network(train_input: np.ndarray,
                         train_label: np.ndarray,
                         batch_size: int = 500,
                         n_iters: int = 1000,
                         lr: float = 5e-3,
                         log: bool = False,
                         save: bool = False,
                         path: str = 'nets/net.pth',
                         it_interval: int = 100,
                         input_dim: int = 2,
                         hidden_dim: int = 10,
                         output_dim: int = 2,
                         weights: List = [1., 1.]) -> NeuralNet:
    '''Trainer for NeuralNet class with standard settings

    Args:
        train_input (np.ndarray): training input, size [N, D]
        train_label (np.ndarray): training label, size [N, 1]
        batch_size (SupportsInt, optional): _description_. Defaults to 500.
        n_iters (SupportsInt, optional): _description_. Defaults to 1000.
        lr (float, optional): learning rate.
        log (bool, optional): _description_. Defaults to False.
        save (bool, optional): _description_. Defaults to False.
        path (str, optional): _description_. Defaults to 'nets/net.pth'.
        it_interval (int, optional): _description_. Defaults to 100.
        input_dim (int, optional): input dimensions. Defaults to 2.
        hidden_dim (int, optional): number of hidden nodes. Defaults to 5.
        output_dim (int, optional): output dimension. Defaults to 2.
        weights (list, optional): class weights. Defaults to [1., 1.].

    Raises:
        Exception: weights

    Returns:
        NeuralNet: _description_
    '''

    N, D = train_input.shape

    if D != input_dim:
        raise InputDimensionError

    if len(weights) != output_dim:
        raise WeightDimensionError

    train_dataset = []
    for i in range(N):
        train_dataset.append([train_input[i, :], train_label[i]])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)

    input_dim = input_dim
    hidden_dim = hidden_dim
    output_dim = output_dim

    num_epochs = n_iters / (len(train_dataset) / batch_size)
    num_epochs = int(num_epochs)

    net = NeuralNet(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    class_weight = torch.Tensor(weights).to(device)
    sample_weight = torch.ones(batch_size).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weight, reduction='none')

    it = 1
    if log:
        print("Logging set as true:")

    for epoch in range(num_epochs):
        for i, (data, target) in enumerate(train_loader):
            it = it + 1
            X = Variable(torch.Tensor(data.float())).to(device)
            y = Variable(torch.Tensor(target.float())).to(device)
            optimizer.zero_grad()
            net_output = net(X)
            loss = criterion(net_output, y.long())
            loss = loss * sample_weight
            loss.mean().backward()
            optimizer.step()

            if log:
                if it % it_interval == 0:
                    pred = net_output.max(1)[1].data
                    label = y.long()
                    false_pos_rate = (torch.logical_and(pred == 1,
                                                        label == 0)).float().mean()
                    false_neg_rate = (torch.logical_and(pred == 0,
                                                        label == 1)).float().mean()
                    print("It: {}, Loss: {loss:0.4f}, False Pos Rate: {fp:0.3f}, False Neg Rate: {fn:0.3f}".format(
                        it,
                        loss=loss.mean().data,
                        fp=false_pos_rate.data,
                        fn=false_neg_rate.data),
                        end="\r")

    if save:
        torch.save(net.state_dict(), path)
        print(f'network saved at: {path}')

    return net.eval()
