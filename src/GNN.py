import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import (Adj)
from torch_geometric.utils import  softmax

class GNN(MessagePassing):
    def __init__(
        self,
        out_channels: int,
        edge_type_num = 1,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.out_channels = out_channels
        self.edge_type_num = edge_type_num
        self.w = Parameter(torch.zeros(size=(edge_type_num, self.out_channels*2)))
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 0.1
        torch.nn.init.normal_(self.w,std=stdv)

    def forward(self, x: Tensor, edge_index: Adj,edge_type=None,
                return_attention_weights=None):
        src_idx = edge_index[0]
        tar_idx = edge_index[1]

        src_x = x[src_idx]
        tar_x = x[tar_idx]


        weight = self.w[edge_type]
        e = F.leaky_relu((weight * torch.concat([src_x, tar_x],dim=-1)).sum(-1)) # attention score
        alpha = softmax(e, tar_idx)
        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor, edge_attr: OptTensor)  # noqa
        out = self.propagate(edge_index, x=x, alpha=alpha)
        return out

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return x_j * alpha.unsqueeze(1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}')
