import torch.nn as nn
from sparse import GraphConv, MLP

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'skip_connect': lambda C, stride, affine: Identity(),
    'conv_1x1': lambda C, stride, affine: MLP([C, C], 'relu', 'batch', bias=False),
    'edge_conv': lambda C, stride, affine: GraphConv(C, C, 'edge', 'relu', 'batch', bias=False),
    'mr_conv': lambda C, stride, affine: GraphConv(C, C, 'mr', 'relu', 'batch', bias=False),
    'gat': lambda C, stride, affine: GraphConv(C, C, 'gat', 'relu', 'batch', bias=False, heads=1),
    'semi_gcn': lambda C, stride, affine: GraphConv(C, C, 'gcn', 'relu', 'batch', bias=False),
    'gin': lambda C, stride, affine: GraphConv(C, C, 'gin', 'relu', 'batch', bias=False),
    'sage': lambda C, stride, affine: GraphConv(C, C, 'sage', 'relu', 'batch', bias=False),
    'res_sage': lambda C, stride, affine: GraphConv(C, C, 'rsage', 'relu', 'batch', bias=False),
    'cheb': lambda C,stride,affine: GraphConv(C,C,'cheb','relu','batch',bias=False),
}
##函数名none等，参数  C, stride, affine，执行操作为GraphConv等。 类似于宏


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, edge_index=None):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x, edge_index=None):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)
