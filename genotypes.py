from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat')
Genotype_normal = namedtuple('Genotype_normal', 'normal normal_concat')

PRIMITIVES = [
    'none',
    'skip_connect',
    'conv_1x1',
    'edge_conv',
    'mr_conv',
    'gat',
    'semi_gcn',
    'gin',
    # 'sage',
    # 'res_sage',
    'cheb',
]

# 后面的数字代表链接的节点。

genotype = Genotype(normal=[('edge_conv', 0), ('skip_connect', 1), ('edge_conv', 0), ('conv_1x1', 2)], normal_concat=range(0, 4))
genotype2 = Genotype(normal=[('semi_gcn', 0), ('skip_connect', 1), ('edge_conv', 0), ('gin', 1)], normal_concat=range(0, 4))
#1 2 的n_step都是2，性能很差。
genotype3 = Genotype(normal=[('cheb', 0), ('sage', 1), ('edge_conv', 0), ('cheb', 1), ('cheb', 2), ('edge_conv', 3)], normal_concat=range(1, 5))
genotype4 = Genotype(normal=[('semi_gcn', 0), ('semi_gcn', 1), ('mr_conv', 1), ('gin', 2), ('edge_conv', 0), ('edge_conv', 1)], normal_concat=range(1, 5))
#4的性能目前最好
genotype5 = Genotype(normal=[('gat', 0), ('gin', 1), ('edge_conv', 0), ('cheb', 1), ('cheb', 2), ('edge_conv', 3)], normal_concat=range(1, 5))
genotype6 = Genotype(normal=[('semi_gcn', 1), ('cheb', 0), ('edge_conv', 0), ('conv_1x1', 2), ('conv_1x1', 0), ('conv_1x1', 2)], normal_concat=range(1, 5))
#6是随机搜索来的，特别容易过拟合。
genotype7 = Genotype(normal=[('edge_conv', 0), ('edge_conv', 1), ('gat', 0), ('cheb', 2), ('edge_conv', 1), ('edge_conv', 2)], normal_concat=range(1, 5))

genotype8 = Genotype(normal=[('semi_gcn', 0), ('skip_connect', 1), ('semi_gcn', 0), ('edge_conv', 2), ('edge_conv', 1), ('edge_conv', 3), ('gat', 1), ('edge_conv', 4)], normal_concat=range(2, 6))

genotype9 = Genotype(normal=[('semi_gcn', 0), ('gin', 1), ('edge_conv', 0), ('semi_gcn', 1), ('conv_1x1', 2), \
                             ('edge_conv', 3), ('gat', 1), ('skip_connect', 3), ('edge_conv', 3), ('edge_conv', 5)], normal_concat=range(3, 7))
