import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import OPS, MLP
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import numpy as np


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            self._ops.append(op)
        #这里是一个连接的所有10个操作

    #这里的weights是选择模型的权重，而非图的权重。
    def forward(self, x, edge_index, weights, selected_idx=None):
        if selected_idx is None:  #向量alpha经过softmax， 乘以 卷积之后的输出。
            return sum(w * op(x, edge_index) for w, op in zip(weights, self._ops))
        else:  # unchosen operations are pruned    这一个就是SGAS的贪婪思想。
            return self._ops[selected_idx](x, edge_index) ##返回某一些选中的operations


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C):#96 96 32
        super(Cell, self).__init__()
        self.preprocess0 = MLP([C_prev_prev, C], 'relu', 'batch', bias=False)
        self.preprocess1 = MLP([C_prev, C], 'relu', 'batch', bias=False)
        self._steps = steps
        self._multiplier = multiplier
        #这里self._ops是一个cell的所有连接。
        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):  #和初始化alpha时一样的操作
                stride = 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, edge_index, weights, selected_idxs=None):
        s0 = self.preprocess0(s0)  #经过一个MLP把维度从96 变回32
        s1 = self.preprocess1(s1)

        states = [s0, s1]  #存储cell中的所有单元。
        offset = 0
        ##搭建一个cell
        for i in range(self._steps):
            o_list = []
            for j, h in enumerate(states):
                if selected_idxs[offset + j] == -1: # undecided mix edges
                    o = self._ops[offset + j](h, edge_index, weights[offset + j])  # call the gcn module
                elif selected_idxs[offset + j] == PRIMITIVES.index('none'): # pruned edges
                    pass
                else:  # decided discrete edges
                    o = self._ops[offset + j](h, edge_index, None, selected_idxs[offset + j])
                o_list.append(o)
            s = sum(o_list)
            offset += len(states)
            states.append(s)
        #把最后四个拼接起来，
        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):
    def __init__(self, C, num_classes, num_cells, criterion, steps=4, multiplier=4, stem_multiplier=3, in_channels=3):
        #C is init_channels
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._num_cells = num_cells
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier    ##
        self._in_channels = in_channels
        C_curr = stem_multiplier * C  #C_curr=96
        self.stem = nn.Sequential(
            MLP([in_channels, C_curr], None, 'batch', bias=False),
        )   ##一层MLP
        ##C=32，
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C  # 96, 96, 32
        self.cells = nn.ModuleList()
        for i in range(self._num_cells):
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr)
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr  #96,128
        # 之后几个cell 一直都是128
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(C_prev + 1, num_classes)  #最后一层

        self._initialize_alphas()

        #选择那个operation的索引？ 初始都为-1
        self.normal_selected_idxs = torch.tensor(len(self.alphas_normal) * [-1], requires_grad=False, dtype=torch.int)
        #要靠这个踢人吧？
        self.normal_candidate_flags = torch.tensor(len(self.alphas_normal) * [True],
                                                   requires_grad=False, dtype=torch.bool)

    def new(self):
        ##当unrolled=false 时用不到这个函数。
        model_new = Network(self._C, self._num_classes, self._num_cells, self._criterion, self._steps,
                            in_channels=self._in_channels).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)  #Copies the elements from src into self tensor and returns self.
        model_new.normal_selected_idxs = self.normal_selected_idxs
        return model_new

    def forward(self, x,edge_index):
        s0 = s1 = self.stem(x)   #经过一层MLP
        for i, cell in enumerate(self.cells):
            weights = []
            n = 2
            start = 0
            for _ in range(self._steps):    #这里进行softmax操作，这里的两个for和初始alpha的不一样。
                end = start + n
                for j in range(start, end):
                    weights.append(F.softmax(self.alphas_normal[j], dim=-1))
                start = end
                n += 1

            selected_idxs = self.normal_selected_idxs
            s0, s1 = s1, cell(s0, s1, edge_index, weights, selected_idxs)
        out = self.global_pooling(s1.unsqueeze(0)).squeeze(0)
        logits = self.classifier(torch.cat((out, s1), dim=1))
        return logits

    def _loss(self, input,edge_index, target,mask_testOld):
        logits = self(input,edge_index)  #这里过模型得到输出
        return self._criterion(logits[mask_testOld], target[mask_testOld])

    def _initialize_alphas(self):
        num_ops = len(PRIMITIVES)  #10
        self.alphas_normal = []
        for i in range(self._steps):  ##'total number of layers in one cell'
            for n in range(2 + i):      # 2，3，4 ？ 1，2，3？
                self.alphas_normal.append(Variable(1e-3 * torch.randn(num_ops).cuda(), requires_grad=True))
        self._arch_parameters = [
            self.alphas_normal
        ]    #这里还套一层是为啥？

    def arch_parameters(self):
        return self.alphas_normal

    def check_edges(self, flags, selected_idxs):
        #这个应该是sags提出的贪婪算法。
        n = 2
        max_num_edges = 2
        start = 0
        for i in range(self._steps):
            end = start + n
            num_selected_edges = torch.sum(1 - flags[start:end].int())
            if num_selected_edges >= max_num_edges:
                for j in range(start, end):
                    if flags[j]:
                        flags[j] = False
                        selected_idxs[j] = PRIMITIVES.index('none') # pruned edges！！！
                        self.alphas_normal[j].requires_grad = False
                    else:
                        pass
            start = end
            n += 1

        return flags, selected_idxs

    def parse_gene_force(self, flags, selected_idxs, alphas):
        gene = []
        n = 2
        max_num_edges = 2
        start = 0
        mat = F.softmax(torch.stack(alphas, dim=0), dim=-1).detach()
        importance = torch.sum(mat[:, 1:], dim=-1)
        masked_importance = torch.min(importance, (2 * flags.float() - 1) * np.inf)
        for _ in range(self._steps):
            end = start + n
            num_selected_edges = torch.sum(1 - flags[start:end].int())
            num_edges_to_select = max_num_edges - num_selected_edges
            if num_edges_to_select > 0:
                post_select_edges = torch.topk(masked_importance[start: end], k=num_edges_to_select).indices + start
            else:
                post_select_edges = []
            for j in range(start, end):
                if selected_idxs[j] == 0:
                    pass
                elif selected_idxs[j] == -1:
                    if num_edges_to_select <= 0:
                        raise Exception("Unknown errors")
                    else:
                        if j in post_select_edges:
                            idx = torch.argmax(alphas[j][1:]) + 1
                            gene.append((PRIMITIVES[idx], j - start))
                else:
                    gene.append((PRIMITIVES[selected_idxs[j]], j - start))
            start = end
            n += 1

        return gene


    def parse_gene(self, selected_idxs):
        gene = []
        n = 2
        start = 0
        for i in range(self._steps):
            end = start + n
            for j in range(start, end):
                if selected_idxs[j] == 0:
                    pass
                elif selected_idxs[j] == -1:
                    raise Exception("Contain undecided edges")
                else:
                    gene.append((PRIMITIVES[selected_idxs[j]], j - start))
            start = end
            n += 1

        return gene

    def get_genotype(self, force=False):
        if force:
            gene_normal = self.parse_gene_force(self.normal_candidate_flags,
                                                self.normal_selected_idxs,
                                                self.alphas_normal)

        else:
            gene_normal = self.parse_gene(self.normal_selected_idxs)
        n = 2
        concat = range(n + self._steps - self._multiplier, self._steps + n)
        genotype = Genotype(normal=gene_normal, normal_concat=concat)
        return genotype
