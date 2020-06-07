import glob
import torch
import tools.utils as utils
import logging
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
from model import NetworkPPI as Network
from tools.data_utils import *
import genotypes
from torch_geometric.data import Data,ClusterData, ClusterLoader, NeighborSampler
parser = argparse.ArgumentParser("evaluate")
parser.add_argument('--classes', type=int, default=15, help='the classes of labels')
parser.add_argument('--phase', type=str, default='train', help='train/test')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=128, help='num of init and hidden channels')
parser.add_argument('--num_cells', type=int, default=3, help='total number of cells')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='weights', help='experiment name')
parser.add_argument('--seed', type=int, default=11, help='random seed')
parser.add_argument('--arch', type=str, default='Genotype_xu4', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--in_channels', default=41, type=int, help='the channel of feature')
args = parser.parse_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    best_OA = 0.
    for epoch in range(args.epochs):
        #scheduler.get_lr()[0]
        print('epoch {} '.format(epoch))
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        Loss ,OA_train, OA_test= train( model, criterion, optimizer)
        f1,OA_test2,loss_test ,CLASS= test( model, criterion)

        print("Loss_train:{:.4f} Loss_test:{:.4f} OA_trian:{:.4f} OA_test:{:.4f}\
         F1:{:.4f}  OA_test:{:.4f}  best_OA:{:.4f}".format(Loss,loss_test,OA_train,OA_test,f1,OA_test2, best_OA))

        if OA_test2 > best_OA:
            best_OA = OA_test2
        print("each classes:",end='')
        for i in range(args.classes):
            print("{:.4f} ".format(CLASS[i]),end='')
        print("\n")
        #utils.save(model, os.path.join(args.save, 'evaluate_weights.pt'))
        scheduler.step()


def train( model, criterion, optimizer):
    model.train()
    total_loss = total_nodes = total_nodes_test = 0
    OA_train = OA_test = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits, logits_aux = model(batch.x, batch.edge_index)

        loss = criterion(logits[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        nodes = batch.train_mask.sum().item()
        total_loss += loss.item() * nodes
        total_nodes += nodes

        pred = logits[batch.train_mask].max(1)[1]
        OA_train += pred.eq(batch.y[batch.train_mask].max(1)[1]).sum().item()

        total_nodes_test += batch.test_mask.sum().item()
        pred_test = logits[batch.test_mask].max(1)[1]
        OA_test += pred_test.eq(batch.y[batch.test_mask].max(1)[1]).sum().item()

    return total_loss / total_nodes,OA_train/total_nodes  ,OA_test/total_nodes_test


def test( model, criterion):
    model.eval()
    micro_f1 = 0.
    CLASSES= []

    with torch.no_grad():
        total_loss_test = total_nodes_test = 0
        OA_test = 0
        for batch in train_loader:
            batch = batch.to(device)
            logits, logits_aux = model(batch.x, batch.edge_index)

            nodes = batch.test_mask.sum().item()
            total_nodes_test += nodes

            loss_test = criterion(logits[batch.test_mask], batch.y[batch.test_mask])
            total_loss_test += loss_test *nodes
            total_loss_test += batch.test_mask.sum().item()

            pred_test = logits[batch.test_mask].max(1)[1]
            OA_test += pred_test.eq(batch.y[batch.test_mask].max(1)[1]).sum().item()
            micro_f1 += utils.mF1(logits[batch.test_mask], batch.y[batch.test_mask]) * nodes
            ###各个类别的准确率
            CLASS = []
            classes = pred_test.eq(batch.y[batch.test_mask].max(1)[1])
            for cl in range(args.classes):
                CLASS.append(classes[batch.y[batch.test_mask].max(1)[1]==cl].sum().item())
            CLASSES.append(CLASS)
        CLASSES = np.sum(np.array(CLASSES),axis=0,dtype=float)
        CLASSES         /= total_test_oneclass
        micro_f1        /= total_nodes_test
        total_loss_test /= total_nodes_test
        OA_test         /= total_nodes_test
    return micro_f1,OA_test,total_loss_test,CLASSES



if __name__ == '__main__':

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    adj, features, labels, mask_train,mask_test, \
    y_test_oneclass, mask_test_oneclass,  mask_train1, = load_data('/content/drive/My Drive/NAS-GCN-SAR/data')

    ###传入数据的一些处理
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    edge_index ,edge_weight = from_scipy_sparse_matrix(adj)
    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels).float()
    mask_test_oneclass = torch.from_numpy(np.array(mask_test_oneclass))
    y_test_oneclass = torch.from_numpy(np.array(y_test_oneclass))

    ##data loader
    data = Data(x=features,edge_index=edge_index,y=labels)
    data.train_mask = torch.from_numpy(mask_train)
    data.test_mask = torch.from_numpy(mask_test)
    data.mask_test_oneclass = mask_test_oneclass
    data.y_test_oneclass = y_test_oneclass
    total_test_oneclass = []
    for i in range(args.classes):
        total_test_oneclass.append(mask_test_oneclass[i].sum())

    cluster_data = ClusterData(data, num_parts=2000, recursive=False,save_dir='./data')
    train_loader = ClusterLoader(cluster_data, batch_size=150, shuffle=True,
                                 num_workers=12)
    subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1], batch_size=1024,
                                      shuffle=False,num_workers=12)

    ######！！！！这里选择结构
    genotype = eval("genotypes.%s" % args.arch)  #eval()执行一个字符串表达式，并返回表达式的值。
    model = Network(args.init_channels, args.classes, args.num_cells, genotype,
                    in_channels=args.in_channels)
    model = model.to(DEVICE)

    print("param size = {:.6f}MB".format(utils.count_parameters_in_MB(model)))

    criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate,momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    main()

