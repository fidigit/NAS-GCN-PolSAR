import os
import sys
import time
import glob
import math
import tools.utils as utils
import logging
import argparse
import torch.utils
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributions.categorical as cate
import torchvision.utils as vutils

from tools.data_utils import *
from model_search import Network
from architect import Architect
from tensorboardX import SummaryWriter
from torch_geometric.data import Data,ClusterData, ClusterLoader, NeighborSampler

# torch_geometric.set_debug(True)
parser = argparse.ArgumentParser("search")
parser.add_argument('--classes', type=int, default=15, help='the classes of labels')
# parser.add_argument('--batch_increase', default=1, type=int, help='how much does the batch size increase after making a decision')
parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=32, help='num of init channels')
parser.add_argument('--num_cells', type=int, default=1, help='total number of cells')
parser.add_argument('--n_steps', type=int, default=3, help='total number of layers in one cell')
# parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='weights', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')  #更改seed值 就能换不同的初始参数。
parser.add_argument('--random_seed', action='store_true', help='use seed randomly')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss') #？？
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--warmup_dec_epoch', type=int, default=9, help='warmup decision epoch')
parser.add_argument('--decision_freq', type=int, default=7, help='decision freq epoch')
parser.add_argument('--history_size', type=int, default=4, help='number of stored epoch scores')
parser.add_argument('--use_history', action='store_true', help='use history for decision')  #代表使用两种不同的策略。
parser.add_argument('--in_channels', default=41, type=int, help='the dimension of feature')

args = parser.parse_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##########################################################################
# args.save = 'log/search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
# utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py')) #glob.glob 返回所有匹配的文件路径列表。
#
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'search_log.txt')) # 建立一个filehandler来把日志记录在文件里
fh.setFormatter(logging.Formatter(log_format))  #括号里面设置日志格式，
logging.getLogger().addHandler(fh)
#########################################################################

writer = SummaryWriter(log_dir=args.save, max_queue=50)


def histogram_average(history, probs):
    histogram_inter = torch.zeros(probs.shape[0], dtype=torch.float).to(DEVICE)
    if not history:
        return histogram_inter
    for hist in history:
        histogram_inter += utils.histogram_intersection(hist, probs)
    histogram_inter /= len(history)
    return histogram_inter



def edge_decision(type, alphas, selected_idxs, candidate_flags, probs_history, epoch, model, args):
    #确定选择哪些边
    mat = F.softmax(torch.stack(alphas, dim=0), dim=-1).detach()
    print(mat)
    importance = torch.sum(mat[:, 1:], dim=-1)
    # logging.info(type + " importance {}".format(importance))

    probs = mat[:, 1:] / importance[:, None]
    # print(type + " probs", probs)
    entropy = cate.Categorical(probs=probs).entropy() / math.log(probs.size()[1])
    # logging.info(type + " entropy {}".format(entropy))

    if args.use_history:  # SGAS Cri.2
        # logging.info(type + " probs history {}".format(probs_history))
        histogram_inter = histogram_average(probs_history, probs)
        # logging.info(type + " histogram intersection average {}".format(histogram_inter))
        probs_history.append(probs)
        if (len(probs_history) > args.history_size):
            probs_history.pop(0)

        score = utils.normalize(importance) * utils.normalize(
            1 - entropy) * utils.normalize(histogram_inter)
        # logging.info(type + " score {}".format(score))
    else:  # SGAS Cri.1
        score = utils.normalize(importance) * utils.normalize(1 - entropy)
        # logging.info(type + " score {}".format(score))

    if torch.sum(candidate_flags.int()) > 0 and \
            epoch >= args.warmup_dec_epoch and \
            (epoch - args.warmup_dec_epoch) % args.decision_freq == 0:
        masked_score = torch.min(score,
                                 (2 * candidate_flags.float() - 1) * np.inf)
        selected_edge_idx = torch.argmax(masked_score)
        selected_op_idx = torch.argmax(probs[selected_edge_idx]) + 1  # add 1 since none op
        selected_idxs[selected_edge_idx] = selected_op_idx

        candidate_flags[selected_edge_idx] = False
        alphas[selected_edge_idx].requires_grad = False
        if type == 'normal':
            reduction = False
        elif type == 'reduce':
            reduction = True
        else:
            raise Exception('Unknown Cell Type')
        candidate_flags, selected_idxs = model.check_edges(candidate_flags,
                                                           selected_idxs)
        logging.info("#" * 30 + " Decision Epoch " + "#" * 30)
        logging.info("epoch {}, {}_selected_idxs {}, added edge {} with op idx {}".format(epoch,
                                                                                          type,
                                                                                          selected_idxs,
                                                                                          selected_edge_idx,
                                                                                          selected_op_idx))
        print(type + "_candidate_flags {}".format(candidate_flags))
        # score_image(type, score, epoch)
        return True, selected_idxs, candidate_flags

    else:
        logging.info("#" * 30 + " Not a Decision Epoch " + "#" * 30)
        logging.info("epoch {}, {}_selected_idxs {}".format(epoch,
                                                            type,
                                                            selected_idxs))
        print(type + "_candidate_flags {}".format(candidate_flags))
        # score_image(type, score, epoch)
        return False, selected_idxs, candidate_flags


def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)
    if args.random_seed:
        args.seed = np.random.randint(0, 1000, 1)
    # reproducible ，再次运行代码时，初始化值不变。
    #you should ensure that all other libraries your code relies on and which use random numbers also use a fixed seed.
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = torch.nn.BCEWithLogitsLoss().cuda()
    ## in_channels是特征维度 ！！
    model = Network(args.init_channels, args.classes, args.num_cells, criterion,
                    args.n_steps, in_channels=args.in_channels).cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    num_edges = model._steps * 2
    post_train = 5
    args.epochs = args.warmup_dec_epoch + args.decision_freq * (num_edges - 1) + post_train + 1
    logging.info("total epochs: %d", args.epochs)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    normal_selected_idxs = torch.tensor(len(model.alphas_normal) * [-1], requires_grad=False, dtype=torch.int).cuda()
    normal_candidate_flags = torch.tensor(len(model.alphas_normal) * [True], requires_grad=False, dtype=torch.bool).cuda()
    logging.info('normal_selected_idxs: {}'.format(normal_selected_idxs))
    logging.info('normal_candidate_flags: {}'.format(normal_candidate_flags))
    model.normal_selected_idxs = normal_selected_idxs
    model.normal_candidate_flags = normal_candidate_flags

    print(F.softmax(torch.stack(model.alphas_normal, dim=0), dim=-1).detach())

    normal_probs_history = []
    train_losses, valid_losses = utils.AverageMeter(), utils.AverageMeter()
    for epoch in range(args.epochs):
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        # training
        train_acc, train_loss = train( model, architect, criterion, optimizer, lr)
        print("!!!!!!!!!!!!!!!!train_loss:" , train_loss)
        valid_acc, valid_losses = infer( model, criterion, valid_losses)
        logging.info('train_acc %f\tvalid_acc %f', train_acc, valid_acc)

        # make edge decisions
        saved_memory_normal, model.normal_selected_idxs, \
        model.normal_candidate_flags = edge_decision('normal',
                                                     model.alphas_normal,
                                                     model.normal_selected_idxs,
                                                     model.normal_candidate_flags,
                                                     normal_probs_history,
                                                     epoch,
                                                     model,
                                                     args)



        writer.add_scalar('stats/train_acc', train_acc, epoch)
        writer.add_scalar('stats/valid_acc', valid_acc, epoch)
        utils.save(model, os.path.join(args.save, 'search_weights.pt'))
        scheduler.step()

    logging.info("#" * 30 + " Done " + "#" * 30)
    logging.info('genotype = %s', model.get_genotype())


def train(model, architect, criterion, optimizer, lr):


    model.train()
    total_loss = total_nodes = 0
    for batch in train_loader:
        batch = batch.to(device)
        architect.step(batch.x, batch.edge_index, batch.y, batch.test_mask, batch.y, lr, optimizer, unrolled=args.unrolled)
        ##unrolled=false的时候， input, target,用不到。只会用验证数据，来更新alpha
        ##architect里面的 alpha参数更新，然后下面是w参数更新。
        ## alpha使用adam ，w使用SGD
        optimizer.zero_grad()
        logits = model(batch.x,batch.edge_index)
        loss = criterion(logits[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        nodes = batch.train_mask.sum().item()
        total_loss += loss.item() * nodes   #加权loss？
        total_nodes += nodes
    return 0,total_loss / total_nodes


def infer( model, valid_losses):
    model.eval()
    micro_f1 = 0.
    valid_losses.reset()

    return micro_f1, valid_losses


if __name__ == '__main__':
    adj, features, labels, mask_train,mask_test, \
    y_test_oneclass, mask_test_oneclass,  mask_train1, = load_data('/content/drive/My Drive/NAS-GCN-SAR/data')


    ###传入数据的一些处理
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    edge_index ,edge_weight = from_scipy_sparse_matrix(adj)
    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels).float()
    mask_test_oneclass = torch.from_numpy(np.array(mask_test_oneclass)).to(device)
    y_test_oneclass = torch.from_numpy(np.array(y_test_oneclass)).to(device)

    ##data loader
    data = Data(x=features,edge_index=edge_index,y=labels)
    data.train_mask = torch.from_numpy(mask_train)
    data.test_mask = torch.from_numpy(mask_test)
    cluster_data = ClusterData(data, num_parts=1024, recursive=False)
    train_loader = ClusterLoader(cluster_data, batch_size=64, shuffle=True,
                                 num_workers=12)
    subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1], batch_size=1024,
                                      shuffle=False,num_workers=12)

    main()
