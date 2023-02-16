""" Utilities """
import os
import logging
import shutil
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import numpy as np
from datetime import datetime
import copy

# distance correlation
def pairwise_dist(A, B):
    """
    Computes pairwise distances between each elements of A and each elements of
    B.
    Args:
        A,    [m,d] matrix
        B,    [n,d] matrix
    Returns:
        D,    [m,n] matrix of pairwise distances
    """
    # with tf.variable_scope('pairwise_dist'):
    # squared norms of each row in A and B
    na = torch.sum(torch.square(A), 1)
    nb = torch.sum(torch.square(B), 1)

    # na as a row and nb as a column vectors
    na = torch.reshape(na, [-1, 1])
    nb = torch.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    D = torch.sqrt(torch.maximum(na - 2 * torch.mm(A, B.T) + nb + 1e-20, torch.tensor(0.0)))
    return D

def tf_distance_cov_cor(input1, input2, debug=False):
    # n = tf.cast(tf.shape(input1)[0], tf.float32)
    n = torch.tensor(float(input1.size()[0]))
    a = pairwise_dist(input1, input1)
    b = pairwise_dist(input2, input2)
    
    # A = a - tf.reduce_mean(a,axis=1) - tf.expand_dims(tf.reduce_mean(a,axis=0),axis=1) + tf.reduce_mean(a)
    A = a - torch.mean(a,axis=1) - torch.unsqueeze(torch.mean(a,axis=0),axis=1) + torch.mean(a)
    B = b - torch.mean(b,axis=1) - torch.unsqueeze(torch.mean(b,axis=0),axis=1) + torch.mean(b)

    dCovXY = torch.sqrt(torch.sum(A * B) / (n ** 2))
    dVarXX = torch.sqrt(torch.sum(A * A) / (n ** 2))
    dVarYY = torch.sqrt(torch.sum(B * B) / (n ** 2))

    dCorXY = dCovXY / torch.sqrt(dVarXX * dVarYY)
    if debug:
        print(("tf distance cov: {} and cor: {}, dVarXX: {}, dVarYY:{}").format(
            dCovXY, dCorXY,dVarXX, dVarYY))
    # return dCovXY, dCorXY
    return dCorXY

# RVFR
def l21_rownorm(X):
    """
    This function calculates the l21 norm of a matrix X, i.e., \sum ||X[i,:]||_2
    Input:
    -----
    X: {numpy array}
    Output:
    ------
    l21_norm: {float}
    """
    return (np.sqrt(np.multiply(X, X).sum(1))).sum()

# RVFR
def l21_colnorm(X):
    """
    This function calculates the l21 norm of a matrix X, i.e., \sum ||X[:,j]||_2
    Input:
    -----
    X: {numpy array}
    Output:
    ------
    l21_norm: {float}
    """
    return (np.sqrt(np.multiply(X, X).sum(0))).sum()

# RVFR
def RAE_purify(args, LocalOutputs, rae_model, low=None, is_train=False):
    
    if low==None:
        L=torch.autograd.Variable(LocalOutputs, requires_grad=True).to(args.device)
    else:
        L=torch.autograd.Variable(low, requires_grad=True).to(args.device)
    
    # temp_L = L.detach().clone()
    rae_loss_object = torch.nn.MSELoss().to(args.device)

    # rae_output,layer_output=rae_model(L)
    # purify_epochs= args.purify_train_epochs  if is_train==True else args.purify_test_epochs 
    purify_epochs = 10 if is_train==True else 0
    rae_output,layer_output=rae_model(L)
    # print(f"[{is_train}] rae_output-L={rae_output-L}")
    for epoch in range(purify_epochs):
        # L=torch.autograd.Variable(temp_L, requires_grad=True).to(args.device)
        L_optimizer = torch.optim.SGD([L], 1)
        rae_output,layer_output=rae_model(L)
        loss = torch.sqrt(rae_loss_object(rae_output, LocalOutputs))

        L_optimizer.zero_grad()
        # L_gradients = torch.autograd.grad(loss, L, retain_graph=True)
        torch.autograd.backward(loss, inputs=L, retain_graph=True)
        # original_L = L.detach().clone()
        # # loss.backward()
        L_optimizer.step()
        # new_L = L.detach().clone()
        # print(f"L difference:{new_L-original_L}")
        # print(f"L_gradients:{L_gradients}")
        # temp_L = L - L_gradients[0]
    # print(f"[logging info]: RAE purify loss{loss.item()}")
    # assert 0==1
    return  torch.split(rae_output, rae_output.size()[-1]//2, -1), L

# Multistep gradient
def multistep_gradient(tensor, bound_abs, bins_num=12):
    # Criteo 1e-3
    max_min = 2 * bound_abs
    interval = max_min / bins_num
    tensor_ratio_interval = torch.div(tensor, interval)
    tensor_ratio_interval_rounded = torch.round(tensor_ratio_interval)
    tensor_multistep = tensor_ratio_interval_rounded * interval
    return tensor_multistep
    

def sharpen(probabilities, T):
    if probabilities.ndim == 1:
        # print("here 1")
        tempered = torch.pow(probabilities, 1 / T)
        tempered = (
                tempered
                / (torch.pow((1 - probabilities), 1 / T) + tempered)
        )

    else:
        # print("here 2")
        tempered = torch.pow(probabilities, 1 / T)
        tempered = tempered / tempered.sum(dim=-1, keepdim=True)

    return tempered


def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_one_hot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))
    # print("pred shape:", pred.shape)
    # print("torch.log(pred) shape:", torch.log(pred).shape)
    # return torch.mean(torch.sum(- target * torch.log(pred), 1))
    # return torch.mean(torch.sum(- target * pred.log(dim=-1), 1))
    # return torch.mean(torch.sum(- target * torch.log(pred, dim=-1), 1))


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def cross_entropy_for_onehot_samplewise(pred, target):
    return - target * F.log_softmax(pred, dim=-1)


def entropy(predictions):
    epsilon = 1e-6
    H = -predictions * torch.log(predictions + epsilon)
    # print("H:", H.shape)
    return torch.mean(H)

def calculate_entropy(matrix, N=2):
    class_counts = np.zeros(matrix.shape[0])
    all_counts = 0
    for row_idx, row in enumerate(matrix):
        for elem in row:
            class_counts[row_idx] += elem
            all_counts += elem

    # print("class_counts", class_counts)
    # print("all_counts", all_counts)

    weight_entropy = 0.0
    for row_idx, row in enumerate(matrix):
        norm_elem_list = []
        class_count = class_counts[row_idx]
        for elem in row:
            if elem > 0:
                norm_elem_list.append(elem / float(class_count))
        weight = class_count / float(all_counts)
        # weight = 1 / float(len(matrix))
        ent = numpy_entropy(np.array(norm_elem_list), N=N)
        # print("norm_elem_list:", norm_elem_list)
        # print("weight:", weight)
        # print("ent:", ent)
        weight_entropy += weight * ent
    return weight_entropy


def get_timestamp():
    return int(datetime.utcnow().timestamp())


def numpy_entropy(predictions, N=2):
    # epsilon = 1e-10
    # epsilon = 1e-8
    epsilon = 0
    # print(np.log2(predictions + epsilon))
    H = -predictions * (np.log(predictions + epsilon) / np.log(N))
    # print("H:", H.shape)
    return np.sum(H)
    # return H


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path, mode='w')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        if n > 0:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    #print('correct', correct)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy2(output, target, M, device, topk=(1,2,)):
    alpha = 0.001
    # print(output.shape)
    maxk = max(topk)
    batch_size = target.size(0)
    pred_count, pred_index = output.topk(maxk, 1, True, True)
    # print(pred_value,pred_index)
    # print("pred.shape", pred_count.shape,batch_size) #[64, 2],batch_size=N=64 for MNIST

    pred = pred_index.t()[0]
    for i in range(pred.shape[0]):
        pa = pred_count[i][0] / M
        pb = pred_count[i][1] / M
        shift = np.sqrt(np.log(1/alpha)/(2*batch_size))
        # print("pa=",pa,", pb=",pb," ,shift=",shift)
        pa = pa - shift
        pb = pb + shift
        # print("pa=",pa,", pb=",pb," ,shift=",shift)
        if pa <= pb:
            pred[i] = -1
    # print(pred)
    # print(target)

    # print('pred in device :',pred.device)
    # print('target in device :',target.device)
    pred = pred.cuda()
    target = target.cuda()
    correct = pred.eq(target.view(1, -1))
    # correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in (1,):
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy3(output, target):
    batch_size = target.size(0)
    correct = output.eq(target).sum()
    # print('correct', correct.sum())
    return correct * (100.0 / batch_size)


def vote(output, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()[0]
    return pred


def create_exp_dir(path, scripts_to_save=None):
    os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(state, ckpt_dir, is_best=False):
    os.makedirs(ckpt_dir, exist_ok=True)
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)


# def ClipAndPerturb(model,device,ro,sigma):
#     model_dict = model.state_dict()
#     temp_dict = {}
#     for k,v in model_dict.items():
#         temp_dict[k] = v
#         _norm = np.linalg.norm(temp_dict[k].cpu(),ord=1)
#         # print("L2 norm of parameter =",_norm)
#         temp_dict[k] = temp_dict[k]/max(1,(_norm/ro))
#         temp_dict[k].to(device)
#         temp_dict[k] += torch.normal(0.0, sigma*sigma, temp_dict[k].shape).to(device)
#     temp_model = copy.deepcopy(model)
#     temp_model.load_state_dict(temp_dict)
#     return temp_model
def ClipAndPerturb(vector,device,ro,sigma):
    _norm = np.linalg.norm(vector.cpu().detach().numpy(),ord=1)
    # print("L2 norm of parameter =",_norm)
    vector = vector/max(1,(_norm/ro))
    vector.to(device)
    vector += torch.normal(0.0, sigma*sigma, vector.shape).to(device)
    return vector
