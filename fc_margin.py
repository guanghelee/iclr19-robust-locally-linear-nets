from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os
from data_utils import get_mnist_train_valid_loader, get_mnist_test_loader
from utils import print_args
from bitarray import bitarray
from collections import defaultdict


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(300, 300)
        self.fc5 = nn.Linear(300, 10)

        self.template = torch.zeros([64, 28 * 28 + 1, 28 * 28], requires_grad=False)
        self.template_n = 28 * 28 + 1
        for i in range(28*28):
            self.template[:, i+1, i] = 1.
        self.template = self.template.view(-1, 1, 28, 28).pin_memory()
        self.grad_template = torch.ones([64], requires_grad=False)

    def get_l2sqr(self, w):
        return (w ** 2).sum(dim=1)

    def get_grad(self, relu_masks, return_last_jacobian=False):
        #with torch.no_grad():      # check Jac correctness
        total_grad = []
        batch_size = relu_masks[0].shape[0]

        x = self.template.view(-1, 28 * 28)
        x = self.fc1(x)
        x = x.view(-1, self.template_n, 300)
        x = x[:batch_size]
        total_grad.append(self.get_l2sqr(
                x[:, 1:, :] - x[:, :1, :]
        ))
        x = x * relu_masks[0].view(-1, 1, 300).type(torch.cuda.FloatTensor)

        x = x.view(-1, 300)
        x = self.fc2(x)
        x = x.view(-1, self.template_n, 300)
        total_grad.append(self.get_l2sqr(
                x[:, 1:, :] - x[:, :1, :]
        ))
        x = x * relu_masks[1].view(-1, 1, 300).type(torch.cuda.FloatTensor)

        x = x.view(-1, 300)
        x = self.fc3(x)
        x = x.view(-1, self.template_n, 300)
        total_grad.append(self.get_l2sqr(
                x[:, 1:, :] - x[:, :1, :]
        ))
        x = x * relu_masks[2].view(-1, 1, 300).type(torch.cuda.FloatTensor)

        x = x.view(-1, 300)
        x = self.fc4(x)
        x = x.view(-1, self.template_n, 300)
        total_grad.append(self.get_l2sqr(
                x[:, 1:, :] - x[:, :1, :]
        ))

        if not return_last_jacobian:
            return torch.cat(total_grad, dim=1)
        else:    
            x = x * relu_masks[3].view(-1, 1, 300).type(torch.cuda.FloatTensor)

            x = x.view(-1, 300)
            x = self.fc5(x)
            x = x.view(-1, self.template_n, 10)
            x = x[:, 1:, :] - x[:, :1, :]

            return torch.cat(total_grad, dim=1), x
    

    def forward(self, x, DEBUG_JAC=False):
        x = x.view(-1, 28 * 28)
        relu_masks = []
        logits = []

        x = self.fc1(x)
        logits.append(x)
        relu_masks.append( (x > 0) )
        x = F.relu(x)
        
        x = self.fc2(x)
        logits.append(x)
        relu_masks.append( (x > 0) )
        x = F.relu(x)
        
        x = self.fc3(x)
        logits.append(x)
        relu_masks.append( (x > 0) )
        x = F.relu(x)
        
        x = self.fc4(x)
        logits.append(x)
        relu_masks.append( (x > 0) )
        x = F.relu(x)
        
        x = self.fc5(x)
        logits = torch.cat(logits, dim=1)

        if DEBUG_JAC:
            raw_pred = x
            return F.log_softmax(x, dim=1), logits, relu_masks, raw_pred
        else:
            return F.log_softmax(x, dim=1), logits, relu_masks


def get_distance(abs_logits, grad_l2sqr):
    eps = 1e-10
    distance = abs_logits / (torch.sqrt(grad_l2sqr) + eps)
    return torch.min(distance, dim=1)

def get_TSVM_loss(abs_logits, grad_l2sqr, args):
    TSVM_loss = grad_l2sqr + args.C * F.relu(1 - abs_logits)

    if args.aggregate == 1:
        return torch.mean(TSVM_loss, dim=1), 0
    elif args.aggregate == 0:
        return torch.max(TSVM_loss, dim=1)
    else:
        n_neurons = TSVM_loss.shape[1]
        topk = int(n_neurons * args.aggregate)
        topk_loss = torch.topk(TSVM_loss, topk, dim = 1)[0]
        return torch.mean(topk_loss, dim=1), 0

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    #model.eval()
    dataset_len = len(train_loader.sampler.indices)
    for batch_idx, (data, target) in enumerate(train_loader):
        #print(data.shape)
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        
        optimizer.zero_grad()
        ###############
        output, logits, relu_masks = model(data)
        grad_l2sqr = model.get_grad(relu_masks)
        ###############
        #check_grad(logits, grad_l2sqr, data, model)

        abs_logits = torch.abs(logits)
        #distance, neuron_index = get_distance(abs_logits, grad_l2sqr)
        TSVM_loss, TSVM_index  = get_TSVM_loss(abs_logits, grad_l2sqr, args)
        loss = F.nll_loss(output, target) + args.lbda * TSVM_loss.mean()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), dataset_len,
                100. * batch_idx / len(train_loader), loss.item()), flush=True)

def get_activation_pattern(relu_masks, data, all_act_pattern):
    relu_masks = torch.cat(relu_masks, dim=1) 
    relu_masks = relu_masks.to('cpu').numpy()
    relu_masks = list(relu_masks)
    for i in range(len(relu_masks)):
        relu_masks[i] = bitarray(list(relu_masks[i])).to01()
        all_act_pattern[relu_masks[i]].append(data[i])
    return all_act_pattern

def get_jacobian(jacobian, data, all_jacobian):
    ##### REMEMBER: emphasize that different Jacobian is not caused by numerical error #####
    jacobian = jacobian.to('cpu').numpy().round(4)
    jacobian = list(jacobian)

    for i in range(len(jacobian)):
        all_jacobian[tuple(jacobian[i].flatten())].append(data[i])
    return all_jacobian

    
def test(args, model, device, test_loader, test_set_name):
    model.eval()
    test_loss = 0
    correct = 0
    all_distance = []
    
    all_act_pattern = defaultdict(list)
    all_jacobian = defaultdict(list)
    with torch.no_grad():
    #with open('tmp.txt','w') as fp:                                                # check Jac correctness
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # data.requires_grad = True                                             # check Jac correctness
            ###############
            output, logits, relu_masks = model(data)
            #output, logits, relu_masks, raw_pred = model(data, DEBUG_JAC=True)     # check Jac correctness
            grad_l2sqr, jacobian = model.get_grad(relu_masks, return_last_jacobian=True)
            # check_jacobian(raw_pred, data, jacobian, model)                       # check Jac correctness
            ###############

            data = data.to('cpu').numpy()
            all_act_pattern = get_activation_pattern(relu_masks, data, all_act_pattern)
            all_jacobian    = get_jacobian(jacobian, data, all_jacobian)

            abs_logits = torch.abs(logits)
            distance, neuron_index = get_distance(abs_logits, grad_l2sqr)
            all_distance.append(distance)
            TSVM_loss, TSVM_index  = get_TSVM_loss(abs_logits, grad_l2sqr, args)
            # sum up batch loss
            test_loss += ( F.nll_loss(output, target, reduction='sum').item() + args.lbda * TSVM_loss.sum() )
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    all_distance = torch.cat(all_distance, dim=0).to('cpu').numpy()

    if test_set_name == 'valid':
        dataset_len = len(test_loader.sampler.indices)
    else:
        dataset_len = len(test_loader.dataset)
    test_loss /= dataset_len
    
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_set_name, test_loss, correct, dataset_len,
        100. * correct / dataset_len))
    print('# of Activation Pattern: {}, # of Jacobian: {}'.format(
        len(all_act_pattern), len(all_jacobian)))
    all_distance = all_distance * 0.3081
    print('Distance Percentiles: {:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(
        np.min(all_distance), np.percentile(all_distance, 25), np.median(all_distance), np.percentile(all_distance, 75), np.max(all_distance)), flush=True)

    return test_loss, all_distance

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # obsolete
    parser.add_argument('--anneal_epoch', type=int, default=0, metavar='S',
                        help='anneal_epoch (default: 0)')
    parser.add_argument('--C', type=float, default=5.,
                        help='C for TSVM regularization (default: 5)')
    parser.add_argument('--lbda', type=float, default=1.,
                        help='lbda for TSVM regularization (default: 1)')
    parser.add_argument('--resume', type=str, default='',
                        help='resume from checkpoint')
    parser.add_argument('--aggregate', type=float, default=0.,
                        help='top percentage of TSVM loss to be aggregated, between 0 for the max and 1 for the mean (default: 1 = mean)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    print_args(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, valid_loader = get_mnist_train_valid_loader('../data', args.batch_size, args.seed)
    test_loader = get_mnist_test_loader('../data', args.batch_size)

    print('batch number: train={}, valid={}, test={}'.format(len(train_loader), len(valid_loader), len(test_loader)))

    model = Net().to(device)
    model.template = model.template.to(device)
    model.grad_template = model.grad_template.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)

    best_loss = 1e30  # best validation loss
    start_epoch = 1  # start from epoch 1 or last checkpoint epoch

    model_name = './checkpoint/mnist_fc_C_{}_lbda_{}_aggregate_{}_anneal_{}_ckpt.t7'.format(args.C, args.lbda, args.aggregate, args.anneal_epoch)

    if args.resume != '':
        # Load checkpoint.
        model_name = args.resume[:-3] + '_resume_mnist_fc_lr_{}_C_{}_lbda_{}_aggregate_{}_anneal_{}_ckpt.t7'.format(args.lr, args.C, args.lbda, args.aggregate, args.anneal_epoch)
        
        print('==> Resuming from checkpoint..', args.resume)
        print('current model name =', model_name)
        
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])

        valid_loss, distance = test(args, model, device, valid_loader, 'valid')
        start_epoch = checkpoint['epoch']

        #best_loss = valid_loss
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'loss': valid_loss,
            'distance': distance,
            'epoch': start_epoch,
        }
        torch.save(state, model_name)
        best_loss = valid_loss

    for epoch in range(start_epoch, args.epochs + start_epoch):
        train(args, model, device, train_loader, optimizer, epoch)
        valid_loss, distance = test(args, model, device, valid_loader, 'valid')
        
        if valid_loss < best_loss:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'loss': valid_loss,
                'distance': distance,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, model_name)
            best_loss = valid_loss

    print('Training finished. Loading best model from validation...')
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['model'])
    valid_loss, distance = test(args, model, device, valid_loader, 'valid')
    test_loss, distance = test(args, model, device, test_loader, 'test ')


if __name__ == '__main__':
    main()


### utility function checking correctness of the implementation
def check_grad(logits, grad_l2sqr, data, model):
    # for all the logits
    safe = True
    for idx in range(900, 1200):
        torch_grad = torch.autograd.grad(outputs=logits[:, idx], inputs=data, grad_outputs=model.grad_template[:data.shape[0]]\
            , only_inputs=True, retain_graph = True)[0]
        torch_grad = torch_grad.view(-1, 28 * 28)
        torch_grad_l2sqr = model.get_l2sqr(torch_grad)

        # check numerical stability
        if (torch.abs(torch_grad_l2sqr - grad_l2sqr[:, idx]).mean() >= 1e-5) :
            print(torch_grad_l2sqr)
            print(grad_l2sqr[:, idx])
            print(torch.abs(torch_grad_l2sqr - grad_l2sqr[:, idx]).mean(), '\n')
            safe = False

    print(logits.shape, torch_grad_l2sqr.shape, 'safe =', safe)


def check_jacobian(raw_pred, data, jacobian, model):
    safe = True
    torch_jacobian = []
    for idx in range(10):
        torch_grad = torch.autograd.grad(outputs=raw_pred[:, idx], inputs=data, grad_outputs=model.grad_template[:data.shape[0]]\
            , only_inputs=True, retain_graph = True)[0]
        torch_grad = torch_grad.view(-1, 28 * 28, 1)
        torch_jacobian.append(torch_grad)

    torch_jacobian = torch.cat(torch_jacobian, dim=2)
    # check numerical stability
    if (torch.abs(torch_jacobian - jacobian).mean() >= 1e-5) :
        print(torch.abs(torch_jacobian - jacobian).mean())
    print(torch.abs(torch_jacobian - jacobian).mean())
