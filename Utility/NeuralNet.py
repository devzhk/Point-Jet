import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
from torch import Tensor
from typing import List, Optional
from torch.optim.optimizer import Optimizer
import numpy as np

##################################################################################################################################
#
#   Fully connected nerual network
#
##################################################################################################################################


class Module(torch.nn.Module):
    '''Standard module format. 
    '''
    def __init__(self):
        super(Module, self).__init__()
        self.activation = None
        self.initializer = None
        
        self.__device = None
        self.__dtype = None
        
    @property
    def device(self):
        return self.__device
        
    @property
    def dtype(self):
        return self.__dtype

    @device.setter
    def device(self, d):
        if d == 'cpu':
            self.cpu()
        elif d == 'gpu':
            self.cuda()
        else:
            raise ValueError
        self.__device = d
    
    @dtype.setter    
    def dtype(self, d):
        if d == 'float':
            self.to(torch.float)
        elif d == 'double':
            self.to(torch.double)
        else:
            raise ValueError
        self.__dtype = d

    @property
    def Device(self):
        if self.__device == 'cpu':
            return torch.device('cpu')
        elif self.__device == 'gpu':
            return torch.device('cuda')
        
    @property
    def Dtype(self):
        if self.__dtype == 'float':
            return torch.float32
        elif self.__dtype == 'double':
            return torch.float64

    @property
    def act(self):
        if self.activation == 'sigmoid':
            return torch.sigmoid
        elif self.activation == 'relu':
            return torch.relu
        elif self.activation == 'tanh':
            return torch.tanh
        elif self.activation == 'elu':
            return torch.elu
        else:
            raise NotImplementedError
    
    @property        
    def Act(self):
        if self.activation == 'sigmoid':
            return torch.nn.Sigmoid()
        elif self.activation == 'relu':
            return torch.nn.ReLU()
        elif self.activation == 'tanh':
            return torch.nn.Tanh()
        elif self.activation == 'elu':
            return torch.nn.ELU()
        else:
            raise NotImplementedError

    @property
    def weight_init_(self):
        if self.initializer == 'He normal':
            return torch.nn.init.kaiming_normal_
        elif self.initializer == 'He uniform':
            return torch.nn.init.kaiming_uniform_
        elif self.initializer == 'Glorot normal':
            return torch.nn.init.xavier_normal_
        elif self.initializer == 'Glorot uniform':
            return torch.nn.init.xavier_uniform_
        elif self.initializer == 'orthogonal':
            return torch.nn.init.orthogonal_
        elif self.initializer == 'default':
            if self.activation == 'relu':
                return torch.nn.init.kaiming_normal_
            elif self.activation == 'tanh':
                return torch.nn.init.orthogonal_
            else:
                return lambda x: None
        else:
            raise NotImplementedError
            
class StructureNN(Module):
    '''Structure-oriented neural network used as a general map based on designing architecture.
    '''
    def __init__(self):
        super(StructureNN, self).__init__()
        
    def predict(self, x, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.Dtype, device=self.Device)
        return self(x).cpu().detach().numpy() if returnnp else self(x)

    
class FNN(StructureNN):
    '''Fully connected neural networks.
    '''
    def __init__(self, ind, outd, layers=2, width=50, activation='relu', initializer='default', outputlayer='None', bias=True):
        super(FNN, self).__init__()
        self.ind = ind
        self.outd = outd
        self.layers = layers
        self.width = width
        self.activation = activation
        self.initializer = initializer
        self.outputlayer = outputlayer
        self.bias = bias
        
        self.modus = self.__init_modules()
        self.__initialize()
        
    def forward(self, x):
        for i in range(1, self.layers):
            LinM = self.modus['LinM{}'.format(i)]
            NonM = self.modus['NonM{}'.format(i)]
            x = NonM(LinM(x))
            
        x = self.modus['LinMout'](x)
        
        if self.outputlayer == "square":
            x = x**2
        elif self.outputlayer == "relu":
            x = F.relu(x)
        elif self.outputlayer == "sigmoid":
            x = F.sigmoid(x)
        elif self.outputlayer == "tanh":
            x = F.tanh(x)
        
        return x
    
    def __init_modules(self):
        modules = nn.ModuleDict()
        if self.layers > 1:
            modules['LinM1'] = nn.Linear(self.ind, self.width, bias=self.bias)
            modules['NonM1'] = self.Act
            for i in range(2, self.layers):
                modules['LinM{}'.format(i)] = nn.Linear(self.width, self.width, bias=self.bias)
                modules['NonM{}'.format(i)] = self.Act
            modules['LinMout'] = nn.Linear(self.width, self.outd, bias=self.bias)
        else:
            modules['LinMout'] = nn.Linear(self.ind, self.outd, bias=self.bias)
            
        return modules
    
    def __initialize(self):
        for i in range(1, self.layers):
            self.weight_init_(self.modus['LinM{}'.format(i)].weight)
            if self.bias:
                nn.init.constant_(self.modus['LinM{}'.format(i)].bias, 0)
                
        self.weight_init_(self.modus['LinMout'].weight)
        if self.bias:
            nn.init.constant_(self.modus['LinMout'].bias, 0)
    
    def update_params(self, theta):
        
        theta_ind = 0
        for i in range(1, self.layers):
            
            n_weights = (self.ind if i == 1 else self.width) * self.width
            self.modus['LinM{}'.format(i)].weight = torch.nn.parameter.Parameter(torch.from_numpy(   theta[theta_ind:theta_ind+n_weights].reshape((self.width, -1)).astype(np.float32)  ))
            theta_ind += n_weights
            
            n_biases = self.width
            self.modus['LinM{}'.format(i)].bias = torch.nn.parameter.Parameter(torch.from_numpy(theta[theta_ind: theta_ind+n_biases].astype(np.float32)))
            theta_ind += n_biases
        
        n_weights = self.width*self.outd if self.layers > 1 else self.ind*self.outd
        self.modus['LinMout'].weight = torch.nn.parameter.Parameter(torch.from_numpy(theta[theta_ind: theta_ind+n_weights].reshape((self.outd, -1)).astype(np.float32)))
        theta_ind += n_weights
        
        n_biases = self.outd
        self.modus['LinMout'].bias = torch.nn.parameter.Parameter(torch.from_numpy(theta[theta_ind: theta_ind+n_biases].astype(np.float32)))
        theta_ind += n_biases
        
    def get_params(self):
        N_theta = self.ind*self.width + (self.layers - 2)*self.width**2 + self.width*self.outd + (self.layers - 1)*self.width + self.outd if self.layers > 1 else self.ind*self.outd + self.outd
        print(self.width, N_theta)
        theta = np.zeros(N_theta) 
        
        theta_ind = 0
        for i in range(1, self.layers):
            
            n_weights = (self.ind if i == 1 else self.width) * self.width
            theta[theta_ind:theta_ind+n_weights] =  self.modus['LinM{}'.format(i)].weight.detach().numpy().flatten() 
            theta_ind += n_weights
            
            n_biases = self.width
            theta[theta_ind: theta_ind+n_biases] = self.modus['LinM{}'.format(i)].bias.detach().numpy().flatten() 
            theta_ind += n_biases
        
        n_weights = self.width*self.outd if self.layers > 1 else self.ind*self.outd
        theta[theta_ind: theta_ind+n_weights] = self.modus['LinMout'].weight.detach().numpy().flatten() 
        theta_ind += n_weights
        
        n_biases = self.outd
        theta[theta_ind: theta_ind+n_biases] = self.modus['LinMout'].bias.detach().numpy().flatten() 
        theta_ind += n_biases
        
        return theta
        
        
##################################################################################################################################
#
#   Optimizer
#
##################################################################################################################################
        
def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)


class Adam(Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            adam(params_with_grad,
                 grads,
                 exp_avgs,
                 exp_avg_sqs,
                 max_exp_avg_sqs,
                 state_steps,
                 amsgrad=group['amsgrad'],
                 beta1=beta1,
                 beta2=beta2,
                 lr=group['lr'],
                 weight_decay=group['weight_decay'],
                 eps=group['eps'])
        return loss
    
##################################################################################################################################
#
#   Normalizer
#
##################################################################################################################################
    
    
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        # x = (x - self.mean) / (self.std + self.eps)
        # return x

        # x -= self.mean
        # x /= (self.std + self.eps)
        return (x - self.mean) / (self.std + self.eps)
    
    def encode_(self, x):
        # x = (x - self.mean) / (self.std + self.eps)
        # return x

        x -= self.mean
        x /= (self.std + self.eps)
        

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        # x = (x * std) + mean
        # return x

        # x *= std 
        # x += mean
        return (x * std) + mean

    def decode_(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        # x = (x * std) + mean
        # return x

        x *= std 
        x += mean

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        
        
        
    

    
##########################################################################
# Kernel-Smoothed neural network
##########################################################################


def create_net(ind, outd, layers, width, activation, initializer, outputlayer, params):

    net = FNN(ind, outd, layers, width, activation, initializer, outputlayer) 
    net.update_params(params)
    return net


def create_fno_net(ind, outd, layers, modes1, fc_dim, activation, params):
    
    net = FNO1d(modes1=modes1, layers=layers,
                  fc_dim=fc_dim, in_dim=ind, out_dim=outd, activation=activation)
    net.update_params(params)
    return net

# The eddy viscosity is 
# mu_scale*Int[g(x - y; sigma) NN(y)]dy
def net_eval(net, x, mu_scale=1.0, non_negative=False, filter_on=False, filter_sigma=5.0, n_data=1):
    mu = net(torch.tensor(x, dtype=torch.float32)).detach().numpy().flatten() 
    # data (prediction) clean 
    
    if non_negative:
        mu[mu <= 0.0] = 0.0
    
    if filter_on:
        # the axis is 1
        n_f = len(x)//n_data
        for i in range(n_data):
     
            mu[i*n_f:(i+1)*n_f] = scipy.ndimage.gaussian_filter1d(mu[i*n_f:(i+1)*n_f], filter_sigma, mode="nearest") 
            
    return mu * mu_scale


# x is nx by n_feature matrix, which is the input for the neural network
def nn_viscosity(net, x, mu_scale=1.0, non_negative=False, filter_on=False, filter_sigma=5.0, n_data=1):  
    mu = net_eval(x=x, net=net, mu_scale=mu_scale, non_negative=non_negative, filter_on=filter_on, filter_sigma=filter_sigma, n_data=n_data) 
    return mu

# x is nx by n_feature matrix, which is the input for the neural network
def fno_viscosity(net, x, mu_scale=1.0, non_negative=False, filter_on=False, filter_sigma=5.0, n_data=1):  
    mu = net_eval(x=x[np.newaxis,...], net=net, mu_scale=mu_scale, non_negative=non_negative, filter_on=filter_on, filter_sigma=filter_sigma, n_data=n_data) 
    return mu


# def nn_flux(net, x,  mu_scale=1.0, non_negative=False, filter_on=False, filter_sigma=5.0, n_data=1):
#     # flux = smooth(nn(x)) * mu_scale
#     flux = net_eval(x=x, net=net, mu_scale=mu_scale, non_negative=False, filter_on=filter_on, filter_sigma=filter_sigma, n_data=n_data) 
#     return flux




# def D_nn_permeability(net, q, dq, mu_const = 0.0, mu_scale=1.0, non_negative=False, filter_on=False, filter_sigma=5.0, n_data=1):
#     Ny = q.size
#     Dq, Ddq = np.zeros(Ny), np.zeros(Ny)
    
#     for i in range(Ny):
#         x = torch.from_numpy(np.array([[q[i],dq[i]]]).astype(np.float32))
#         x.requires_grad = True
#         y = net(x)  #.detach().numpy().flatten()
#         d = torch.autograd.grad(y, x)[0].numpy().flatten()
#         Dq[i], Ddq[i] = d[0], d[1]
    
#     return Dq, Ddq


def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", a, b)


def compl_mul2d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return torch.einsum("bixy,ioxy->boxy", a, b)


def compl_mul3d(a, b):
    return torch.einsum("bixyz,ioxyz->boxyz", a, b)

################################################################
# 1d fourier layer
################################################################


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=2)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels,
                             x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = compl_mul1d(
            x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=[x.size(-1)], dim=[2])
        return x
    
class FNO1d(nn.Module):
    def __init__(self, modes1,
                 width=64, fc_dim=128,
                 layers=None,
                 in_dim=2, out_dim=1,
                 activation='tanh'):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, c)
        output: the solution 
        output shape: (batchsize, x=s, 1)
        """

        self.modes1 = modes1
        self.width = width

        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, mode1_num)
            for in_size, out_size, mode1_num
            in zip(self.layers, self.layers[1:], self.modes1)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu_
        elif activation == 'elu':
            self.activation = F.elu_
        else:
            raise ValueError(f'{activation} is not supported')

    def forward(self, x):
        '''
        Args:
            - x : (batch size, x_grid, y_grid, 2)
        Returns:
            - x: (batch size, x_grid, y_grid, 1)
        '''
        length = len(self.ws)
        batchsize = x.shape[0]
        size_x = x.shape[1]

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(
                batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x)
            x = x1 + x2
            if i != length - 1:
                x = self.activation(x)
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    
    # todo
    def update_params(self, theta):
        theta = torch.from_numpy(theta)
        index = 0
        with torch.no_grad():
            for p in self.parameters():
                len_p = p.numel()

                if torch.is_floating_point(p):
                    print(len_p)
                    p[:] = theta[index:index+len_p].reshape(p.shape)
                    index += len_p

                elif torch.is_complex(p):
                    print(len_p)
                    p[:] = torch.complex( theta[index:(index + len_p)] , theta[(index + len_p):(index + len_p * 2)]).reshape(p.shape)
                    index += 2*len_p

                else:
                    print("Bad element type in FNO params")
    
        
    def get_params(self):
        theta = []
        for p in self.parameters():
            if torch.is_floating_point(p):
                theta.extend(p.detach().numpy().flatten())
       

            elif torch.is_complex(p):
                theta.extend(p.real.detach().numpy().flatten())
                theta.extend(p.imag.detach().numpy().flatten())
                
            else:
                print("Bad element type in FNO params")
            
        return np.array(theta)
    
def count_params(model):
    num_param = 0
    for p in model.parameters():
        num_param += p.numel()
    return num_param