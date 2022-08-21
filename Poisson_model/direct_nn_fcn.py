import scipy.ndimage
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
import matplotlib.pyplot as plt
from functools import partial

import torch
from Solver import *
# from NeuralNet import *
from timeit import default_timer
from utils import get_data

import sys
sys.path.append('../Utility')
import NeuralNet
import Numerics
import PlotDefault


## Data preparation
L = 1.0
test_res = 100      # test data resolution
train_res = 100     # train data resolution

Nx = train_res
Ny = test_res
yy = np.linspace(0.0, L, Ny)
dy = yy[1] - yy[0]
f = -np.ones_like(yy)
dbc = np.array([0.0, 0.0]) 

GENERATE_TRAIN = False
GENERATE_TEST = False

# Training data  
xx, f, q, xx_test, f_test, q_test = get_data(train_res=train_res, 
                                             test_res=test_res, 
                                             generate_train=GENERATE_TRAIN, 
                                             generate_test=GENERATE_TEST)
dx = xx[1] - xx[0]

dq = np.copy(q)
ddq = np.copy(q)
for i in range(q.shape[0]):
    dq[i, :]  = Numerics.gradient_first(q[i,:], dx, bc = "one-sided")
    ddq[i, :] = Numerics.gradient_second(q[i,:], dx, bc = "one-sided")
 
dq_test = np.copy(q_test)
ddq_test = np.copy(q_test)
for i in range(q_test.shape[0]):
    dq_test[i, :]  = Numerics.gradient_first(q_test[i,:], dy, bc = "one-sided")
    ddq_test[i, :] = Numerics.gradient_second(q_test[i,:], dy, bc = "one-sided")
  
mu, flux, source = np.copy(q), np.copy(q), np.copy(q)
for i in range(q.shape[0]):
    mu[i,:], flux[i,:], source[i,:] = permeability_ref(np.vstack((q[i,:], dq[i,:])).T), flux_ref(np.vstack((q[i,:], dq[i,:])).T), source_ref_q(q[i,:], dy)
    
mu_test, flux_test, source_test = np.copy(q_test), np.copy(q_test), np.copy(q_test)
for i in range(q_test.shape[0]):
    mu_test[i,:], flux_test[i,:], source_test[i,:] = permeability_ref(np.vstack((q_test[i,:], dq_test[i,:])).T), flux_ref(np.vstack((q_test[i,:], dq_test[i,:])).T), source_ref_q(q_test[i,:], dy)


## training code 
model_list = ["diffusivity.nn"] 
learning_rate = 0.001
step_size = 100
gamma = 0.5  
epochs = 20001
batch_size = 64

ind, outd, width = 1, 1, 10
layers = 2
activation, initializer, outputlayer = "sigmoid", "default", "None"


for nn_save_name in model_list:
    print("start train nn : ", nn_save_name)
    if nn_save_name == "diffusivity.nn":
        x_train = q.flatten()[:, np.newaxis]
        y_train = mu.flatten()[:,np.newaxis]
    elif nn_save_name == "flux.nn":
        x_train = q.flatten()[:, np.newaxis]
        y_train = flux.flatten()[:,np.newaxis]
    elif nn_save_name == "source.nn":
        x_train = np.vstack((q.flatten(), dq.flatten(), ddq.flatten())).T
        y_train = source.flatten()[:,np.newaxis]
    else:
        print("nn_save_name : ", nn_save_name, " is not recognized")

    x_train = torch.from_numpy(x_train.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))

    ind = x_train.shape[1]
    outd = y_train.shape[1]     

    net = NeuralNet.FNN(ind, outd, layers, width, activation, initializer, outputlayer) 
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    optimizer = NeuralNet.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    myloss = torch.nn.MSELoss(reduction='sum')
    t0 = default_timer()
    for ep in range(epochs):
        net.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:

            optimizer.zero_grad()
            out = net(x)

            loss = myloss(out , y)*100
            loss.backward()

            optimizer.step()
            train_l2 += loss.item()


        if ep % 1000 == 0:
            # train_l2/= ntrain
            t2 = default_timer()
            print("Epoch : ", ep, " Epoch time : ", t2-t1, " Train L2 Loss : ", train_l2)
            if nn_save_name is not None:
                torch.save(net, nn_save_name)

print('Training done. Start test..')

L, Nx =1.0, test_res
xx = np.linspace(0.0, L, Nx)
dx = xx[1] - xx[0]
dbc = np.array([0.0, 0.0])  

def f_func1(xx_test):
    return 6*(1-2*xx_test)**2 - 2*(xx_test - xx_test**2)*(1 - 2*xx_test)**2 + 2*(xx_test - xx_test**2)**2 + 2

def f_func2(xx_test):
    f = np.ones_like(xx_test)
    f[xx_test <= 0.5] = 0.0
    f[xx_test > 0.5] = 10.0
    return f

def f_func3(xx_test):
    L = 1
    return 10*np.sin(2*np.pi*xx_test/L)

f_funcs = [f_func1, f_func2, f_func3]


for nn_save_name in model_list:
# for nn_save_name in ["source.nn", "diffusivity.nn", "flux.nn"]:
    trained_net = torch.load(nn_save_name)
    fig, ax = plt.subplots(ncols=3, nrows=2, sharey="row", figsize=((22,12)))

    for i in tqdm(range(3)):

        f_func = f_funcs[i]
        f = f_func(xx)   
        filter_on = False
        if nn_save_name  == "diffusivity.nn":
            nn_model = partial(NeuralNet.nn_viscosity, net=trained_net, mu_scale = mu_scale, non_negative=True, filter_on=filter_on, filter_sigma=filter_sigma)
            model = lambda q, yy, res : nummodel(nn_model, q, yy, res)
        elif nn_save_name  == "flux.nn":
            nn_model = partial(NeuralNet.nn_viscosity, net=trained_net, mu_scale = flux_scale, non_negative=False, filter_on=filter_on, filter_sigma=filter_sigma)
            model = lambda q, yy, res : nummodel_flux(nn_model, q, yy, res)
        elif nn_save_name  == "source.nn":
            nn_model = partial(NeuralNet.nn_viscosity, net=trained_net, mu_scale = source_scale, non_negative=False, filter_on=filter_on, filter_sigma=filter_sigma)
            model = lambda q, yy, res : nummodel_source(nn_model, q, yy, res)
        else:
            print("nn_save_name : ", nn_save_name, " is not recognized")
            
        _, _, q_data = explicit_solve(model, f, dbc, dt = 1.0e-7, Nt = 10_000_000, save_every = 1_000_000, L = L)
        q_pred = q_data[-1, :]
        dq_pred  = Numerics.gradient_first(q_pred, dy, bc = "one-sided")
        ax[0,i].plot(xx_test[i,:], q_test[i,:],  "--o", color="black", fillstyle="none", label="Reference", markevery=10)
        ax[0,i].plot(xx_test[i,:], q_pred,  "--*", color="red", label="Prediction", markevery=10)
        if i == 0:
            ax[0,i].set_ylabel("q")
        ax[0,i].set_xlabel("x")

        for j in range(q.shape[0]):
            ax[1,i].plot(q[j, :], dq[j, :],  "o", color = "black", fillstyle="none", alpha=0.1)

        ax[1,i].plot(q_test[i,:], dq_test[i,:],  "--o", color="black", fillstyle="none", label="Reference", markevery=10)
        ax[1,i].plot(q_pred, dq_pred,  "--*", color="red", label="Prediction", markevery=10)
        ax[1,i].set_xlim([-1,1])
        ax[1,i].set_ylim([-2,2])
        if i == 0:
            ax[1,i].set_ylabel("dq")
        ax[1,i].set_xlabel("q")

handles, labels = ax[0,0].get_legend_handles_labels()
fig.subplots_adjust(bottom=0.08,top=0.92,left=0.08,right=0.97)
fig.legend(handles,labels,loc = "upper center",bbox_to_anchor=(0.5,0.98),ncol=2,frameon=False)
fig.savefig("Poisson-NN.png")