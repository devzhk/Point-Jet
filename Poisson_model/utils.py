import pickle 
import numpy as np
from Solver import generate_data, generate_data_helper, permeability_ref


def get_data(
    train_res=100, 
    test_res=100,
    generate_train=True, 
    generate_test=True):

    Ny = test_res

    if generate_train:  
        xx, f, q, q_c, dq_c = generate_data(n_data=9)
        data_dict = {'xx': xx, 'f': f, 'q': q}
        with open(f'train-data-s{train_res}.pickle', 'wb') as f:
            pickle.dump(data_dict, f)
    else:
        with open(f"train-data-s{train_res}.pickle",'rb') as f:
            data_dict = pickle.load(f)   
        xx, f, q = data_dict["xx"], data_dict["f"], data_dict["q"]

    if generate_test:
        # Test data
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
        xx_test, f_test, q_test, q_c_test, dq_c_test = np.zeros((3, Ny)), np.zeros((3, Ny)), np.zeros((3, Ny)), np.zeros((3, Ny-1)), np.zeros((3, Ny-1))

        for i in range(3):
            f_func = f_funcs[i]
            xx_test[i,:], f_test[i,:], q_test[i,:], q_c_test[i,:], dq_c_test[i,:] = generate_data_helper(permeability_ref, f_func, 
                                                                                                        dt=1e-7, Nt=10_000_000, L=1.0, Nx =Ny)
        
        data_dict = {'xx_test': xx_test, 'f_test': f_test, 'q_test': q_test}
        with open(f'test-data-s{test_res}.pickle', 'wb') as f:
            pickle.dump(data_dict, f)
    else:
        
        with open(f"test-data-s{test_res}.pickle",'rb') as f:
            data_dict = pickle.load(f)   
        xx_test, f_test, q_test = data_dict["xx_test"], data_dict["f_test"], data_dict["q_test"]
    return xx, f, q, xx_test, f_test, q_test