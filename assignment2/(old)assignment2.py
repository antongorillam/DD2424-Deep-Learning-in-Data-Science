from functions import *
import numpy as np
import numpy.matlib
import time


def GetResult(dir, n_epochs, n_batch, lamda, eta):
    W_star, b_star = W, b
    eta_min, eta_max = eta

    cost_train = []
    cost_test = []
    
    loss_train = []
    loss_test = []

    acc_train = []
    acc_test = []

    eta_list = []
    total_iteations = 0
    tic = time.perf_counter()

    for epoch in range(n_epochs):

        if epoch % 5 == 0:
            elapsed = time.perf_counter() - tic
            print(f'Epoch number: {epoch}/{n_epochs}, Time elapsed: {int(elapsed/60)}min {int(elapsed%60)}sec')

        # Shuffle the data
        indices = np.arange(0, X_train.shape[1])
        np.random.shuffle(indices)
        X = X_train[:, indices]
        Y = Y_train[:, indices]
        W_star, b_star, total_iteations, eta_temp = MiniBatchGD(X, Y, W_star, b_star, total_iteations, lamda, GDParams=[n_batch, n_epochs, eta_min, eta_max])
        
        eta_list += eta_temp 
        train_cost, train_loss = ComputeCost(X, Y, W_star, b_star, lamda)
        test_cost, test_loss = ComputeCost(X_test, Y_test, W_star, b_star, lamda)

        train_acc = ComputeAccuracy(X_train, y_train, W_star, b_star)
        test_acc = ComputeAccuracy(X_test, y_test, W_star, b_star)

        cost_train.append(train_cost)
        cost_test.append(test_cost)

        loss_train.append(train_loss)
        loss_test.append(test_loss)    

        acc_train.append(train_acc)
        acc_test.append(test_acc)

    elapsed = time.perf_counter() - tic
    print(f'Epoch number: {n_epochs}/{n_epochs}, Time elapsed: {int(elapsed/60)}min {int(elapsed%60)}sec')

    plotLostVEpochs(cost_train, cost_test, metric='Cost', x_axis='epoch', title_string=f'Cost Function\nn_epochs:{n_epochs}, n_batch:{n_batch}, lambda:{lamda}, test acc:{test_acc}', dir=dir)
    plotLostVEpochs(loss_train, loss_test, metric='Loss', x_axis='epoch', title_string=f'Loss Function\nn_epochs:{n_epochs}, n_batch:{n_batch}, lambda:{lamda}, test acc:{test_acc}', dir=dir)
    plotLostVEpochs(acc_train, acc_test, metric='Acc', x_axis='epoch', title_string=f'Accuarcy \nn_epochs:{n_epochs}, n_batch:{n_batch}, lambda:{lamda}, test acc:{test_acc}', dir=dir)
    plotLostVEpochs(eta_list, [], metric='eta', x_axis='steps', title_string=f'Learning Rate (eta) \nn_epochs:{n_epochs}, n_batch:{n_batch}, lambda:{lamda}, test acc:{test_acc}', dir=dir)


def plotLostVEpochs(loss_train, loss_test, metric, x_axis, title_string, dir, y_lim=[0,2]):
    import matplotlib.pyplot as plt

    [y_min, y_max] = y_lim
    plt.figure()
    plt.plot(loss_train, label=f'Training {metric}', color='g')
    plt.plot(loss_test, label=f'Validation {metric}', color='r')
    plt.title(title_string)
    plt.xlabel(x_axis)	
    plt.ylabel(metric)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()
    filename = title_string
    filename = filename.replace(":", "")
    filename = filename.replace(",", "")
    filename = filename.replace("\n", "")
    filename = filename.replace(" ", "_")
    filename = filename.replace(".", "")
    filename = filename = filename + ".png"
    plt.savefig(dir + filename)

def getData(seed):
    np.random.seed(seed)
    X_train, Y_train, y_train = Load('cifar-10-batches-py/data_batch_1', show_images=False)
    X_test, Y_test, y_test = Load('cifar-10-batches-py/data_batch_2')

    k = Y_train.shape[0]
    d = X_train.shape[0]

    mean_X_train = np.mean(X_train, 1)
    mean_X_train = mean_X_train.reshape(d, 1)
    std_X_train = np.std(X_train, 1)
    std_X_train = std_X_train.reshape(d, 1)

    mean_X_test = np.mean(X_test, 1)
    mean_X_test = mean_X_test.reshape(d, 1)
    std_X_test = np.std(X_test, 1)
    std_X_test = std_X_test.reshape(d, 1)

    X_train = X_train-np.matlib.repmat(mean_X_train, 1, X_train.shape[1])
    X_train = X_train/np.matlib.repmat(std_X_train, 1, X_train.shape[1])

    X_test = X_test-np.matlib.repmat(mean_X_test, 1, X_test.shape[1])
    X_test = X_test/np.matlib.repmat(std_X_test, 1, X_test.shape[1])
    
    return X_train, Y_train, y_train, X_test, Y_test, y_test 

''' 
X: dxn, 3072 x 10000 
Y: Kxn, 10 x 10000
y: n-vector, 10000-vector
W1: mxd
W2: kxm
'''
# from functions import *
# SEED = 100
# np.random.seed(SEED)
# X_train, Y_train, y_train = Load('cifar-10-batches-py/data_batch_1', show_images=False)
# X_test, Y_test, y_test = Load('cifar-10-batches-py/data_batch_2')

# X_train, Y_train, y_train, X_test, Y_test, y_test = getData(SEED) 

# k = Y_train.shape[0]
# d = X_train.shape[0]
# m = 50

# [W1, W2], [b1, b2] = GetWeightBias(k, d, m, seed=100)
# dims = 3072
# batch_size = 10000
# LAMDA = 0.01

# X_train = X_train[:dims,:batch_size]
# # Y_train = Y_train[:,:batch_size]
# y_train = y_train[:batch_size]
# W1 = W1[:,:dims]

# W = np.array([W1, W2], dtype=object)
# b = np.array([b1, b2], dtype=object)


# p, X = EvaluateClassifier(X_train, W, b)
# c = ComputeCost(X_train, Y_train, W, b, lamda=LAMDA)
# acc = ComputeAccuracy(X_train, y_train, W, b)

# grad_W, grad_b = ComputeGradients(X, Y_train, p, W, lamda=LAMDA)

# # grad_W_num, grad_b_num = ComputeGradsNumSlow(X_train, Y_train, W[0], b[0], W[1], b[1], lamda=LAMDA, h=1e-5)

# # print(f'W: {grad_W[0].shape}, {grad_W[1].shape}, b: {grad_b[0].shape}, {grad_b[1].shape}')

# # print(f'W num: {grad_W_num[0].shape}, {grad_W_num[1].shape}, b num: {grad_b_num[0].shape}, {grad_b_num[1].shape}')
# # W_rel_error = CalcRelativeError(grad_W, grad_W_num)
# # b_rel_error = CalcRelativeError(grad_b, grad_b_num)

# # print(f'grad_W: {grad_W.shape}, grad_b: {grad_b.shape}')

# # GetResult(case=1, n_epochs=40, n_batch=100, eta=0.001, lamda=0)
# GetResult(dir='Result_Pics/exercise3/', n_epochs=10, n_batch=100, lamda=LAMDA, eta=[1e-5, 1e-1])
# # print(f'W_rel_error for dim {dims}, batch size {batch_size}, lambda {LAMDA}: {W_rel_error}')
# # print(f'b_rel_error for dim {dims}, batch size {batch_size}, lambda {LAMDA}: {b_rel_error}')
    