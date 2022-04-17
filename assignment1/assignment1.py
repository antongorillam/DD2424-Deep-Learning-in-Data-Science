from functions import *
import numpy as np
import numpy.matlib

def GetResult(case, n_epochs, n_batch, eta, lamda):
    W_star, b_star = W, b
    cost_train = []
    cost_val = []
    
    loss_train = []
    loss_val = []

    for epoch in range(n_epochs):
        if epoch % 10 == 0:
            print(f'Epoch number: {epoch}/{n_epochs}')
        # Shuffle the data
        indices = np.arange(0, X_train.shape[1])
        np.random.shuffle(indices)
        X = X_train[:, indices]
        Y = Y_train[:, indices]
        W_star, b_star = MiniBatchGD(X, Y, W_star, b_star, lamda, GDParams=[n_batch, eta, n_epochs])
        train_cost, train_loss = ComputeCost(X, Y, W_star, b_star, lamda, get_cost_and_reg=True)
        test_cost, test_loss = ComputeCost(X_test, Y_test, W_star, b_star, lamda, get_cost_and_reg=True)
        
        cost_train.append(train_cost)
        cost_val.append(test_cost)

        loss_train.append(train_loss)
        loss_val.append(test_loss)    

    acc_train = ComputeAccuracy(X_train, y_train, W_star, b_star)
    acc_test = ComputeAccuracy(X_test, y_test, W_star, b_star)

    plotLostVEpochs(
        loss_train=cost_train,
        loss_test=cost_val,
        lamda=lamda,
        n_epochs=n_epochs,
        n_batch=n_batch,
        eta=eta,
        acc=acc_test,
        case=case,
        function="Cost"
        )

    plotLostVEpochs(
        loss_train=loss_train,
        loss_test=loss_val,
        lamda=lamda,
        n_epochs=n_epochs,
        n_batch=n_batch,
        eta=eta,
        acc=acc_test,
        case=case,
        function="Loss"
        )

    print(f'acc_train: {acc_train}')
    print(f'acc_test: {acc_test}')

    # montage(W_star, case=case, params=[n_epochs, n_batch, eta, lamda])

def plotLostVEpochs(loss_train, loss_test, lamda, n_epochs, n_batch, eta, acc, case, function, y_lim=[0,2]):
	import matplotlib.pyplot as plt

	[y_min, y_max] = y_lim
	titleString = f"Case:{case}, {function} function, lambda:{lamda}, epochs:{n_epochs}, batch size:{n_batch}, eta:{eta}\nAccuarcy:{acc}"
	plt.figure()
	plt.plot(loss_train, label='Training loss', color='g')
	plt.plot(loss_test, label='Validation loss', color='r')
	plt.title(titleString)
	plt.xlabel("Epochs")	
	plt.ylabel("Loss")
	# plt.ylim(y_min, y_max)
	plt.legend()
	plt.grid()
	filename = titleString
	filename = filename.replace(":", "")
	filename = filename.replace(",", "")
	filename = filename.replace("\n", "")
	filename = filename.replace(" ", "_")
	filename = filename.replace(".", "")
	filename = filename = filename + ".png"
	plt.savefig(f'Result_Pics/case{case}/' + filename)
    
''' 
X: dxn, 3072 x 10000 
Y: Kxn, 10 x 10000
y: n-vector, 10000-vector
W: kxd
'''
np.random.seed(100)
X_train, Y_train, y_train = Load('cifar-10-batches-py/data_batch_1', show_images=True)
X_test, Y_test, y_test = Load('cifar-10-batches-py/test_batch')

k = Y_train.shape[0]
d = X_train.shape[0]

W = np.random.normal(0, 0.01, size=(k, d))
b = np.random.normal(0, 0.01, size=(k, 1))

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

# GetResult(case=1, n_epochs=40, n_batch=100, eta=0.1, lamda=0)
GetResult(case=2, n_epochs=40, n_batch=100, eta=0.001, lamda=0)
# GetResult(case=3, n_epochs=40, n_batch=100, eta=0.001, lamda=0.1)
# GetResult(case=4, n_epochs=40, n_batch=100, eta=0.001, lamda=1)

# X_train = X_train[:dims,:batch_size]
# Y_train = Y_train[:,:batch_size]
# y_train = y_train[:batch_size]
# W = W[:,:dims]

# p = EvaluateClassifier(X_train, W, b)
# grad_W, grad_b = ComputeGradients(X_train, Y_train, p, W, lamda=0)
# grad_W_test, grad_b_test = ComputeGradsNum(X_train, Y_train, p, W, b, lamda=0   , h=1e-6)
# W_rel_error = CalcRelativeError(grad_W, grad_W_test)
# b_rel_error = CalcRelativeError(grad_b, grad_b_test)

# print(f'W_rel_error for dim {dims}, batch size {batch_size}, lambda {0}: {W_rel_error}')
# print(f'b_rel_error for dim {dims}, batch size {batch_size}, lambda {0}: {b_rel_error}')