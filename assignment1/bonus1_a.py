from functions import *
import numpy as np
import numpy.matlib

def GetResult(case, n_epochs, n_batch, eta, lamda):
    W_star, b_star = W, b
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
        W_star, b_star = MiniBatchGD(X_train, Y_train, W_star, b_star, lamda, GDParams=[n_batch, eta, n_epochs])
        loss_train.append(ComputeCost(X_train, Y_train, W_star, b_star, lamda))
        loss_val.append(ComputeCost(X_test, Y_test, W_star, b_star, lamda))    

    acc_train = ComputeAccuracy(X_train, y_train, W_star, b_star)
    acc_test = ComputeAccuracy(X_test, y_test, W_star, b_star)

    plotLostVEpochsBonus1(
        loss_train=loss_train,
        loss_test=loss_val,
        lamda=lamda,
        n_epochs=n_epochs,
        n_batch=n_batch,
        eta=eta,    
        acc=acc_test,
        case=case,
        )

    np.savetxt(f'Result_Pics/bonus1a/loss_train_lambda{lamda}_nEpochs{n_epochs}_nBatch{n_batch}_eta{eta}.csv', loss_train, delimiter=",")
    np.savetxt(f'Result_Pics/bonus1a/loss_validation_lambda{lamda}_nEpochs{n_epochs}_nBatch{n_batch}_eta{eta}.csv', loss_val, delimiter=",")


    print(f'acc_train: {acc_train}')
    print(f'acc_test: {acc_test}')

def plotLostVEpochsBonus1(loss_train, loss_test, lamda, n_epochs, n_batch, eta, acc, case ,y_lim=[0,2]):
	import matplotlib.pyplot as plt

	[y_min, y_max] = y_lim
	titleString = f"Case {case} Optimization a: lambda {lamda}, epochs {n_epochs}, number bacthes {n_batch} and eta {eta}\nAccuarcy:{acc}"
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
	plt.savefig(f'Result_Pics/bonus1a/' + filename)

''' 
X: dxn, 3072 x 10000 
Y: Kxn, 10 x 10000
y: n-vector, 10000-vector
W: kxd
'''
np.random.seed(100)
X_train1, Y_train1, y_train1 = Load('cifar-10-batches-py/data_batch_1', show_images=False)
X_train2, Y_train2, y_train2 = Load('cifar-10-batches-py/data_batch_2', show_images=False)
X_train3, Y_train3, y_train3 = Load('cifar-10-batches-py/data_batch_3', show_images=False)
X_train4, Y_train4, y_train4 = Load('cifar-10-batches-py/data_batch_4', show_images=False)
X_train5, Y_train5, y_train5 = Load('cifar-10-batches-py/data_batch_5', show_images=False)

X_train = np.concatenate((X_train1, X_train2, X_train3, X_train4, X_train5), axis=1)
Y_train = np.concatenate((Y_train1, Y_train2, Y_train3, Y_train4, Y_train5), axis=1)
y_train = np.concatenate((y_train1, y_train2, y_train3, y_train4, y_train5), axis=0)

X_test, Y_test, y_test = Load('cifar-10-batches-py/test_batch')
X_test = X_test[:,:1000]
Y_test = Y_test[:,:1000]
y_test = y_test[:1000] 

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

# GetResult(case='1', n_epochs=40, n_batch=100, eta=0.1, lamda=0)
# GetResult(case='2', n_epochs=40, n_batch=100, eta=0.001, lamda=0)
GetResult(case='3', n_epochs=40, n_batch=100, eta=0.001, lamda=0.1)
# GetResult(case='4', n_epochs=40, n_batch=100, eta=0.001, lamda=1)