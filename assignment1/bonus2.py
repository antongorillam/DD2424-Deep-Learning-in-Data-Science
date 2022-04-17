from functions import *

import numpy as np
import numpy.matlib
import pandas as pd

def GetResult(case, n_epochs, n_batch, eta, lamda):
    W_star_bce, b_star_bce = W, b
    W_star_base, b_star_base = W, b

    loss_train_bce = []
    loss_test_bce = []

    loss_train_base = []
    loss_test_base = []
    print(f'Starting case: {case}')
    for epoch in range(n_epochs):
        if epoch % 10 == 0:
            print(f'Epoch number: {epoch}/{n_epochs}')
        # Shuffle the data
        np.random.seed(epoch)
        indices = np.arange(0, X_train.shape[1])
        np.random.shuffle(indices)
        X = X_train[:, indices]
        Y = Y_train[:, indices]
        
        W_star_bce, b_star_bce = MiniBatchGD(X, Y, W_star_bce, b_star_bce, lamda, GDParams=[n_batch, eta, n_epochs], multiple_bce=True)
        _, temp_train_bce =  ComputeCost(X_train, Y_train, W_star_bce, b_star_bce, lamda, get_cost_and_reg=True, multiple_bce=True)
        _, temp_test_bce =  ComputeCost(X_test, Y_test, W_star_bce, b_star_bce, lamda, get_cost_and_reg=True, multiple_bce=True)
        loss_train_bce.append(temp_train_bce)
        loss_test_bce.append(temp_test_bce)    

        W_star_base, b_star_base = MiniBatchGD(X, Y, W_star_base, b_star_base, lamda, GDParams=[n_batch, eta, n_epochs], multiple_bce=False)
        _, temp_train_base =  ComputeCost(X_train, Y_train, W_star_base, b_star_base, lamda, get_cost_and_reg=True, multiple_bce=False)
        _, temp_test_base =  ComputeCost(X_test, Y_test, W_star_base, b_star_base, lamda, get_cost_and_reg=True, multiple_bce=False)
        loss_train_base.append(temp_train_base)
        loss_test_base.append(temp_test_base)    

    acc_train_bce = ComputeAccuracy(X_train, y_train, W_star_bce, b_star_bce, multiple_bce=True)
    acc_test_bce = ComputeAccuracy(X_test, y_test, W_star_bce, b_star_bce, multiple_bce=True)

    acc_train_base = ComputeAccuracy(X_train, y_train, W_star_base, b_star_base, multiple_bce=False)
    acc_test_base = ComputeAccuracy(X_test, y_test, W_star_base, b_star_base, multiple_bce=False)

    print(f'\nFor n_epochs: {n_epochs}, n_batch: {n_batch}, eta: {eta}, lambda: {lamda}')
    print(f'acc_test_bce: {acc_test_bce}')
    print(f'acc_test_base: {acc_test_base}')

    print(f'acc_train_bce: {acc_train_bce}')
    print(f'acc_train_base: {acc_train_base}\n')


    ''' Plot histogram '''
    correct_classified_bce, incorrect_classified_bce = GetClassAccuracy(X_test, y_test, W_star_bce, b_star_bce)
    correct_classified_base, incorrect_classified_base = GetClassAccuracy(X_test, y_test, W_star_base, b_star_base)

    df_correct_bce = pd.DataFrame({'Label':correct_classified_bce, 'Loss Function':['BCE' for _, _ in enumerate(correct_classified_bce)]})
    df_correct_bce.Label.astype(str)
    df_correct_base = pd.DataFrame({'Label':correct_classified_base, 'Loss Function':['Baseline' for _, _ in enumerate(correct_classified_base)]})
    df_correct_base.Label.astype(str)
    df_correct = pd.concat([df_correct_bce, df_correct_base], ignore_index=True) 

    plotHistoGram(
        data=df_correct,
        title_string=f'Histogram for correctly classified labels\ncase: {case}, n_epochs: {n_epochs}, n_batch: {n_batch}, eta: {eta}, lambda: {lamda}',
        case=case,
        )

    df_incorrect_bce = pd.DataFrame({'Label':incorrect_classified_bce, 'Loss Function':['BCE' for _, _ in enumerate(incorrect_classified_bce)]})
    df_incorrect_bce.Label.astype(str)
    df_incorrect_base = pd.DataFrame({'Label':incorrect_classified_base, 'Loss Function':['Baseline' for _, _ in enumerate(incorrect_classified_base)]})
    df_incorrect_base.Label.astype(str)
    df_incorrect =  pd.concat([df_incorrect_bce, df_incorrect_base], ignore_index=True)

    plotHistoGram(
        data=df_incorrect,
        title_string=f'Histogram for incorrectly classified labels\ncase: {case}, n_epochs: {n_epochs}, n_batch: {n_batch}, eta: {eta}, lambda: {lamda}',
        case=case,
        )

    ''' Make line-plot '''
    plotLostVEpochs(
        loss_train=loss_train_bce,
        loss_test=loss_test_bce,
        title_string=f"Case:{case}, Loss Function: BCE\nlambda:{lamda}, epochs:{n_epochs}, batch size:{n_batch}, eta:{eta}, Accuarcy:{acc_test_bce}",
        case=case,
    )

    plotLostVEpochs(
        loss_train=loss_train_base,
        loss_test=loss_test_base,
        title_string=f"Case:{case}, Loss Function: Baseline\nlambda:{lamda}, epochs:{n_epochs}, batch size:{n_batch}, eta:{eta}, Accuarcy:{acc_test_base}",
        case=case,
    )    

    # montage(W_star, case=case, params=[n_epochs, n_batch, eta, lamda])

def plotHistoGram(data, title_string, case):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure()    
    sns.histplot(data=data, x="Label", stat="percent", multiple="dodge", hue="Loss Function", discrete=True, shrink=0.8).set(title=title_string)
    # sns.set(style='whitegrid')
    filename = title_string
    filename = filename.replace(":", "")
    filename = filename.replace(",", "")
    filename = filename.replace("\n", "")
    filename = filename.replace(" ", "_")
    filename = filename.replace(".", "")
    # plt.legend(loc='upper right')
    plt.ylim(0, 8)
    plt.savefig(f'Result_Pics/bonus2/case{case}/' + filename)

def plotLostVEpochs(loss_train, loss_test, title_string, case, y_lim=[0,2]):
	import matplotlib.pyplot as plt

	[y_min, y_max] = y_lim
	plt.figure()
	plt.plot(loss_train, label='Training loss', color='g')
	plt.plot(loss_test, label='Validation loss', color='r')
	plt.title(title_string)
	plt.xlabel("Epochs")	
	plt.ylabel("Loss")
	# plt.ylim(y_min, y_max)
	plt.legend()
	plt.grid()

	filename = title_string
	filename = filename.replace(":", "")
	filename = filename.replace(",", "")
	filename = filename.replace("\n", "")
	filename = filename.replace(" ", "_")
	filename = filename.replace(".", "")        
	plt.savefig(f'Result_Pics/bonus2/case{case}/' + filename)
    
''' 
X: dxn, 3072 x 10000 
Y: Kxn, 10 x 10000
y: n-vector, 10000-vector
W: kxd
'''
np.random.seed(100)
X_train, Y_train, y_train = Load('cifar-10-batches-py/data_batch_1')
X_test, Y_test, y_test = Load('cifar-10-batches-py/test_batch')

# X_train1, Y_train1, y_train1 = Load('cifar-10-batches-py/data_batch_1', show_images=False)
# X_train2, Y_train2, y_train2 = Load('cifar-10-batches-py/data_batch_2', show_images=False)
# X_train3, Y_train3, y_train3 = Load('cifar-10-batches-py/data_batch_3', show_images=False)
# X_train4, Y_train4, y_train4 = Load('cifar-10-batches-py/data_batch_4', show_images=False)
# X_train5, Y_train5, y_train5 = Load('cifar-10-batches-py/data_batch_5', show_images=False)

# X_train = np.concatenate((X_train1, X_train2, X_train3, X_train4, X_train5), axis=1)
# Y_train = np.concatenate((Y_train1, Y_train2, Y_train3, Y_train4, Y_train5), axis=1)
# y_train = np.concatenate((y_train1, y_train2, y_train3, y_train4, y_train5), axis=0)

X_test, Y_test, y_test = Load('cifar-10-batches-py/test_batch')
# X_test = X_test[:,:1000]
# Y_test = Y_test[:,:1000]
# y_test = y_test[:1000] 

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

# dims=3072
# batch_size=10000

# X_train = X_train[:dims,:batch_size]
# Y_train = Y_train[:,:batch_size]
# y_train = y_train[:batch_size]
# W = W[:,:dims]

# p = EvaluateClassifier(X_train, W, b, multiple_bce=False)
# s = np.dot(W , X_train) + b
# grad_W, grad_b = ComputeGradientsMBCE(X_train, Y_train, s, W, lamda=0)
# grad_W, grad_b = ComputeGradients(X_train, Y_train, p, W, lamda=0)
# grad_W_test, grad_b_test = ComputeGradsNum(X_train, Y_train, p, W, b, lamda=0, h=1e-6)
# W_rel_error = CalcRelativeError(grad_W, grad_W_test)
# b_rel_error = CalcRelativeError(grad_b, grad_b_test)

# print(f'W_rel_error for dim {dims}, batch size {batch_size}, lambda {0}: {W_rel_error}')
# print(f'b_rel_error for dim {dims}, batch size {batch_size}, lambda {0}: {b_rel_error}')

GetResult(case=1, n_epochs=100, n_batch=100, eta=0.01, lamda=0)
GetResult(case=2, n_epochs=100, n_batch=100, eta=0.001, lamda=0.1)
GetResult(case=3, n_epochs=100, n_batch=100, eta=0.001, lamda=0.1)
GetResult(case=4, n_epochs=100, n_batch=100, eta=0.001, lamda=1)