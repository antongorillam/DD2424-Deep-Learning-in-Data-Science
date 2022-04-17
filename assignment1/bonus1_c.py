from functions import *
import numpy as np
import numpy.matlib
import time 
import pandas as pd
import csv  

def GetResult(n_epochs, n_batch, eta, lamda):
    W_star, b_star = W, b

    for epoch in range(n_epochs):

        # Shuffle the data
        indices = np.arange(0, X_train.shape[1])
        np.random.shuffle(indices)
        X = X_train[:, indices]
        Y = Y_train[:, indices]
        W_star, b_star = MiniBatchGD(X_train, Y_train, W_star, b_star, lamda, GDParams=[n_batch, eta, n_epochs])

    acc_train = ComputeAccuracy(X_train, y_train, W_star, b_star)
    acc_test = ComputeAccuracy(X_test, y_test, W_star, b_star)

    print(f'acc_train: {acc_train}')
    print(f'acc_test: {acc_test}')
    return acc_test

''' 
X: dxn, 3072 x 10000 
Y: Kxn, 10 x 10000
y: n-vector, 10000-vector
W: kxd
'''
np.random.seed(100)
X_train, Y_train, y_train = Load('cifar-10-batches-py/data_batch_1', show_images=False)

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

'''
Grid Search
'''
lamdas = [1, 0.1, 0.01, 0.001, 0.0001]
etas = [1, 0.1,0.01, 0.001, 0.0001]
batch_sizes = [10, 50, 100, 500, 1000, 5000]
result = {}
runs = len(lamdas) * len(etas) * len(batch_sizes)
run = 0
tic = time.perf_counter()

for lamda in lamdas:
    for eta in etas:
        for n_batch in batch_sizes:
            toc = time.perf_counter()
            print(f'At run {run+1}/{runs}, time elapsed: {toc-tic:0.1f} seconds')
            acc = GetResult(n_epochs=40, n_batch=n_batch, eta=eta, lamda=lamda)
            result[acc] = [lamda, eta, n_batch]
            run += 1

sorted_result = [[k]+result[k] for k in sorted(result.keys(), key = lambda x:float(x), reverse=True)] 
print(f'Best result is {sorted_result[0]}')
with open('Result_Pics/bonus1c/grid_search_result.npy', 'wb') as f:
    np.save(f, np.array(sorted_result))

df = pd.DataFrame(sorted_result, dtype='float')
df.to_csv('Result_Pics/bonus1c/grid_search_result.csv', header=None,index=False)
print(df)
# with open(f'Result_Pics/bonus1c/grid_search_result.csv', "wb") as f:
#     writer = csv.writer(f)
#     writer.writerows(sorted_result)

# np.savetxt(f'Result_Pics/bonus1c/grid_search_result.csv', sorted_result, delimiter=',   ')