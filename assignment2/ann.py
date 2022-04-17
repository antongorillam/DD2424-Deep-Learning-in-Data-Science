import numpy as np
import time

class ANN:

    def __init__(self, m, eta_min, eta_max, n_s, batch_size, random_state=100) -> None:
        """
        Class attributes
        ----------------
        random_stat : int
            the random seed for the stochastic moments of the network.
        W : np.array(dtype=object) 
            The weight matrices,  W = [W0, W1] where W0~(mxd), W1~(kxm) 
        b : np.array(dtype=object)
            The bias vectors,  b = [b0, b1] where b0~(mx1), b0~(kx1), 
        k : int
            number of labels
        d : int
            number of dimensions for each data (Number of pixels in RBG, 3x32x32 for our case)     
        m : int
            number of hidden-layer (hyperparameter) 
        n : int
            number of training data 
        eta_min : int
            the minimum learning rate during cyclic learning 
        eta_max : int
            the maximum learning rate during cyclic learning 
        n_s : int 
            half of cycle during cyclic learning, so if n_s=500, 
            then a full cycle will be reached at 1000 iterations 
        batch_size : int 
            what size of batch size to train during mini-batch SD  
        history: dict
            bookkeeping of desirible fact over each iterations
        """

        self.random_state = random_state
        self.m = m 
        self.W = None
        self.b = None
        self.k = None 
        self.d = None
        self.X = None
        self.X_train = None
        self.Y_train = None
        self.y_train = None
        self.X_test = None
        self.Y_test = None
        self.y_test = None
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.n_s = n_s
        self.batch_size = batch_size

        self.history = {
            "cost_train": [],
            "cost_test" : [],
            "loss_train": [],
            "loss_test" : [],
            "acc_train" : [],
            "acc_test"  : [],
            "eta_list"  : [],
            "tot_iters" : 0,
        }

    def softmax(self, x):
        """ Standard definition of the softmax function """
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def relu(self, x):
        """ Standard definition of the relu function """
        return np.maximum(0, x)

    def sigmoid(self, x):
        """ Sigmoid function but independece is assumed """
        return np.exp(x) / (np.exp(x) + 1)

    def getOneHot(self, y):
        """
        Get one-hot representation of labels
        Params:
        -------
            y: array
                Array containing all the labels y~(n,)
        Returns:
            Y: np.array(dtype=float)
                Matrix of shape Y~(k,n)
        """	
        n = len(y)
        Y = np.zeros((k, n))
        for idx, elem in enumerate(y):
            Y[elem, idx] = 1
        return Y

    def getWeightBias(self):
        """ Initzilize weight and bias from a gaussian distribution """
        np.random.seed(self.random_state)
        W1 = np.random.normal(0, 1/np.sqrt(self.d), size=(self.m, self.d))
        b1 = np.zeros((self.m, 1))
        W2 = np.random.normal(0, 1/np.sqrt(self.m), size=(self.k, self.m))
        b2 = np.zeros((self.k, 1))
        
        self.W = np.array([W1, W2], dtype=object)
        self.b = np.array([b1, b2], dtype=object)
    

    def computeAccuracy(self, X, y):
        """ 
        Calculate the accuarcy of the model
        -----------------------------------
        Params:
        -------
        X: np.array(dtype=float)
            Input matrix of dimension dxn
        y: array(int)
            Label of corresponding X, list of lenght n
        Returns:
            acc: float
                number representing the percentage of corretly classified labels
        """ 
        p = self.evaluateClassifier(X)
        y_pred = np.argmax(p, axis=0)
        acc = np.mean(y_pred==y)
        
        return acc

    def evaluateClassifier(self, X_train):
        """ Performs forward-feed returns the probability of each label in X, as well as instanciate Xs"""
        layers = self.W.size
        X = np.empty(shape=(layers,), dtype=object)
        X[0] = X_train

        for l in range(layers-1):
            
            s = np.dot(self.W[l], X[l]) + self.b[l] 
            X[l+1] = self.relu(s) 
        
        s = np.dot(self.W[layers-1], X[layers-1]) + self.b[layers-1]
        P = softmax(s)
        self.X = X
        return P

    def computeCost(self, X, Y, lamda):
        n = X.shape[1]

        p = self.evaluateClassifier(X)
        
        loss = 1/ n * np.sum(-Y * np.log(p))
        reg = lamda * sum([sum(w) for w in [sum(ws**2)for ws in self.W]])
        cost = loss + reg

        return cost, loss


    def computeGradients(self, Y, P, lamda):
        """ Performs backwards-pass an calculate the gradient for weight and biases """
        layers = self.W.size 
        n = Y.shape[1] # For readability purposes
        grad_W = np.empty(shape=self.W.shape, dtype=object)
        grad_b = np.empty(shape=self.b.shape, dtype=object)
        G = P-Y   
        
        for l in range(layers-1, 0, -1):

            grad_W[l] = 1/n * (G @ self.X[l].T) + 2 * lamda * self.W[l]    
            grad_b[l] = 1/n * G.sum(axis=1) 
            grad_b[l] = grad_b[l].reshape(-1, 1)
            G = self.W[l].T @ G
            G = G * (self.X[l] > 0).astype(int)

        grad_W[0] = 1/n * G @ self.X[0].T + 2 * lamda * self.W[0]
        grad_b[0] = 1/n * G.sum(axis=1)
        grad_b[0] = grad_b[0].reshape(-1, 1)

        return grad_W, grad_b

    def fit(self, X_train, y_train, X_test, y_test, n_epochs, lamda):
        """ Fit the model to the input- and ouput data """
        self.d = X_train.shape[0]
        self.k = max(y_train) + 1
        self.X_train = X_train
        self.Y_train = self.getOneHot(y_train)
        self.y_train = y_train
        self.X_test = X_test
        self.Y_test = self.getOneHot(y_test)
        self.y_test = y_test
        self.getWeightBias() # Initiate weights and biases
        tic = time.perf_counter()

        for epoch in range(n_epochs):
            
            if epoch % 5 == 0:
                # Print something every n-th epoch
                time_elapsed = time.perf_counter() - tic
                print(f'Epoch number: {epoch}/{n_epochs}, Time elapsed: {int(time_elapsed/60)}min {int(time_elapsed%60)}sec')

            indices = np.arange(0, X_train.shape[1])
            np.random.shuffle(indices)
            X = X_train[:, indices]
            Y = Y_train[:, indices]        
            y = np.array(y_train)[indices]
            eta = self.miniBatchGD(X, Y, y, lamda)
        
        time_elapsed = time.perf_counter() - tic
        self.final_test_acc = self.computeAccuracy(self.X_test, self.y_test)
        print(f'Epoch number: {n_epochs}/{n_epochs}, Time elapsed: {int(time_elapsed/60)}min {int(time_elapsed%60)}sec')

    def miniBatchGD(self, X, Y, y, lamda=0):
        """ Performs Minibatch Gradiend Decent """
        n = X.shape[1]
        n_batch = int(n/self.batch_size)

        for j in range(self.batch_size):
            j_start = int(j*n_batch)
            j_end = int((j+1)*n_batch)
            # inds = np.arange(j_start, j_end).astype(int)
            X_batch = X[:, j_start:j_end]
            Y_batch = Y[:, j_start:j_end]
            y_batch = y[j_start:j_end]

            p = self.evaluateClassifier(X_batch)
            grad_W, grad_b = self.computeGradients(Y_batch, p, lamda=lamda)

            eta = self.getCyclicETA()
            
            self.W = self.W - eta * grad_W
            self.b = self.b - eta * grad_b
            
            ''' Performs bookkeeping '''
            self.history['tot_iters'] += 1
            self.history['eta_list'].append(eta)

            if j%50==0:

                cost_train, loss_train = self.computeCost(X_batch, Y_batch, lamda=lamda) 
                cost_test, loss_test = self.computeCost(self.X_test, self.Y_test, lamda=lamda) 
                acc_train = self.computeAccuracy(X_batch, y_batch)
                acc_test = self.computeAccuracy(self.X_test, self.y_test)
                self.history['cost_train'].append(cost_train)
                self.history['cost_test'].append(cost_test)
                self.history['loss_train'].append(loss_train)
                self.history['loss_test'].append(loss_test)
                self.history['acc_train'].append(acc_train)
                self.history['acc_test'].append(acc_test)

        cost_train, loss_train = self.computeCost(X_batch, Y_batch, lamda=lamda) 
        cost_test, loss_test = self.computeCost(self.X_test, self.Y_test, lamda=lamda) 
        acc_train = self.computeAccuracy(X_batch, y_batch)
        acc_test = self.computeAccuracy(self.X_test, self.y_test)
        self.history['cost_train'].append(cost_train)
        self.history['cost_test'].append(cost_test)
        self.history['loss_train'].append(loss_train)
        self.history['loss_test'].append(loss_test)
        self.history['acc_train'].append(acc_train)
        self.history['acc_test'].append(acc_test)

        return eta

    def getCyclicETA(self):
        '''
        Calculate the learning rate eta for Cyclic Learning
        Equation inspired by: https://www.jeremyjordan.me/nn-learning-rate/
        '''
        from math import floor
        current_iteration = self.history['tot_iters']
        cycle = floor(1 + current_iteration/(2*self.n_s))
        x = abs(current_iteration/self.n_s - 2*(cycle) + 1)
        new_eta = self.eta_min + (self.eta_max - self.eta_min) * (1-x)
        return new_eta 

""" For testing purposes """
from functions import *
SEED = 1000
np.random.seed(SEED)
X_train, Y_train, y_train = Load('cifar-10-batches-py/data_batch_1', show_images=False)
X_test, Y_test, y_test = Load('cifar-10-batches-py/data_batch_2')

X_train, Y_train, y_train, X_test, Y_test, y_test = getData(SEED) 

k = Y_train.shape[0]
d = X_train.shape[0]
m = 50

# dims = 20
# batch_size = 1
# LAMDA = 0

# X_train = X_train[:dims,:batch_size]
# Y_train = Y_train[:,:batch_size]
# y_train = y_train[:batch_size]

N_EPOCHS = 10
LAMDA = 0.01
DIR = 'Result_Pics/exercise3/'
n = X_train.shape[1] 
ann = ANN(
    m=50, 
    eta_min=1e-5, 
    eta_max=1e-1, 
    n_s=500, 
    batch_size=100,
    )
ann.fit(X_train, y_train, X_test, y_test, n_epochs=N_EPOCHS, lamda=LAMDA)
final_acc = ann.history['acc_test'][-1]
plotLostVEpochs(
    ann.history['cost_train'], 
    ann.history['cost_test'], metric='Cost', 
    x_axis='update steps', 
    title_string=f'Cost Function\nn_epochs:{N_EPOCHS}, n_batch:{n/ann.batch_size}, lambda:{LAMDA}, test acc:{final_acc}', dir=DIR)
plotLostVEpochs(
    ann.history['loss_train'], 
    ann.history['loss_test'], metric='Loss', 
    x_axis='update steps', 
    title_string=f'Loss Function\nn_epochs:{N_EPOCHS}, n_batch:{n/ann.batch_size}, lambda:{LAMDA}, test acc:{final_acc}', dir=DIR)
plotLostVEpochs(
    ann.history['acc_train'], 
    ann.history['acc_test'], 
    metric='Acc', 
    x_axis='update steps', 
    title_string=f'Accuarcy \nn_epochs:{N_EPOCHS}, n_batch:{n/ann.batch_size}, lambda:{LAMDA}, test acc:{final_acc}', dir=DIR)
plotLostVEpochs(
    ann.history['eta_list'], 
    [], 
    metric='eta', 
    x_axis='update steps', 
    title_string=f'Learning Rate (eta) \nn_epochs:{N_EPOCHS}, n_batch:{n/ann.batch_size}, lambda:{LAMDA}, test acc:{final_acc}', dir=DIR)


# p = ann.evaluateClassifier(X_train)
# grad_W, grad_b = ann.computeGradients(Y_train, p ,lamda=LAMDA)

# grad_num_W, grad_num_b = ComputeGradsNumSlow(
#     X_train, 
#     Y_train,
#     ann.W[0],
#     ann.b[0], 
#     ann.W[1], 
#     ann.b[1], 
#     lamda=0, 
#     h=1e-6
#     )


# W_rel_error = CalcRelativeError(grad_W, grad_num_W)
# b_rel_error = CalcRelativeError(grad_b, grad_num_b)

# print(f'W_rel_error for dim {dims}, batch size {batch_size}, lambda {LAMDA}: {W_rel_error}')
# print(f'b_rel_error for dim {dims}, batch size {batch_size}, lambda {LAMDA}: {b_rel_error}')

    