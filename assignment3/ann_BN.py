import numpy as np
import time
import pandas as pd

class ANN_BN:

    def __init__(self, eta_min, eta_max, n_s, batch_size, random_state=100, dropout=False) -> None:
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
        final_test_acc : int
            Contains the final accuracy of the validation set
        dropout : False or float
            If false, no dropout is performed, else, dropout is the probability of droping a weight 
        """

        self.random_state = random_state
        self.W = None
        self.b = None
        self.k = None 
        self.d = None
        self.X = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.y_test = None
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.n_s = n_s
        self.batch_size = batch_size
        self.final_test_acc = None
        self.dropout = False 
        assert ((dropout > 0 and dropout <= 1) or dropout==False)
        self.history = {
            "cost_train": [],
            "cost_test" : [],
            "loss_train": [],
            "loss_test" : [],
            "acc_train" : [],
            "acc_test"  : [],
            "eta_list"  : [],
            "iters_list": [],
            "tot_iters" : 0,
        }

    def softmax(self, x):
        """ Standard definition of the softmax function """
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def relu(self, x):
        """ Standard definition of the relu function """
        return np.maximum(0, x)

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
        Y = np.zeros((self.k, n))
        for idx, elem in enumerate(y):
            Y[elem, idx] = 1
        return Y

    def getWeightBias(self, layers):
        """ Initzilize weight and bias from a gaussian distribution """
        np.random.seed(self.random_state)
        
        n_hidden = len(layers)
        self.W = np.empty((n_hidden+1), dtype=object)
        self.b = np.empty((n_hidden+1), dtype=object)

        W = np.random.normal(0, 1/np.sqrt(self.d), size=(layers[0], self.d))
        b = np.zeros((layers[0], 1))
    
        self.W[0] = W
        self.b[0] = b
        last_layer = layers[0]
        for l in range(n_hidden):
            
            if l == n_hidden-1:
                W = np.random.normal(0, 1/np.sqrt(layers[l]), size=(self.k, last_layer))
                b = np.zeros((self.k, 1))
            else:
                 
                W = np.random.normal(0, 1/np.sqrt(layers[l]), size=(layers[l], last_layer))
                b = np.zeros((layers[l], 1))
                last_layer = layers[l]

            self.W[l+1] = W
            self.b[l+1] = b

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

    def evaluateClassifier(self, X_train, dropout=False):
        """ Performs forward-feed returns the probability of each label in X, as well as instanciate Xs"""
        layers = self.W.size
        self.X = np.empty(shape=(layers,), dtype=object)
        self.gammas = np.empty(shape=(layers,), dtype=object)
        self.betas  = np.empty(shape=(layers,), dtype=object)
        # print(f'x shape: {x.shape}')
        self.X[0] = X_train
        self.gammas[0] = np.zeros_like(X_train)
        self.betas[0] = np.ones_like(X_train)
        
        for l in range(layers-1):

            s = self.W[l] @ self.X[l] + self.b[l] 
            self.X[l+1] = self.relu(s)
            self.batchNormForward(self.x[l], self.gammas[l], self.betas[l])
            if dropout:
                u = (np.random.rand(s.shape[0], s.shape[1]) < dropout) / dropout # u: Matrix of size self.X[l+1], consisting of true:s and false:s
                self.X[l+1] = self.X[l+1] * u
        
        s = self.W[layers-1] @ self.X[layers-1] + self.b[layers-1]
        P = self.softmax(s)

        return P

    def batchNormForward(self, x, gamma, beta):
        d, n = x.shape
        mu = x.
        pass    

    def computeCost(self, X, Y, lamda):
        n   = X.shape[1]
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
            # print(f"l: {l}")
            # print(f"grad_W: {grad_W.shape}")
            grad_W[l] = 1/n * (G @ self.X[l].T) + 2 * lamda * self.W[l]    
            grad_b[l] = 1/n * G.sum(axis=1) 
            grad_b[l] = grad_b[l].reshape(-1, 1)
            G = self.W[l].T @ G
            G = G * (self.X[l] > 0).astype(int)

        grad_W[0] = 1/n * G @ self.X[0].T + 2 * lamda * self.W[0]
        grad_b[0] = 1/n * G.sum(axis=1)
        grad_b[0] = grad_b[0].reshape(-1, 1)

        return grad_W, grad_b

    def fit(self, X_train, y_train, X_test, y_test, n_epochs, lamda, layers ,dropout=False, print_every=2, save_every=50, data_augmentation=False,) -> None:
        """
        Fit the model to the input- and ouput data
        ------------------------------------------
        Params:
        -------
        X_train: np.array(dtype=float) of shape (d, n_batch)
            Training samples
        y_train: array of length n_batch
            Training labels
        X_test: np.array(dtype=float) of shape (d, n_batch)
            Test samples
        y_test: array of length n_batch
            Test labels
        n_epochs: int

        lamda: int

        dropout: boolean or float (between 1 and 0)
            Set to false if no dropout should occure. Otherwise Represents the probability of dropping weight node.
        """ 
        self.d = X_train.shape[0]
        self.k = max(y_train) + 1
        self.X_train = X_train
        self.Y_train = self.getOneHot(y_train)
        self.y_train = y_train
        self.X_test = X_test
        self.Y_test = self.getOneHot(y_test)
        self.y_test = y_test
        self.getWeightBias(layers) # Initiate weights and biases
        self.dropout = dropout
        assert ((dropout > 0 and dropout <= 1) or dropout==False) # Makes sure dropout value is valid

        tic = time.perf_counter()

        for epoch in range(n_epochs):   
            
            if epoch % print_every == 0:
                # Print something every n-th epoch
                time_elapsed = time.perf_counter() - tic
                print(f'Epoch number: {epoch}/{n_epochs}, Time elapsed: {int(time_elapsed/60)}min {int(time_elapsed%60)}sec')

            indices = np.arange(0, X_train.shape[1])
            np.random.shuffle(indices)
            X = self.X_train[:, indices]
            Y = self.Y_train[:, indices]        
            y = np.array(y_train)[indices]
            eta = self.miniBatchGD(X, Y, y, save_every, data_augmentation, lamda)
        
        time_elapsed = time.perf_counter() - tic
        self.final_test_acc = self.computeAccuracy(self.X_test, self.y_test)
        print(f'Epoch number: {n_epochs}/{n_epochs}, Time elapsed: {int(time_elapsed/60)}min {int(time_elapsed%60)}sec')

    def miniBatchGD(self, X, Y, y, save_every, data_augmentation, lamda=0):
        """ 
        Performs Minibatch Gradiend Decent         
        ----------------------------------
        Params:
        -------
        X: np.array(dtype=float)
            Input matrix of dimension dxn
        y: array(int)
            Label of corresponding X, list of lenght n
        """ 
        n = X.shape[1]
        n_batch = int(n/self.batch_size)

        for j in range(self.batch_size):
            j_start = int(j*n_batch)
            j_end = int((j+1)*n_batch)
            # inds = np.arange(j_start, j_end).astype(int)
            X_batch = np.copy(X[:, j_start:j_end])
            Y_batch = np.copy(Y[:, j_start:j_end])

            if data_augmentation:
                X_batch = self.get_augmentation_data(X_batch, data_augmentation)

            p = self.evaluateClassifier(X_batch, dropout=self.dropout)
            grad_W, grad_b = self.computeGradients(Y_batch, p, lamda=lamda)

            eta = self.getCyclicETA()
            
            self.W = self.W - eta * grad_W
            self.b = self.b - eta * grad_b
            
            ''' Performs bookkeeping '''
            temp_iter = self.history['tot_iters']
            if (save_every is not None and temp_iter%save_every==0):

                cost_train, loss_train = self.computeCost(self.X_train, self.Y_train, lamda=lamda) 
                cost_test, loss_test = self.computeCost(self.X_test, self.Y_test, lamda=lamda) 
                acc_train = self.computeAccuracy(self.X_train, self.y_train)
                acc_test = self.computeAccuracy(self.X_test, self.y_test)
                self.history['cost_train'].append(cost_train)
                self.history['cost_test'].append(cost_test)
                self.history['loss_train'].append(loss_train)
                self.history['loss_test'].append(loss_test)
                self.history['acc_train'].append(acc_train)
                self.history['acc_test'].append(acc_test)
                temp_iter = self.history['tot_iters']
                self.history['iters_list'].append(temp_iter)

            self.history['tot_iters'] += 1
            self.history['eta_list'].append(eta)

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

    def get_augmentation_data(self, X_train, aug_prob):
        """
        Augment the input data with a probability of aug_prob
        -----------------------------------------------------
        Params:
        -------
        X_train: np.array(dtype=float) of shape (d, n_batch)
            Training samples
        aug_prob: float (between 0 and 1)
            probabilty of augmenting the data
        
        Returns: np.array(dtype=float) of shape (d, n_batch)
            A copy of X_train with some samples of it augmented
        """
        for img_idx, _ in enumerate(X_train.T):
            # augment data sample with a probabilty of aug_prob 
            if aug_prob > np.random.rand(): 
                # Either perform transition- or mirror augmentation  
                rand_shift = tuple(np.random.randint(-3, 3, 2))
                X_train[:,img_idx] = self.translate_image(X_train[:,img_idx], shift=rand_shift) # Shift data sample
            if aug_prob > np.random.rand():
                X_train[:,img_idx] = self.mirror_image(X_train[:,img_idx]) # Mirror data sample

        return X_train

    def mirror_image(self, img):
        """
        Flips an image horizontal
        -----------------------------------
        Params:
        -------
        img: float np.array of shape (3072, )
            array of image that need flipping

        Returns: float np.array of shape (3072, )
            flipped representations of the input image
        """
        flipped_img = np.flip(img.reshape(32,32,3, order="F"), axis=0)
        return flipped_img.reshape(-1, order="F")


    def translate_image(self, img, shift=tuple(np.random.randint(-3, 3, 2))):
        """
        Shift an image
        -----------------------------------
        Params:
        -------
        img: float np.array of shape (3072, )
            array of image that need flipping
        shift: 2-tuple with ints (optional)
            Default is uniformed random integeger between -3 and 3 for both x and y

        Returns: float np.array of shape (3072, )
            flipped representations of the input image
        """
        img = img.reshape(32,32,3, order="F").T
        x_shift, y_shift = shift
        shifted_img = np.roll(img, shift=(x_shift, y_shift), axis=(2,1))
        
        return shifted_img.T.reshape(-1, order="F")


""" For testing purposes """
# from functions import getLargeData, getData, plotLostVEpochs
# import numpy as np

# SEED = 100
# np.random.seed(SEED)
# X_train, Y_train, y_train, X_test, Y_test, y_test = getLargeData(SEED) 
# n = X_train.shape[1]
# X ~ (d, n)

""" Parameters for our ANN model """
# n_batch = 100
# batch_size = int(n/n_batch)
# n_s = 5 * 45000 / n_batch
# cycle = 2 # How many cycles we want (2*n_s = 1 cycle)
# total_iterations = 2 * n_s * cycle 
# n_epochs = int(total_iterations / batch_size)
# np.random.seed(SEED)

# LAMDA = 0.005
# results = {}
# HIDDEN_LAYERS =[10]
# print(f'starting for lambda {LAMDA} and {len(HIDDEN_LAYERS)+1} layers...')
# ann = ANN(
#     eta_min=1e-5, 
#     eta_max=1e-1, 
#     n_s=n_s, 
#     batch_size=batch_size,
#     random_state=SEED,   
#     )
# ann.fit(X_train, y_train, X_test, y_test, layers=HIDDEN_LAYERS ,n_epochs=n_epochs, lamda=LAMDA, print_every=5, save_every=None)
# final_acc = ann.final_test_acc
# print(f'{final_acc} for lambda {LAMDA}')
# results[final_acc] = LAMDA
# DIR = "Result_Pics"

# plotLostVEpochs(
#     ann.history['cost_train'], 
#     ann.history['cost_test'],  
#     ann.history['iters_list'],
#     metric='Cost',
#     x_axis='update steps', 
#     title_string=f'Cost Function\ncycles:{cycle}, n_batch:{n_batch}, lambda:{LAMDA}, test acc:{final_acc}', 
#     dir=DIR, 
#     )
# plotLostVEpochs(
#     ann.history['loss_train'], 
#     ann.history['loss_test'], 
#     ann.history['iters_list'],
#     metric='Loss', 
#     x_axis='update steps', 
#     title_string=f'Loss Function\ncycles:{cycle}, n_batch:{n_batch}, lambda:{LAMDA}, test acc:{final_acc}', 
#     dir=DIR,
#     )
# plotLostVEpochs(
#     ann.history['acc_train'], 
#     ann.history['acc_test'], 
#     ann.history['iters_list'],
#     metric='Acc', 
#     x_axis='update steps', 
#     title_string=f'Accuarcy \ncycles:{cycle}, n_batch:{n_batch}, lambda:{LAMDA}, test acc:{final_acc}', 
#     dir=DIR, 
#     )
# plotLostVEpochs(
#     ann.history['eta_list'], 
#     [], 
#     None,
#     metric='eta', 
#     x_axis='update steps', 
#     title_string=f'Learning Rate (eta) \ncycles:{cycle}, n_batch:{n_batch}, lambda:{LAMDA}, test acc:{final_acc}', 
#     dir=DIR, 
#     )