
import numpy as np
import time

class RNN:
    """
    Class attributes
    ----------------
    random_stat : int
        the random seed for the stochastic moments of the network.
    m : int
        number of hidden-layer (hyperparameter)     
    eta : float
        learning rate (hyperparameter)
    seq_length : int
        length of the inputs sequence     
    b : np.array of dimension (m, 1)
        bias vector    
    c : np.array of dimension (K, 1)
        bias vector
    U : np.array of dimension (m, K)
        weight vector    
    W : np.array of dimension (m, m)
        weight vector
    V : np.array of dimension (K, m)
        weight vector    
    K : int
        Number of unique characters (label)    
    """
    def __init__(self, m, seq_length=25, random_state=100,) -> None:
        self.m = m
        self.seq_length = seq_length
        self.random_state = random_state
        self.book_data = None
        self.char_to_ind = None
        self.ind_to_char = None

        self.params = {
            "b" : None,
            "c" : None,
            "U" : None,
            "W" : None,
            "V" : None,
        }

        self.momentum = {
            "b" : None,
            "c" : None,
            "U" : None,
            "W" : None,
            "V" : None,
        }

        self.K = None   
        self.smooth_loss = None
        self.iterations = 0
        self.history = {
            "smooth_loss" : [],
            "iterations" : [],
            "generated_text" : [],
            "generated_long_text" : [],
        }

        np.random.seed(random_state)

    def softmax(self, x):
        """ Standard definition of the softmax function """

        # s = np.exp(x - np.max(x, axis=0)) / \
        #         np.exp(x - np.max(x, axis=0)).sum(axis=0)
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def readText(self, file_name):
        """ 
        Reads text file         
        ---------------
        Params:
        -------
        file_name: str
            the directory and filename of the textfile
        
        return: array of size 3
            1. Vector containing all characters for the textfile
            2. dict: index -> unique character
            3. dict: unique character -> index  
        """ 
        book_data = []
        with open(file_name, 'r') as file:
            
            for line in file:
                for char in line:
                    char = ' ' if char in ['\n'] else char # Replace new line with space
                    char = '' if char in ['\t'] else char # Remove tabs
                    book_data += char

        book_char = set(book_data)
        K = len(book_char) # Perhaps needed
        self.ind_to_char = dict(enumerate(book_char, 0)) 
        self.char_to_ind = {char: ind for ind, char in self.ind_to_char.items()}
        return book_data

    def computeLoss(self, Y, P):
        """ Standard definition of cross-entropy when Y is one-hot representation  """
        return -np.sum(Y*np.log(P))

    def forward_step(self, xt, h_prev):
        """
        Performs one step in the forward algoritm
        -----------------------------------------
        Params:
        -------
        xt : ndarray of dimension (K, 1)  
            the current character input
        h_prev : ndarray of dimension (m, 1)
            previous hidden state

        returns:
        --------
            pt : ndarray of dimension (K, 1)
                Probabilty of the 
            ot : ndarray of dimension (K, 1)
                output layer at time t
            at : ndarray of dimension (m, 1)
        """
        at = self.params["W"] @ h_prev + self.params["U"] @ xt + self.params["b"]
        ht = np.tanh(at)
        ot = self.params["V"] @ ht + self.params["c"]
        pt = self.softmax(ot)
        
        return at, ht, ot, pt

    def forward(self, X, Y, h_prev, calculate_loss=False):
        """
        Forward pass
        ------------
        X : ndarray of dimension (K, seq_length)
            
        h_prev : ndarray of dimension (m,1)
        
        return:
        -------
        O : ndarray of dimension (K, seq_length)

        loss : float

        H : ndarray of dimension (m, seq_length)
        
        y_pred : ndarray of dimension (K, seq_length)
        """
        loss = 0
        A_list = np.zeros(shape=(self.m, self.seq_length))
        H_list = np.zeros(shape=(self.m, self.seq_length))
        O_list = np.zeros(shape=(self.K, self.seq_length))
        P_list = np.zeros(shape=(self.K, self.seq_length))
        ht = h_prev
        for t in range(self.seq_length):
            
            xt = X.T[t].reshape(-1, 1)
            yt = Y.T[t].reshape(-1, 1)
            at, ht, ot, pt = self.forward_step(xt, ht)
            A_list[:, t] = at.ravel()
            H_list[:, t] = ht.ravel()
            O_list[np.argmax(pt), t] = 1
            P_list[:, t] = pt.ravel()

        loss = self.computeLoss(Y, P_list)

            
        self.smooth_loss = loss if self.smooth_loss==None else self.smooth_loss   
        self.smooth_loss = 0.999 * self.smooth_loss + 0.001 * loss
        # self.history["smooth_loss"].append(self.smooth_loss) 
        # self.history["iterations"].append(self.iterations)

        return A_list, H_list, O_list, P_list, loss

    def backward(self, X, Y, A_list, H_list, O_list, P_list, clipping="value"):
        grads = {
            "V" : [],
            "W" : [],
            "U" : [],
            "b" : [],
            "c" : [],
        }

        grad_o = (P_list-Y).T

        grad_h = np.zeros(shape=(self.seq_length, self.m))
        grad_a = np.zeros(shape=(self.seq_length, self.m))   

        grad_h[-1] = grad_o[-1,:].reshape(1,-1) @ (self.params["V"])
        grad_a[-1] = grad_h[-1,:]  * (1-H_list[:,-1]**2)

        for t in reversed(range(self.seq_length-1)):
            grad_h[t] = grad_o[t,:].reshape(1,-1) @ self.params["V"] + grad_a[t+1] @ self.params["W"]
            grad_a[t] = grad_h[t,:] * (1-H_list[:,t]**2)
        
        temp_h = np.append(np.zeros(shape=(self.m, 1)), H_list[:,:-1], axis=1) # H_list but with zero-vec in the first element
        grads["V"] = grad_o.T @ H_list.T
        grads["W"] = grad_a.T @ temp_h.T
        grads["U"] = grad_a.T @ X.T
        grads["b"] = grad_a.sum(axis=0).reshape(-1, 1)
        grads["c"] = grad_o.sum(axis=0).reshape(-1, 1)   

        h_next = H_list[:,-1].reshape(-1, 1)
        if clipping=="value":
            for key, value in grads.items():
                np.clip(value, -5, 5, out=value) # Performs value clipping

        return grads, h_next

    def generateSentenceHelper(self, p):
        cp = np.cumsum(p)
        a = np.random.rand()
        ixs = np.where(cp-a>0)[0]
        ii = ixs[0]
        return ii

    def generateSequence(self, h0, x0, generated_seq_length=200):
        ht, xt = h0, x0
        Y = np.zeros((self.K, generated_seq_length), dtype=int)
        for seqCnt in range(generated_seq_length):
            _, ht, _, pt = self.forward_step(xt, ht)
            # print(f"ii {ii}")
            ii = self.generateSentenceHelper(pt)
            Y[ii, seqCnt] = 1 
            xt = Y[:, seqCnt].reshape(-1, 1)

        seq_idx = [np.where(Y[:,i])[0][0] for i,_ in enumerate(Y.T)] 
        seq = [self.ind_to_char[word_idx] for word_idx in seq_idx]   
        return ''.join(map(str, seq))
        
    def seq2OneHot(self, seq):
        seq_one_hot = np.zeros((self.K, len(seq)), dtype=int)
        
        for idx, char in enumerate(seq):
            char_idx = self.char_to_ind[char]
            seq_one_hot[char_idx, idx] = 1

        return seq_one_hot

    def fit(self, file_name, epochs, save_every, calculate_loss=100, eta=0.001, sigma=0.01):
        np.random.seed(self.random_state)
        self.book_data = np.array(self.readText(file_name))
        self.K = len(self.char_to_ind) 
        # Initilize the biases
        self.params["b"] = np.zeros((self.m,1))
        self.params["c"] = np.zeros((self.K,1))
        # Initilize the weights
        self.params["U"] = np.random.normal(0, sigma, size=(self.m, self.K))
        self.params["W"] = np.random.normal(0, sigma, size=(self.m, self.m))
        self.params["V"] = np.random.normal(0, sigma, size=(self.K, self.m))
        self.momentum = {
            "b" : np.zeros_like(self.params["b"]),
            "c" : np.zeros_like(self.params["c"]),
            "U" : np.zeros_like(self.params["U"]),
            "W" : np.zeros_like(self.params["W"]),
            "V" : np.zeros_like(self.params["V"]),
        }
        total_iterations = epochs * len([i for i in range(0, self.book_data.size-self.seq_length, self.seq_length)])
        tic = time.perf_counter()

        for epoch in range(epochs):
            h = np.zeros((self.m, 1)) # Initilize h as the zero vector
            e_list = [i for i in range(0, self.book_data.size-self.seq_length, self.seq_length)]
            
            # Performs Stochastic Gradient Decent
            for e in e_list:

                X_chars = self.book_data[e : e+self.seq_length]
                Y_chars = self.book_data[e+1 : e+self.seq_length+1]
                X = self.seq2OneHot(X_chars)        
                Y = self.seq2OneHot(Y_chars) 

                A_list, H_list, O_list, P_list, loss = self.forward(X, Y, h, calculate_loss=calculate_loss)

                if self.iterations % 10000==0:
                    time_elapsed = time.perf_counter() - tic
                    # self.smooth_loss = loss if self.smooth_loss==None else self.smooth_loss   
                    # self.smooth_loss = 0.999 * self.smooth_loss + 0.001 * loss
                    # generated_seq_length = 1000 if self.iterations == total_iterations-1 else 200
                    print(f"Iteration: {self.iterations}/{total_iterations}, epoch: {epoch}/{epochs}, Time elapsed: {int(time_elapsed/60)}min {int(time_elapsed%60)}sec")
                    print(f"Smooth lost: {self.smooth_loss}, eta: {eta}")
                    print('{:.1%} finished...'.format(self.iterations / total_iterations))
                    sentence = self.generateSequence(x0=X[:,0].reshape(-1, 1), h0=h, generated_seq_length=200)
                    long_sentence = self.generateSequence(x0=X[:,0].reshape(-1, 1), h0=h, generated_seq_length=1000)
                    print(f"Generated sentence:\n{sentence}\n")
                    self.history["smooth_loss"].append(self.smooth_loss) 
                    self.history["iterations"].append(self.iterations)
                    self.history["generated_text"].append(str(sentence))
                    self.history["generated_long_text"].append(str(long_sentence))

                grads, h = self.backward(X, Y, A_list, H_list, O_list, P_list)
                
                self.adaGrad(grads, eta)
                self.iterations += 1
                e += self.seq_length
        
        print(f"Iteration: {self.iterations}/{total_iterations}, epoch: {epoch+1}/{epochs}, Time elapsed: {int(time_elapsed/60)}min {int(time_elapsed%60)}sec")
        print(f"Smooth lost: {self.smooth_loss}, eta: {eta}")
        print('{:.1%} finished...'.format(self.iterations / total_iterations))
        sentence = self.generateSequence(x0=X[:,0].reshape(-1, 1), h0=h, generated_seq_length=200)
        long_sentence = self.generateSequence(x0=X[:,0].reshape(-1, 1), h0=h, generated_seq_length=1000)
        print(f"Generated sentence:\n{long_sentence}\n")
        self.history["smooth_loss"].append(self.smooth_loss) 
        self.history["iterations"].append(self.iterations)
        self.history["generated_text"].append(str(sentence))
        self.history["generated_long_text"].append(str(long_sentence))

    def adaGrad(self, grads, eta, error=1e-8):

        for key in grads:
            self.momentum[key] = self.momentum[key] + grads[key]**2
            nom = eta * grads[key]
            denom = np.sqrt(self.momentum[key] + error)
            self.params[key] = self.params[key] - (nom/denom)

    def computeGradsNumSlow(self, X, Y, h):
        """
        Parameters:
            X (K, n): the input matrix
            Y (K, n): the output matrix
            h: initial hidden states
        
        Returns:
            grads: gradient values of all hyper-parameters
        """
        grad_U, grad_W, grad_V, grad_b, grad_c = (
            np.zeros_like(self.params["U"]),
            np.zeros_like(self.params["W"]),
            np.zeros_like(self.params["V"]),
            np.zeros_like(self.params["b"]),
            np.zeros_like(self.params["c"]),
        )
        h0 = np.zeros((self.m, 1))

        b_try = np.copy(self.params["b"])
        for i in range(len(self.params["b"])):
            self.params["b"] = np.array(b_try)
            self.params["b"][i] -= h
            _, _, _, _, c1 = self.forward(X, Y, h0, calculate_loss=1)

            self.params["b"] = np.array(b_try)
            self.params["b"][i] += h
            _, _, _, _, c2 = self.forward(X, Y, h0, calculate_loss=1)

            grad_b[i] = (c2 - c1) / (2 * h)
        self.params["b"] = b_try

        c_try = np.copy(self.params["c"])
        for i in range(len(self.params["c"])):
            self.params["c"] = np.array(c_try)
            self.params["c"][i] -= h
            _, _, _, _, c1 = self.forward(X, Y, h0, calculate_loss=1)

            self.params["c"] = np.array(c_try)
            self.params["c"][i] += h
            _, _, _, _, c2 = self.forward(X, Y, h0, calculate_loss=1)

            grad_c[i] = (c2 - c1) / (2 * h)
        self.params["c"] = c_try

        U_try = np.copy(self.params["U"])
        for i in np.ndindex(self.params["U"].shape):
            self.params["U"] = np.array(U_try)
            self.params["U"][i] -= h
            _, _, _, _, c1 = self.forward(X, Y, h0, calculate_loss=1)

            self.params["U"] = np.array(U_try)
            self.params["U"][i] += h
            _, _, _, _, c2 = self.forward(X, Y, h0, calculate_loss=1)

            grad_U[i] = (c2 - c1) / (2 * h)
        self.params["U"] = U_try

        W_try = np.copy(self.params["W"])
        for i in np.ndindex(self.params["W"].shape):
            self.params["W"] = np.array(W_try)
            self.params["W"][i] -= h
            _, _, _, _, c1 = self.forward(X, Y, h0, calculate_loss=1)

            self.params["W"] = np.array(W_try)
            self.params["W"][i] += h
            _, _, _, _, c2 = self.forward(X, Y, h0, calculate_loss=1)

            grad_W[i] = (c2 - c1) / (2 * h)
        self.params["W"] = W_try

        V_try = np.copy(self.params["V"])
        for i in np.ndindex(self.params["V"].shape):
            self.params["V"] = np.array(V_try)
            self.params["V"][i] -= h
            _, _, _, _, c1 = self.forward(X, Y, h0, calculate_loss=1)

            self.params["V"] = np.array(V_try)
            self.params["V"][i] += h
            _, _, _, _, c2 = self.forward(X, Y, h0, calculate_loss=1)

            grad_V[i] = (c2 - c1) / (2 * h)
        self.params["V"] = V_try

        grads = {
            "U": grad_U,
            "W": grad_W,
            "V": grad_V,
            "b": grad_b,
            "c": grad_c,
        }
        for g in grads:
            grads[g] = np.clip(grads[g], -5, 5)

        return grads

""" Main for testing purposes """
if __name__=='__main__':
    from numpy.testing import (assert_array_equal, assert_array_almost_equal)
    import seaborn as sns
    import pandas as pd
    DIR = 'Result_Pics/'
    m = 100
    rnn = RNN(
        m=m, 
        random_state=100,
    )
    file_name = "Datasets/goblet_book.txt"
    
    rnn.fit(
        file_name, 
        epochs=8, 
        calculate_loss=100, 
        eta=0.1,
        save_every=1000
        )
    
    result_df = pd.DataFrame(rnn.history)
    result_df.to_csv(f'{DIR}results.csv', index=False)
