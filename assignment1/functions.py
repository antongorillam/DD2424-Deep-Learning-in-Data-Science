import numpy as np
#from assignment1 import ComputeCost

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def sigmoid(x):
    """ Sigmoid function but independece is assumed """
    return np.exp(x) / (np.exp(x) + 1)

def LoadBatch(filename):
	""" Copied from the dataset website """
	import pickle
	with open('Datasets/'+filename, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def ComputeGradsNum(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));

	c = ComputeCost(X, Y, W, b, lamda);
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2-c) / h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)
			grad_W[i,j] = (c2-c) / h

	return [grad_W, grad_b]

def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h
		c1 = ComputeCost(X, Y, W, b_try, lamda)

		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)

		grad_b[i] = (c2-c1) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] -= h
			c1 = ComputeCost(X, Y, W_try, b, lamda)

			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)

			grad_W[i,j] = (c2-c1) / (2*h)

	return [grad_W, grad_b]

def montage(W, case=0, params=[]):
	""" Display the image for each label in W """
	import matplotlib.pyplot as plt

	fig, ax = plt.subplots(2,5)
	if params != []:
		n_epochs, n_batch, eta, lamda = params
		fig.suptitle(f"Case {case}: lambda {lamda}, epochs {n_epochs}, number bacthes {n_batch} and eta {eta}")

	for i in range(2):
		for j in range(5):
			im  = W[i*5+j,:].reshape(32,32,3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1,0,2)
			ax[i][j].imshow(sim, interpolation='nearest')
			ax[i][j].set_title("y="+str(5*i+j))
			ax[i][j].axis('off')
	if case != 0:
		plt.savefig(f'Result_Pics/case{case}' + '/imgs')
	else:
		plt.show()

def save_as_mat(data, name="model"):
	""" Used to transfer a python model to matlab """
	import scipy.io as sio
	sio.savemat(f'{name}.mat')
    
	
def Load(file_name, show_images=False):
    test_dict = LoadBatch(file_name)
    X = test_dict[b'data'].T
    y = test_dict[b'labels']
    Y = GetOneHot(y)
    if show_images:
        montage(X.T, case=0)
    return X, Y, y

def EvaluateClassifier(X, W, b, multiple_bce=False):
	n = X.shape[1]
	s = np.dot(W , X) + b

	if multiple_bce:
		return sigmoid(s)
	else:
		return softmax(s)

def GetOneHot(y):
    k = max(y) + 1
    d = len(y)
    Y = np.zeros((k,d))
    for idx, elem in enumerate(y):
        Y[elem, idx] = 1
    return Y

def ComputeCost(X, Y, W, b, lamda, get_cost_and_reg=False, multiple_bce=False):
	n = X.shape[1]
	p = EvaluateClassifier(X, W, b, multiple_bce=multiple_bce)

	if multiple_bce:
		loss_cross = -np.mean((1-Y) * np.log(1-p) + Y*np.log(p))
		reg_term = lamda * np.sum(W**2)	
	else:
		loss_cross = 1/n * -np.sum(Y * np.log(p))
		reg_term = lamda * np.sum(W**2)
	
	if get_cost_and_reg:
		return loss_cross + reg_term, loss_cross 
	else:
		return loss_cross + reg_term

def ComputeAccuracy(X, y, W, b, multiple_bce=False):
	n = X.shape[1]
	p = EvaluateClassifier(X, W, b, multiple_bce)
	y_pred = np.argmax(p, axis=0)
	acc = np.mean(y_pred==y)
	return acc

def ComputeGradients(X, Y, P, W, lamda):
	n = X.shape[1] 
	G = P-Y    
	grad_W = 1/n * (G @ X.T) + 2 * lamda * W
	grad_b = 1/n * G.sum(axis=1)
	grad_b = grad_b.reshape(-1, 1)
	return grad_W, grad_b

def ComputeGradientsMBCE(X, Y, s, W, lamda):
	'''
	Inspired by: https://medium.com/@andrewdaviesul/chain-rule-differentiation-log-loss-function-d79f223eae5
	'''
	n = X.shape[1]
	k = Y.shape[0]

	G = sigmoid(s) - Y
	grad_W = 1/n * (G @ X.T) + 2 * lamda * W
	grad_b = 1/n * G.sum(axis=1)
	grad_b = grad_b.reshape(-1, 1)
	return grad_W, grad_b

def CalcRelativeError(grad_a, grad_b, eps=1e-6):
    nom = np.linalg.norm(grad_a-grad_b)
    denom = max(eps, np.linalg.norm(grad_a)+np.linalg.norm(grad_b))
    return nom/denom

def GetClassAccuracy(X, ys, W, b):
	n = X.shape[1]
	k = max(ys) + 1

	p = EvaluateClassifier(X, W, b)
	ys_pred = np.argmax(p, axis=0)	

	class_correct = []
	class_incorrect = []

	for y, y_pred in zip(ys, ys_pred):
		if y==y_pred:
			class_correct.append(y)
		else:
			class_incorrect.append(y_pred) 

	return class_correct, class_incorrect

def MiniBatchGD(X, Y, W, b, lamda=0, GDParams=[100, 0.001, 20], multiple_bce=False):

	n_batch, eta, n_epochs = GDParams
	n = X.shape[1]
	batch_size = int(n/n_batch)
	
	for j in range(batch_size):
		
		j_start = int(j*n_batch)
		j_end = int((j+1)*n_batch - 1)
		inds = np.arange(j_start, j_end).astype(int)
		X_batch = X[:, j_start:j_end]
		Y_batch = Y[:, j_start:j_end]
		if multiple_bce:
			# p = EvaluateClassifier(X_batch, W, b, multiple_bce=True)
			s = np.dot(W , X_batch) + b
			grad_W, grad_b = ComputeGradientsMBCE(X_batch, Y_batch, s, W, lamda=lamda)
		else:
			p = EvaluateClassifier(X_batch, W, b, multiple_bce=False)
			grad_W, grad_b = ComputeGradients(X_batch, Y_batch, p, W , lamda=lamda)

		W = W - eta * grad_W
		b = b - eta * grad_b
	
	return W, b
