import numpy as np
#from assignment1 import ComputeCost

def LoadBatch(filename):
	""" Copied from the dataset website """
	import pickle
	with open('../assignment1/Datasets/'+filename, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def softmax(x):
	""" Standard definition of the softmax function """
	return np.exp(x) / np.sum(np.exp(x), axis=0)

def relu(x):
	""" Standard definition of the relu function """
	return np.maximum(0, x)

def montage(W):
	""" Display the image for each label in W """
	import matplotlib.pyplot as plt

	fig, ax = plt.subplots(2,5)

	for i in range(2):
		for j in range(5):
			im  = W[i*5+j,:].reshape(32,32,3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1,0,2)
			ax[i][j].imshow(sim, interpolation='nearest')
			ax[i][j].set_title("y="+str(5*i+j))
			ax[i][j].axis('off')

	plt.show()

def ComputeGradsNum(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	# grad_W = np.zeros(W.shape);
	# grad_b = np.zeros((no, 1));

	grad_W = np.empty(shape=W.shape, dtype=object)
	grad_b = np.empty(shape=b.shape, dtype=object)

	c = ComputeCost(X, Y, W, b, lamda)
	
	for j in range(len(b)):
		grad_b[j] = np.zeros(shape=(b[j].shape))

		for i in range(len(b[j])):
			b_try = np.array(b)
			b_try[j][i] += h
			c2 = ComputeCost(X, Y, W, b_try, lamda)
			grad_b[j][i] = (c2-c) / h
	
	for k in range(len(W)):
		grad_W[k] = np.zeros(shape=(W[k].shape))

		for i in range(W[k].shape[0]):
			for j in range(W[k].shape[1]):
				W_try = np.array(W)
				W_try[k][i,j] += h
				c2 = ComputeCost(X, Y, W_try, b, lamda)

				grad_W[k][i,j] = (c2-c) / h

	return [grad_W, grad_b]

def ComputeGradsNumSlow(X, Y, W1, b1, W2, b2, lamda, h):

	dL_dW1 = np.zeros(shape=W1.shape)
	dL_db1 = np.zeros(shape=b1.shape)

	dL_dW2 = np.zeros(shape=W2.shape)
	dL_db2 = np.zeros(shape=b2.shape)


	for i in range(b1.shape[0]):
		b1_try = b1.copy()
		b1_try[i, 0] = b1_try[i, 0] - h
		c1, _ = ComputeCost(X, Y, np.array([W1, W2], dtype=object), np.array([b1_try, b2], dtype=object), lamda)

		b1_try = b1.copy()
		b1_try[i, 0] = b1_try[i, 0] + h
		c2, _ = ComputeCost(X, Y, np.array([W1, W2], dtype=object), np.array([b1_try, b2], dtype=object), lamda)

		dL_db1[i, 0] = (c2 - c1) / (2 * h)

	for i in range(W1.shape[0]):
		for j in range(W1.shape[1]):
			W1_try = W1.copy()
			W1_try[i, j] = W1_try[i, j] - h
			c1, _ = ComputeCost(X, Y, np.array([W1_try, W2], dtype=object), np.array([b1, b2], dtype=object), lamda)
			
			W1_try = W1.copy()
			W1_try[i, j] = W1_try[i, j] + h
			c2, _ = ComputeCost(X, Y, np.array([W1_try, W2], dtype=object), np.array([b1, b2], dtype=object), lamda)
			dL_dW1[i, j] = (c2 - c1) / (2*h)

	for i in range(b2.shape[0]):
		b2_try = b2.copy()
		b2_try[i, 0] = b2_try[i, 0] - h
		c1, _ = ComputeCost(X, Y, np.array([W1, W2], dtype=object), np.array([b1, b2_try], dtype=object), lamda)

		b2_try = b2.copy()
		b2_try[i, 0] = b2_try[i, 0] + h
		c2, _ = ComputeCost(X, Y, np.array([W1, W2], dtype=object), np.array([b1, b2_try], dtype=object), lamda)
		dL_db2[i, 0] = (c2 - c1) / (2*h)

	for i in range(W2.shape[0]):
		for j in range(W2.shape[1]):
			W2_try = W2.copy()
			W2_try[i, j] = W2_try[i, j] - h
			c1, _ = ComputeCost(X, Y, np.array([W1, W2_try], dtype=object), np.array([b1, b2], dtype=object), lamda)

			W2_try = W2.copy()
			W2_try[i, j] = W2_try[i, j] + h
			c2, _ = ComputeCost(X, Y, np.array([W1, W2_try], dtype=object), np.array([b1, b2], dtype=object), lamda)
			
			dL_dW2[i, j] = (c2 - c1) / (2*h)

	return np.array([dL_dW1, dL_dW2], dtype=object), np.array([dL_db1, dL_db2], dtype=object)

def montage(W, case=0, dir=None):
	""" Display the image for each label in W """
	import matplotlib.pyplot as plt

	fig, ax = plt.subplots(2,5)

	for i in range(2):
		for j in range(5):
			im  = W[i*5+j,:].reshape(32,32,3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1,0,2)
			ax[i][j].imshow(sim, interpolation='nearest')
			ax[i][j].set_title("y="+str(5*i+j))
			ax[i][j].axis('off')
	if case != 0:
		plt.savefig(dir + f'{case}_imgs')
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

def EvaluateClassifier(X, W, b):
	k = W.size
	
	# H = np.maximum(W[0] @ X + b[0], 0)
	# P = softmax(W[1] @ H + b[1])
	X_temp = np.empty(shape=(k,), dtype=object)
	X_temp[0] = X
	for l in range(k-1):
		
		s = np.dot(W[l], X_temp[l]) + b[l] 
		X_temp[l+1] = np.maximum(s, 0) # Relu
		
	s = np.dot(W[k-1], X_temp[k-1]) + b[k-1]
	return softmax(s), X_temp
	# return P, H

def GetOneHot(y):
    k = max(y) + 1
    d = len(y)	
    Y = np.zeros((k,d))
    for idx, elem in enumerate(y):
        Y[elem, idx] = 1
    return Y

def ComputeCost(X, Y, W, b, lamda):
	n = X.shape[1]
	p, _ = EvaluateClassifier(X, W, b)
	loss = 1/ n * np.sum(-Y * np.log(p))
	# print(f'reg {sum([sum(w) for w in [sum(ws**2)for ws in W]])}') 
	reg = lamda * sum([sum(w) for w in [sum(ws**2)for ws in W]])
	cost = loss + reg

	return cost, loss

def ComputeAccuracy(X, y, W, b):

	n = X.shape[1]
	p, _ = EvaluateClassifier(X, W, b)
	y_pred = np.argmax(p, axis=0)
	acc = np.mean(y_pred==y)
	return acc

def ComputeGradients(X, Y, P, W, lamda, H=None):

	n = X[0].shape[1]
	k = X.size
	grad_W = np.empty(shape=W.shape, dtype=object)
	grad_b = np.empty(shape=W.shape, dtype=object)
	G = P-Y   
	
	for l in range(k-1, 0, -1):

		grad_W[l] = 1/n * (G @ X[l].T) + 2 * lamda * W[l]    
		grad_b[l] = 1/n * G.sum(axis=1) 
		grad_b[l] = grad_b[l].reshape(-1, 1)
		G = W[l].T @ G
		G = G * (X[l] > 0).astype(int)

	grad_W[0] = 1/n * G @ X[0].T + 2 * lamda * W[0]
	grad_b[0] = 1/n * G.sum(axis=1)
	grad_b[0] = grad_b[0].reshape(-1, 1)

	return grad_W, grad_b

def CalcRelativeError(grad_a, grad_b, eps=1e-5):
	rel_error = []
	for a, b in zip(grad_a, grad_b):

		nom = np.linalg.norm(a-b)
		denom = max(eps, np.linalg.norm(a)+np.linalg.norm(b))
		rel_error.append(nom/denom)
	return rel_error

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

def GetWeightBias(k, d, m, seed=100):
	np.random.seed(seed)

	W1 = np.random.normal(0, 1/np.sqrt(d), size=(m, d))
	b1 = np.zeros((m,1))
	W2 = np.random.normal(0, 1/np.sqrt(m), size=(k, m))
	b2 = np.zeros((k,1))
	# Cell = np.array([[W1, W2], [b1,b2]]) # Not sure if needed
	return np.array([W1, W2], dtype=object), np.array([b1, b2], dtype=object)

def MiniBatchGD(X, Y, W, b, total_iteations, lamda=0, GDParams=[100, 20, 1e-5, 1e-1]):

	n_batch, n_epochs, eta_min, eta_max  = GDParams
	n = X.shape[1]
	batch_size = int(n/n_batch)
	eta_list = []

	for j in range(batch_size):
		j_start = int(j*n_batch)
		j_end = int((j+1)*n_batch - 1)
		# inds = np.arange(j_start, j_end).astype(int)
		X_batch = X[:, j_start:j_end]
		Y_batch = Y[:, j_start:j_end]

		p, Xs = EvaluateClassifier(X_batch, W, b)
		grad_W, grad_b = ComputeGradients(Xs, Y_batch, p, W , lamda=lamda)


		eta = GetCyclicETA(iterations=total_iteations, eta_min=1e-5, eta_max=1e-1, n_s=500)
		total_iteations += 1
		
		eta_list.append(eta)
		W = W - eta * grad_W
		b = b - eta * grad_b
	
	return W, b, total_iteations, eta_list

def GetCyclicETA(eta_min, eta_max, n_s):
	'''
	Calculate the learning rate eta for Cyclic Learning
	Equation inspired by: https://www.jeremyjordan.me/nn-learning-rate/
	'''
	from math import floor
	cycle = floor(1 + iterations/(2*n_s))
	x = abs(iterations/n_s - 2*(cycle) + 1)
	
	return eta_min + (eta_max - eta_min) * (1-x)

def getData(seed):
	import numpy.matlib
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

def getLargeData(seed):
	import numpy.matlib
	np.random.seed(seed)
	X_train1, Y_train1, y_train1 = Load('cifar-10-batches-py/data_batch_1', show_images=False)
	X_train2, Y_train2, y_train2 = Load('cifar-10-batches-py/data_batch_2', show_images=False)
	X_train3, Y_train3, y_train3 = Load('cifar-10-batches-py/data_batch_3', show_images=False)
	X_train4, Y_train4, y_train4 = Load('cifar-10-batches-py/data_batch_4', show_images=False)
	X_train5, Y_train5, y_train5 = Load('cifar-10-batches-py/data_batch_5', show_images=False)

	X_train = np.concatenate((X_train1, X_train2, X_train3, X_train4, X_train5), axis=1)
	Y_train = np.concatenate((Y_train1, Y_train2, Y_train3, Y_train4, Y_train5), axis=1)
	y_train = np.concatenate((y_train1, y_train2, y_train3, y_train4, y_train5), axis=0)

	X_test, Y_test, y_test = Load('cifar-10-batches-py/test_batch')
	X_test = X_test[:,:5000]
	Y_test = Y_test[:,:5000]
	y_test = y_test[:5000] 

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
	
	return X_train, Y_train, y_train, X_test, Y_test, y_test 

def plotLostVEpochs(train, test, iters_list, metric, x_axis, title_string, dir, y_lim=None):
	import matplotlib.pyplot as plt

	plt.figure()
	if test != []:
		plt.plot(iters_list, train, label=f'Training {metric}', color='g')
		plt.plot(iters_list, test, label=f'Validation {metric}', color='r')
	else:
		plt.plot(train, label=f'{metric}', color='r')

	if y_lim != None:
		[y_min, y_max] = y_lim
		plt.ylim(y_min, y_max)
	else: 
		plt.ylim(bottom=0)

	plt.title(title_string)
	plt.xlabel(x_axis)	
	plt.ylabel(metric)
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

def plotBarAccVsHidden(n_hidden, acc, y_label, x_label, title_string, dir, y_lim=[0,1]):

	import matplotlib.pyplot as plt
	plt.figure()
	plt.bar(x=n_hidden, height=acc)
	plt.title(title_string)
	plt.xlabel(x_label)	
	plt.ylabel(y_label)
	plt.ylim(y_lim)
	filename = title_string
	filename = filename.replace(":", "")
	filename = filename.replace(",", "")
	filename = filename.replace("\n", "")
	filename = filename.replace(" ", "_")
	filename = filename.replace(".", "")
	filename = filename = filename + ".png"
	plt.savefig(dir + filename)