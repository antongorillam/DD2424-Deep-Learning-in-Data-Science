from rnn import RNN 
import numpy as np

def CalcRelativeError(grad_a, grad_b, eps=1e-5):
	rel_error = []
	for a, b in zip(grad_a, grad_b):

		nom = np.linalg.norm(a-b)
		denom = max(eps, np.linalg.norm(a)+np.linalg.norm(b))
		rel_error.append(nom/denom)

	return rel_error

if __name__=='__main__':
    from numpy.testing import (assert_array_equal,assert_array_almost_equal)

    m = 100
    sig = 0.01

    rnn = RNN(
        m = m, 
        random_state=100,
        seq_length=25   
    )
    file_name = "Datasets/goblet_book.txt"
    
    rnn.fit(file_name, sigma=sig, epochs=0)
    book_data = rnn.readText(file_name)
    X_chars = book_data[0 : rnn.seq_length]
    Y_chars = book_data[1 : rnn.seq_length+1]
    X = rnn.seq2OneHot(X_chars)
    Y = rnn.seq2OneHot(Y_chars)
    h0 = np.zeros((rnn.m, 1))
    A_list, H_list, O_list, P_list, loss = rnn.forward(X, Y, h0, calculate_loss=1)
    grads, _ = rnn.backward(X, Y, A_list, H_list, O_list, P_list)

    # grads = {
    #     "V" : [],
    #     "W" : [],
    #     "U" : [],
    #     "b" : [],
    #     "c" : [],
    # }
    
    # grad_o = (P_list-Y).T

    # grad_h = np.zeros(shape=(rnn.seq_length, rnn.m))
    # grad_a = np.zeros(shape=(rnn.seq_length, rnn.m))   

    # grad_h[-1] = grad_o[-1,:].reshape(1,-1) @ (rnn.params["V"])
    # grad_a[-1] = grad_h[-1,:]  * (1-H_list[:,-1]**2)

    # for t in reversed(range(rnn.seq_length-1)):
    #     grad_h[t] = grad_o[t,:].reshape(1,-1) @ rnn.params["V"] + grad_a[t+1] @ rnn.params["W"]
    #     grad_a[t] = grad_h[t,:] * (1-H_list[:,t]**2)
    
    # temp_h = np.append(np.zeros(shape=(rnn.m, 1)), H_list[:,:-1], axis=1) # H_list but with zero-vec in the first element
    # grads["V"] = grad_o.T @ H_list.T
    # grads["W"] = grad_a.T @ temp_h.T
    # grads["U"] = grad_a.T @ X.T
    # grads["b"] = grad_a.sum(axis=0).reshape(-1, 1)
    # grads["c"] = grad_o.sum(axis=0).reshape(-1, 1)   

    # h_next = H_list[:,-1].reshape(-1, 1)

    # for key, value in grads.items():
    #     np.clip(value, -5, 5, out=value) # Performs value clipping

    grads_num = rnn.computeGradsNumSlow(X, Y, h=1e-4)  
    for key, value in grads_num.items():
        np.clip(value, -5, 5, out=value) # Performs value clipping

    b = CalcRelativeError(grads['b'], grads_num['b'])
    c = CalcRelativeError(grads['c'], grads_num['c'])
    V = CalcRelativeError(grads['V'], grads_num['V'])
    W = CalcRelativeError(grads['W'], grads_num['W'])
    U = CalcRelativeError(grads['U'], grads_num['U'])

    print(f'loss: {loss}')
    print(f'Relative error for grad_b: {np.mean(b)} (mean) and {np.max(b)} (max)')
    print(f'Relative error for grad_c: {np.mean(c)} (mean) and {np.max(c)} (max)')
    print(f'Relative error for grad_V: {np.mean(V)} (mean) and {np.max(V)} (max)')
    print(f'Relative error for grad_W: {np.mean(W)} (mean) and {np.max(W)} (max)')
    print(f'Relative error for grad_U: {np.mean(U)} (mean) and {np.max(U)} (max)')
    
    # assert_array_equal(grads['b'], grads_num['b'])
    # assert_array_equal(grads['c'], grads_num['c'])
    # assert_array_equal(grads['V'], grads_num['V'])
    # assert_array_equal(grads['W'], grads_num['W'])
    # assert_array_equal(grads['U'], grads_num['U'])
    