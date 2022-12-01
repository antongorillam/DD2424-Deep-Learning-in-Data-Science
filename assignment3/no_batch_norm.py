""" For testing purposes """
from functions import getLargeData, getData, plotLostVEpochs
from ann import ANN
import numpy as np

def run_test(HIDDEN_LAYERS, DIR):
    SEED = 100
    np.random.seed(SEED)
    X_train, Y_train, y_train, X_test, Y_test, y_test = getLargeData(SEED) 
    n = X_train.shape[1]

    """ Parameters for our ANN model """
    n_batch = 100
    batch_size = int(n/n_batch)
    n_s = 5 * 45000 / n_batch
    cycle = 2 # How many cycles we want (2*n_s = 1 cycle)
    total_iterations = 2 * n_s * cycle 
    n_epochs = int(total_iterations / batch_size)
    np.random.seed(SEED)

    LAMDA = 0.005
    results = {}

    print(f'starting for lambda {LAMDA} and {len(HIDDEN_LAYERS)+1} layers...')
    ann = ANN(
        eta_min=1e-5, 
        eta_max=1e-1, 
        n_s=n_s, 
        batch_size=batch_size,
        random_state=SEED,   
        )
    ann.fit(X_train, y_train, X_test, y_test, layers=HIDDEN_LAYERS ,n_epochs=n_epochs, lamda=LAMDA, print_every=5, save_every=100)
    final_acc = ann.final_test_acc
    print(f'{final_acc} for lambda {LAMDA}')
    results[final_acc] = LAMDA

    plotLostVEpochs(
        ann.history['cost_train'], 
        ann.history['cost_test'],  
        ann.history['iters_list'],
        metric='Cost',
        x_axis='update steps', 
        title_string=f'Cost Function (without batch normalization), lambda:{LAMDA}, test acc:{final_acc}\nLayers={HIDDEN_LAYERS}', 
        dir=DIR, 
        )
    plotLostVEpochs(
        ann.history['loss_train'], 
        ann.history['loss_test'], 
        ann.history['iters_list'],
        metric='Loss', 
        x_axis='update steps', 
        title_string=f'Loss Function (without batch normalization), lambda:{LAMDA}, test acc:{final_acc}\nLayers={HIDDEN_LAYERS}', 
        dir=DIR,
        )
    plotLostVEpochs(
        ann.history['acc_train'], 
        ann.history['acc_test'], 
        ann.history['iters_list'],
        metric='Acc', 
        x_axis='update steps', 
        title_string=f'Accuarcy (without batch normalization), lambda:{LAMDA}, test acc:{final_acc}\nLayers={HIDDEN_LAYERS}', 
        dir=DIR, 
        y_lim=[0, 0.7]
        )
    plotLostVEpochs(
        ann.history['eta_list'], 
        [], 
        None,
        metric='eta', 
        x_axis='update steps', 
        title_string=f'Learning Rate (without batch normalization), lambda:{LAMDA}, test acc:{final_acc}\nLayers={HIDDEN_LAYERS}', 
        dir=DIR, 
        )


if __name__ == '__main__':
    run_test(HIDDEN_LAYERS=[50, 50], DIR="Result_Pics/no_batch_norm/")
    # run_test(HIDDEN_LAYERS=[50, 30, 20, 20, 10, 10, 10, 10], DIR="Result_Pics/no_batch_norm/")
    