
def read_text(file_name):
    import numpy as np
    """ 
    Reads text file         
    ----------------------------------
    Params:
    -------
    file_name: str
        the directory and filename of the textfile
    
    return: array of size 3
        1. Vector containing all characters for the textfile
        2. dict: index -> unique character
        3. dict: unique character -> index  
    """ 
    char_vec = []
    with open(file_name, 'r') as file:
        
        for line in file:
            for word in line:
                word = ' ' if word in ['\n'] else word # Replace new line with space
                word = '' if word in ['\t'] else word # Remove tabs
                char_vec += word
    char_to_ind = dict(enumerate(set(char_vec), 0)) 
    ind_to_char = {ind: char for char, ind in char_to_ind.items()}
    return {
        'char_vec': np.array(char_vec), 
        'char_to_ind': char_to_ind, 
        'ind_to_char': ind_to_char
        }


""" Main for testing purposes """
if __name__=='__main__':
    import pandas as pd
    
    file_name = 'Datasets/goblet_book.txt'
    data_dict = read_text(file_name)
    char_vec, char_to_ind, ind_to_char = data_dict['char_vec'], data_dict['char_to_ind'], data_dict['ind_to_char'] 
    print(char_vec[0][-100:-1])