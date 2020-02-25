import numpy as np
import matplotlib.pyplot as plt
from p_methods import *


def main():
    matrix, y = read_libsvm()
    matrix = numpy_hstack(matrix, y)
    alpha_values = [0.00275, 0.0026]
       
    print("######################### Normalizing ##########################")
    matrix[:,1], matrix[:,2] = normalize(matrix[:,1], matrix[:,2])

    print("########## START TRAINING FOR DIFFERENT LEARNING RATES #########")
    
    for alpha in alpha_values:
        print(f"################## Learning rate alpha={alpha} ##################")
     
        # LEAVE ONE OUT CROSS EVALUATION
        w = looc(matrix, alpha)

        print("###################### Perceptron Weights ######################")
        print(w)
            
        plot_perceptron(matrix, w, alpha)


if __name__ == "__main__":
    main()
