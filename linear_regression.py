import random
import numpy as np
import matplotlib.pyplot as plt
from lr_methods import *


def main():
    X_en, y_en = read_tsv('/Users/sofiflink/Skola/Pågående Kurser/Artificiell Intelligens EDAP01/machine-learning/salammbo_a_en.tsv')
    X_fr, y_fr = read_tsv('/Users/sofiflink/Skola/Pågående Kurser/Artificiell Intelligens EDAP01/machine-learning/salammbo_a_fr.tsv')
    X_en, y_en = numpy_array(X_en, y_en) 
    X_fr, y_fr = numpy_array(X_fr, y_fr)
    alpha_values = [1.0, 0.2, 0.1, 0.05, 1.4, 1.5]

    print("######################### Normalizing - English ##########################")
    X_en[:,1], y_en = normalize(X_en[:,1], y_en)
    print("########################## Normalizing - French ##########################")
    X_fr[:,1], y_fr = normalize(X_fr[:,1], y_fr)
    
    for alpha in alpha_values:
        print("Learning rate alpha: ", alpha)
        # ################### Batch Gradient Descent - English #####################
        wb_en = plot_bgd(X_en, y_en, alpha, 500, "Batch gradient descent: Weights (English, alpha: " + str(alpha) + ")")

        # #################### Batch Gradient Descent - French #####################
        wb_fr = plot_bgd(X_fr, y_fr, alpha, 500, "Batch gradient descent: Weights (French, alpha: " + str(alpha) + ")")

    
        print("################ Batch Gradient Descent Weights - English ################")
        print(wb_en)

        print("################ Batch Gradient Descent Weights - French #################")
        print(wb_fr)

        # ################## Stochastic Gradient Descent - English  ################
        ws_en = plot_sgd(X_en, y_en, alpha, 500, "Stochastic gradient descent: Weights (English, alpha: " + str(alpha) + ")")

        # ################### Stochastic Gradient Descent - French #################
        ws_fr = plot_sgd(X_fr, y_fr, alpha, 500, "Batch gradient descent: Weights (French, alpha: " + str(alpha)+ ")")

        print("############## Stochastic Gradient Descent Weights - English #############")
        print(ws_en)

        print("############## Stochastic Gradient Descent Weights - French ##############")
        print(ws_fr)

if __name__ == "__main__":
    main()

