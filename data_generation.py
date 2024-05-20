# General imports.
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import random
import itertools
import math
from sklearn.preprocessing import StandardScaler


class DataGeneration():
    
    
    def generate_linear_gaussian(self,
                                 n: int = 1000, 
                                 x_causes_y: bool = True,
                                 xy_coeff: float = 1.0,
                                 coefficient_range: tuple = (1.1, 1.25)):

        '''
        '''

        # Construct noise terms of structural equation.
        # Indices: 0 = X, 1 = Y, 2 = Z1, 3 = Z2, 4 = Z3, 5 = Z4, 6 = Z5, 7 = Z6, 8 = Z7, 9 = Z8.
        total_vars = 18
        noise = []
        for var in range(total_vars):
            noise.append(np.random.normal(loc = 0.0, scale = 1.0, size = n).reshape(-1, 1))
        

        # Define coefficient generator.
        coeff = lambda : np.random.uniform(low = coefficient_range[0],
                                           high = coefficient_range[1],
                                           size = 1)
        
        # Define variables.
        if not x_causes_y:
            xy_coeff = 0
        Z1 = noise[0]
        Z4a = noise[1]
        Z4b = noise[2]
        Z5 = noise[3]
        Z8 = noise[4]
        M1 = noise[5]
        M2 = noise[6]
        M3 = coeff()*M1 + coeff()*M2 + noise[7]
        B1 = noise[8]
        B2 = noise[9]
        B3 = coeff()*B1 + coeff()*B2 + noise[10]
        X = coeff()*Z1 + coeff()*Z5 + coeff()*M1 + coeff()*B1 + coeff()*B3 + noise[11]
        Z3a = coeff()*X + noise[12]
        Z3b = coeff()*Z3a + noise[13]
        Z3c = coeff()*Z3b + noise[14]
        Z3d = coeff()*X + coeff()*Z4a + noise[15]
        Y = xy_coeff*X + coeff()*Z1 + coeff()*Z3c + coeff()*Z3d + coeff()*Z4b + coeff()*M2 + coeff()*B2 + coeff()*B3 + noise[16]
        Z7 = coeff()*X + noise[17]
            
        # Construct dataframes.
        df_vars = pd.DataFrame({"X": X.reshape(-1), 
                                "Y": Y.reshape(-1), 
                                "Z1": Z1.reshape(-1),
                                "Z3a": Z3a.reshape(-1), 
                                "Z3b": Z3b.reshape(-1),
                                "Z3c": Z3c.reshape(-1),
                                "Z3d": Z3d.reshape(-1),
                                "Z4a": Z4a.reshape(-1), 
                                "Z4b": Z4b.reshape(-1),
                                "Z5": Z5.reshape(-1), 
                                "Z7": Z7.reshape(-1), 
                                "Z8": Z8.reshape(-1),
                                "M1": M1.reshape(-1),
                                "M2": M2.reshape(-1),
                                "M3": M3.reshape(-1),
                                "B1": B1.reshape(-1),
                                "B2": B2.reshape(-1),
                                "B3": B3.reshape(-1)})

        return df_vars
    