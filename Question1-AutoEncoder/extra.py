
"""
accuracyBool = (imageData == imgcheck)
no_zeros = accuracyBool[np.where(accuracyBool==True)]
print(no_zeros)
"""

import numpy as np

# Given values
Y_true = [1,1,2,2,4] # Y_true = Y (original values)

# Calculated values
Y_pred = [0.6,1.29,1.99,2.69,3.4] # Y_pred = Y'

# Mean Squared Error
MSE = np.square(np.subtract(Y_true,Y_pred))
print(np.sum(MSE)/5)
print(MSE.mean())
