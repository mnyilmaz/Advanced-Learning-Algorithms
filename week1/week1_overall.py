import tensorflow as tf
import numpy as np
from keras.layers import Activation, Dense
import keras.models
from keras.models import Sequential

# Array to initialize degree and duration of roasting (1 x 2 matrix) (rows x columns)
x = np.array([[200.0, 17.0]])  # 200 degrees and 17-minutes duration and don^t forget to double square brackets

# Layer 1
layer_1 = Dense(units=3, activation='sigmoid')  # 3 neurons and with using 'g' function a1 = g(w1.x + b1)

# a1
a1 = layer_1(x)  # x indicates 'input' in here
print(a1)  # that will print out 1 x 3 matrix with different values, remember the shape on your notebook
# output will represent as tensor datatype, to convert this into numpy array use a1.numpy()

# Layer 2
layer_2 = Dense(units=1, activation='sigmoid')

# a2
a2 = layer_2(a1)
print(a2)  # that will print out 1 x 1 matrix as tf.Tensor([[0.7746036]], shape=(1, 1), dtype=float32)

# Testing
if a2 >= 0.5:
    yhat = 1  # yhat -> ŷ
else:
    yhat = 0

# Alternative way for a neural network model
layer_1 = Dense(units=3, activation='sigmoid')
layer_2 = Dense(units=2, activation='sigmoid')

'''
Dear tf, please form a neural network for me by sequentially string together these two layers that I just formed.
So that equals to a1 = layer_1(x) and a2 = layer_2(a1)
'''
nn_model = Sequential([layer_1, layer_2])

# Much better representation
model = Sequential([
    Dense(units=3, activation='sigmoid'),
    Dense(units=1, activation='sigmoid')])

# Inputs X
x = np.array([[200.0, 17.0],
              [120.0, 5.0],
              [425.0, 20.0],
              [212.0, 18.0]])

# Outputs Y
y = np.array([[1, 0, 0, 1]])

# How to train your model:
model.compile(...)  # more about "compile" next week, but this should come after model definition in general
model.fit(x, y)

# If you have a new X inputs and wondering about the result
x_new = np.array([[100.0, 26.0],
                  [220.0, 3.0],
                  [546.0, 28.0],
                  [234.0, 12.0]])
model.predict(x_new)

# Formula Implementation

'''
    Write down the code as implemented in formula
    a1_1 = g(w1_1 . x) + b1_1 | w1 -> first layer | w1_1 -> first layer's first neuron/unit
    a1_2 = g(w1_2 . x) + b1_2
    a1_3 = g(w1_3 . x) + b1_3
'''

# First Layer (includes 3 units)
x = np.array([200.0, 17.0])
w1_1 = np.array([1, 2])
b1_1 = np.array([-1])
z1_1 = np.dot(w_1, x) + b1_1  # np.dot allows implementing dot product
a1_1 = sigmoid(z1_1)

w1_2 = np.array([12, 5])
b1_2 = np.array([9])
z1_2 = np.dot(w1_2, x) + b1_2
a1_2 = sigmoid(z1_2)

w1_3 = np.array([8, 1])
b1_3 = np.array([2])
z1_3 = np.dot(w1_3, x) + b1_3
a1_3 = sigmoid(z1_3)

# Finalization
a1 = np.array[a1_1, a1_2, a1_3]

# Second Layer (includes 1 unit)
w2_1 = np.array([3, 7])
b2_1 = np.array([5])
z2_1 = np.dot(w2_1, a1) + b2_1
a2_1 = sigmoid(z2_1)

# Implementing own Dense function
a_in = np.array([-2, 4])  # equals to x or a_0 as a result, this is called 'layer0'
W = np.array([[1, 12, 8],
              [2, 5, 1]])

b = np.array([-1, 9, 2])


# Related function
def dense(a_in, W, b):
    units = W.shape[1]  # output is number of columns
    a_out = np.zeros(units)  # array of zeros to fill a_out = [0, 0, 0]
    for j in range(units):
        w = W[:, j]  # pull out the jth column of the matrix
        z = np.dot[w, a_in] + b[j]
        a_out[j] = sigmoid(z)  # in optional lab instead of sigmoid(z) -> g(z) occurs
    return a_out


def sequential(layer):
    # consider you have W_num and b_num matrices above
    a1 = dense(layer, W1, b1)
    a2 = dense(a1, W2, b2)
    a3 = dense(a2, W3, b3)
    a4 = dense(a3, W4, b4)
    final_layer = a4
    return final_layer


# A different approach for layer implementation
X = np.array([[200.0, 17.0]])  # there are 2 SQUARE BRACKETS that mean 2D array
W = np.array([[1, 12, 8],
              [2, 5, 1]])

B = np.array([[-1, 9, 2]])  # 1 x 3 matrix, 2D array,

'''
Reason to convert B (b) into 2D is multiplication process. In multiplication process below after multiplication of 
A_in and W, B had to be added, for that B have to be also 2D array.

Also sigmoid function can be implemented as:
    g = sigmoid
and use as
    A_out = g(z)
'''


def quick_dense(A_in, W, B):
    Z = np.matmul(A_in, W) + B  # matmul: Matrix multiplication, instead of a dot product element by element
    A_out = sigmoid(Z)
    return A_out


'''
Matrix Recap

Dot product 

A = [1]   B = [3]     -> Dot product of A.B = Z = (1 x 3) + (2 x 4) = 11
    [2]       [4]

Transpose
- On transpose remember, one column at a time. No matter what how many 
A = [1]  -> Transpose of A_t = [1, 2]  | Now this is no longer matrix but a vector
    [2]
    
thus, Dot product of A and B (A.B) equals to vector vector multiplication of transpose of A and B (A_t x B)

                                    A.B = A_t x B

Vector Matrix Multiplication
a = [1]  -> Transpose of a_t = [1, 2]
    [2]
                                        Z = a_t x W = [(1 x 3) + (2 x 4), (1 x 5) + (2 x 6)] = [11, 17]
W = [3  5]    
    [4  6]

    
Matrix Matrix Multiplication
A = [1  -1]    ->   A_t = [1     2]       and      W = [3   5]
    [2  -2]               [-1   -2]                    [4   6]

Z = (A_t)(W) => multiply first row of the A_t with first column of W 

Z = [(row1 x col1)     (row1 x col2)]   -> first row of A_t and col represents W columns
    [(row2 x col1)     (row2 x col2)]   -> second row of A_t and col represents W columns
                   
Z = [(1 x 3) + (2 x 4)     (1 x 5) + (2 x 6)]  = [11    17]
    [(-1 x 3) + (-2 x 4) (-1 x 5) + (-2 x 6)]    [-11  -17]

Matrix Multiplication Rules
A = [1  -1  0.1]    ->      A_t = [1      2]        and     W = [3  5   7   9]
    [2  -2  0.2]                  [-1    -2]                    [4  6   8   0]
                                  [0.1  0.2]

Z = [(row1 x col1)  (row1 x col2)   (row1 x col3)  (row1 x col4)]
    [(row2 x col1)  (row2 x col2)   (row2 x col3)  (row2 x col4)]
    [(row3 x col1)  (row3 x col2)   (row3 x col3)  (row3 x col4)]
    
  = [(1 x 3) + (2 x 4)          (1 x 5) + (2 x 6)        (1 x 7) + (2 x 8)          (1 x 9) + (2 x 0)]
    [(-1 x 3) + (-2 x 4)       (-1 x 5) + (-2 x 6)      (-1 x 7) + (-2 x 8)       (-1 x 9) + (-2 x 0)]
    [(0.1 x 3) + (0.2 x 4)    (0.1 x 5) + (0.2 x 6)    (0.1 x 7) + (0.2 x 8)    (0.1 x 9) + (0.2 x 0)]
    
  = [11      17    23      9]
    [-11    -17   -23     -9]
    [1.1    1.7   2.3    0.9]  
    
REMEMBER! If only column number of A_t equals to row number of W, matrix multiplication is valid.

Output row number must be equal to A_t row number.
Output column number must be equal to W column number.

A_t = np.array([[1,     2],
                [-1,   -2],
                [0.1, 0.2]])

'''

# Matrix Multiplication Code
A = np.array([[1, -1, 0.1],
              [2, -2, 0.2]])

W = np.array([[3, 5, 7, 9],
              [4, 6, 8, 0]])

A_t = A.T  # implementation of transpose function
Z = np.matmul(A_t, W)
Z_alt = A_t @ W
print(Z)

# Dense Layer Vectorized
C_T = np.array([200, 17])  # Coffee transpose
C_W = np.array([[1, -3, 5],
                [-2, 4, -6]])  # Coffee 'w' values

C_B = np.array([[-1, 1, 2]])  # Coffee 'b' values and remember to obtain 2D B values in order to perform addition

'''
In that case remember the layers,  Z = (A_t)(W) + B
'''

def dense(T, W, B):
    Z = np.matmul(T, W) + B
    a_out = sigmoid(Z)
    print(a_out)
    return a_out


dense(C_T, C_W, C_B)
