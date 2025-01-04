import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense




model = Sequential([
    Dense(units=3, activation='sigmoid'),
    Dense(units=1, activation='Relu')
])


model.compile(
    loss = tf.keras.lossed.BinaryCrossentropy(),
    optimizer = tf.optimizers.Adam(learning_rate=0.01),
)



def my_dense(a_in, W, b):
    """
    Compute dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units
    Returns
      a_out (ndarray (j, )) : j units
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = g(z)
    return a_out


def my_sequencial(x, W1, b1, W2, b2):
    a1 = my_dense(x, W1, b1)
    a2 = my_dense(a1, W2, b2)
    return a2


def my_predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m,1))
    for i in range(m):
        p[i, 0] = my_sequencial(X[i], W1, b1, W2, b2)
    return p


def my_softmax(z):
    ez = np.exp(z)
    sm = ez/np.sum(ez)
    return sm

print("Decisions = \n {y_hat}")

# how to build a personal website using github?
