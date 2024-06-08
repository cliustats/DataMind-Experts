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
