# Model evaluation and and selection
 import numpy as np

 from sklearn.linear_model import LinearRegression
 from sklearn.preprocessing imort StandardScaler, PolynomialFeatures
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import mean_squared_error

 import tensorflow as tf

 import utils

 data = ...

 x = data[:, 0]
 y = data[;, 1]

 ## Split the dataset into training, cross validation, and test sets

# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
x_train, x_, y_train, y_ = train_test_split(x, y, test_size= 0.40, random_state=1)

# Split the 40% subset above into two: one half for cross validation and the other for the test set
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.5, random_state=1)

# Delete temporary variables
del x_, y_

## Fit a liner model
scaler_linear = StandardScaler()
X_train_scaled = scaler_linear.fit_transform(x_train)

## Train the model

linear_model = LinearRegression()

linear_model.fit(x_train_scaled, y_train)

## Evaluate the model
yhat = linear_model.predict(X_train_scaled)


# Use scikit-learn's utility function and divide by 2
print(f"training MSE (using sklearn function): {mean_squared_error(y_train, yhat) / 2}")


# for-loop implementation
total_squared_error = 0

for i in range(len(yhat)):
    squared_error_i = (yhat[i] - y_train[i])**2
    total_squared_error += squared_error_i

mse = total_squared_error / (2*len(yhat))


X_cv_scaled = scaler_linear.transform(x_cv)


# create the additional PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)

X_train_mapped = poly.fit_transform(x_train)

scaled_poly = StandardScaler()

X_train_mapped_scaled = scaled_poly.fit_transform(X_train_mapped)



## adding polynomial features



# select the model with the lowest error
model_num = 3

# compute the test error
yhat = model_bc[model_num-1].predict(x_bc_test_scaled)
yhat = tf.math.sigmoid(yhat)
yhat = np.where(yhat >= threshold, 1, 0)
nn_test_error = np.mean(yhat != y_bc_test)








# Transformer

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float))

    def forward(self, Q, K, V, mask=None):
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        output, attn_weights = self.attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.fc(output)
        return output, attn_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_seq_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        output = self.fc_out(x)
        return output
