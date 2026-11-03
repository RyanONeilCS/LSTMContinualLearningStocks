"""
LSTM written from scratch with NumPy

No deep learning libraries
The cell the forward pass the backward pass and the training loop are all by hand

Shapes used throughout
    B  batch size          how many windows at once
    T  window length       how many days per window
    D  number of features  per day
    H  hidden size         number of memory slots

    X        (B, T, D)     input windows
    x_t      (B, D)        one day for the whole batch
    h, c     (B, H)        hidden state and cell state
    W        (D + H, 4H)   all four gates fused
    b        (4H,)         all four biases fused
    W_y      (H, 1)        readout layer
    b_y      (1,)          readout bias
    y_hat    (B, 1)        the prediction


HW1 Gradient Descent
    predictions = X @ self.weights + self.bias
    errors = predictions - y
    gradient_w = (2/n_samples) * (X.T @ errors)
    gradient_b = (2/n_samples) * np.sum(errors)
    self.weights = self.learning_rate * gradient_w
"""

from __future__ import annotations

import numpy as np



#Sigmoid squashes anything into 0 to 1
#NumPy has np.tanh but no sigmoid so it has to be written
#exp(-|z|) form so it does not overflow
def sigmoid(z: np.ndarray):

    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


#Initialise the parameters and these four arrays are the model
#Needs W (D+H 4H) random / sqrt(D+H) so the sigmoids do not saturate
#Needs b (4H,) zeros with the forget slice at 1 so the gates start open
#Needs W_y (H, 1) random / sqrt(H) scaled down so first guesses are small
#Needs b_y (1,) zeros
def parameters(D: int, H: int, seed: int = 0):

    #initialise the weights for each gate at a random number
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((D+H, 4*H)) / np.sqrt(D + H)

    #initialise the biases for all gates to 0
    b = np.zeros((4*H,))

    #gate order
    #b[0:H] = f forget
    #b[H:2*H] = i input
    #b[2*H:3*H] = g candidate
    #b[3*H:4*H] = o output

    #set forget gate biases to 1 so they start open
    b[0:H] = 1.0

    #Create the readout layer
    W_y = rng.standard_normal((H, 1)) / np.sqrt(H) *.02
    b_y = np.zeros((1,))

    return {"W": W, "b": b, "W_y": W_y, "b_y": b_y}



#Run the cell for one day and this is the LSTM itself
#Needs concat of x_t and h_prev then one matmul then split into 4
#Needs c = f * c_prev + i * g then h = o * tanh(c)
#Needs the cache saved for backward
#c is never squashed and only the things reading it are
def lstm_step(x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray, parameters: dict,):

    #x_t = (B, D) Features for this day
    #h_prev = (B, H) Hidden state from the previous day
    #c_prev = (B, H) Cell state from the previous day

    ConcatXandH = np.hstack((x_t, h_prev))

    #Multiply XandH by the weights and bias added together
    z = ConcatXandH @ parameters["W"] + parameters["b"]

    #Split z into 4 groups for the 4 gates
    #f forget  i input  g candidate  o output
    f, i, g, o = np.split(z, 4, axis = 1)

    #sigmoid f i and o so they are between 0 and 1
    f = sigmoid(f)
    i = sigmoid(i)
    o = sigmoid(o)

    #tanh g so that it is between -1 and 1
    g = np.tanh(g)

    #Calculate the current cell state
    c = f * c_prev + i * g

    #Calculate the current hidden state
    #store tanc for the cache
    tanc = np.tanh(c)
    h = o * tanc

    #Create the cache for the backward pass
    cache_t = (ConcatXandH, f, i, g, o, c_prev, c, tanc)

    return h, c, cache_t



#Forward pass reads a batch of windows one day at a time
#Needs h and c starting at zeros so every window starts from scratch
#Needs a loop over T calling lstm_step and collecting the caches
#Needs y_hat from the last h only
def forward(X: np.ndarray, parameters: dict):

    #Create the shape variables for the input data
    B, T, D = X.shape

    #Calculate the hidden size from the parameters
    H = parameters["W"].shape[1] // 4

    #Create the initial hidden and cell states as zeros
    h = np.zeros((B, H))
    c = np.zeros((B, H))

    #Create a list to store the caches for each timestep
    caches = []

    #Run the cell over each day carrying h and c forward
    for t in range(T):
        h, c, cache = lstm_step(X[:, t, :], h, c, parameters)
        caches.append(cache)

    #Final output from the last hidden state and the readout layer
    y_hat = h @ parameters["W_y"] + parameters["b_y"]

    return y_hat, caches, h


#Backward pass works out how each parameter should move
#Every line of lstm_step mirrored in reverse using the product rule
#dW and db use += since W and b were used on all T days
#dh and dc use = since they get handed back one day at a time
#dc is the exception and uses += since c reaches the loss two ways
#Returns the same keys as the parameters so the update is one loop
def backward(y_hat: np.ndarray, y: np.ndarray, h_T: np.ndarray, caches: list, parameters: dict):

    #Shape variables from the parameters and caches
    B = y_hat.shape[0]
    H = parameters["W"].shape[1] // 4
    D = parameters["W"].shape[0] - H
    T = len(caches)

    #Calculate the loss with mean squared error
    loss = np.mean((y_hat - y) ** 2)

    #Calculate the gradient of the loss for the prediction
    dy_hat = 2*(y_hat - y)/B

    #Calculate the gradients for the output layer
    dW_y = h_T.T @ dy_hat
    db_y = dy_hat.sum(axis = 0)

    #Calculate the gradient going into the final hidden state
    dh = dy_hat @ parameters["W_y"].T

    #Gate weight and bias gradients start at zeros so they can add up
    dW = np.zeros_like(parameters["W"])
    db = np.zeros_like(parameters["b"])

    #Cell state gradient starts at zeros
    dc = np.zeros((B, H))

    #Loop backwards over each timestep from the last day to the first
    for t in reversed(range(T)):

        #Unpack this timestep's cache in the order it was packed
        ConcatXandH, f, i, g, o, c_prev, c, tanc = caches[t]

        #Calculate the output gate and cell state gradients from h = o * tanc
        do = dh * tanc
        dc = dc + dh * o * (1 - tanc ** 2)

        #Calculate the gate gradients from c = f * c_prev + i * g
        df = dc * c_prev
        di = dc * g
        dg = dc * i
        dc_prev = dc * f

        #Undo the sigmoid and tanh using the values saved in the cache
        dz_f = df * f * (1 - f)
        dz_i = di * i * (1 - i)
        dz_g = dg * (1 - g ** 2)
        dz_o = do * o * (1 - o)

        #Stack the four back together in gate order to undo the split
        dz = np.hstack((dz_f, dz_i, dz_g, dz_o))

        #Add this timestep's share to the weight and bias gradients
        dW += ConcatXandH.T @ dz
        db += dz.sum(axis = 0)

        #Split the concat gradient and keep only the hidden state part
        dConcat = dz @ parameters["W"].T
        dh_prev = dConcat[:, D:]

        #Pass the hidden and cell state gradients back a timestep
        dh = dh_prev
        dc = dc_prev

    return loss, {"W": dW, "b": db, "W_y": dW_y, "b_y": db_y}


#Training loop using mini batch gradient descent
#Needs a new shuffle of the window order each epoch
#Only shuffles which windows go together and never the days inside one
#Needs clipping before the update or one bad batch can wreck the weights
#Clip on the total across all four arrays and not each one alone
#Changes parameters in place and returns one loss per epoch
def train(X: np.ndarray, y: np.ndarray, parameters: dict, lr: float = 1e-3, epochs: int = 20, batch_size: int = 64, clip: float = 5.0, seed: int = 0):

    #Random number generator for the shuffling
    rng = np.random.default_rng(seed)

    #The number of windows
    N  = X.shape[0]

    #One average loss per epoch
    history = []

    for epoch in range(epochs):
        perm = rng.permutation(N)
        epoch_total = 0.0

        for start in range(0, N, batch_size):
            idx = perm[start:start+batch_size]
            X_b = X[idx]
            y_b = y[idx]
            y_hat, caches, h_T = forward(X_b, parameters)
            loss, grads = backward(y_hat, y_b, h_T, caches, parameters)
            epoch_total += loss * len(idx)
            total = np.sqrt(sum((g ** 2).sum() for g in grads.values()))

            if total > clip:
                for k in grads:
                    grads[k] *= clip / total

            for k in parameters:
                parameters[k] -= lr * grads[k]

        history.append(epoch_total / N)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {history[-1]:.6f}")

    return history
