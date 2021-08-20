import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import time
import datetime

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

import pennylane as qml
from pennylane.templates import RandomLayers
from sklearn.datasets import load_digits

data = load_digits().data

train_samples = 1400
test_samples = data.shape[0]-train_samples
x_train = data[:train_samples]
x_test = data[-test_samples:]
n_features = x_train.shape[1]
latent_dim = int(math.log(n_features, 2))

n_qubits = latent_dim
dev = qml.device("default.qubit.tf", wires=n_qubits)

# Construct encoder Variational Quantum Circuit.
@qml.qnode(dev, interface='tf', diff_method='backprop')
def qnode_e(inputs, weights):
    qml.templates.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize = True)
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Construct decoder Variational Quantum Circuit.
@qml.qnode(dev, interface='tf', diff_method='backprop')
def qnode_d(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.probs(wires=[i for i in range(n_qubits)])

# Variational quantum weights in encoder and decoder.
weight_shapes_e = {"weights": (6, n_qubits, 3)} # 6 quantum layers and each qubit has 3 parameters per layer
weight_shapes_d = {"weights": (6, n_qubits, 3)} # 6 quantum layers and each qubit has 3 parameters per layer

qlayer_e = qml.qnn.KerasLayer(qnode_e, weight_shapes_e, output_dim=latent_dim)
qlayer_d = qml.qnn.KerasLayer(qnode_d, weight_shapes_d, output_dim=n_features)

# Define the Quantum Autoencoder (QAE) class.
class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.classical_optimizer = tf.keras.optimizers.Adam(0.01) # initial classical learning rate
        self.quantum_optimizer = tf.keras.optimizers.Adam(0.01) # initial quanutm learning rate

        self.vqc_e = tf.keras.Sequential([qlayer_e])
        self.cls_e = tf.keras.Sequential([layers.Dense(self.latent_dim)])

        self.vqc_d = tf.keras.Sequential([qlayer_d])
        self.cls_d = tf.keras.Sequential([layers.Dense(n_features)])
        
    def call(self, x):
        result = self.vqc_e(x)
        encoded = self.cls_e(result)
        result = self.vqc_d(encoded)
        decoded = self.cls_d(result)
        return decoded

model = Autoencoder(latent_dim)

print('Start training...')
start_time = time.time()

BATCH_SIZE = 32
batches = len(x_train) // BATCH_SIZE
epochs = 50
for epoch in range(epochs):
    # for adjusting learning rates (optional)
    if epoch == 25:
        model.classical_optimizer = tf.keras.optimizers.Adam(0.005)
        model.quantum_optimizer = tf.keras.optimizers.Adam(0.005)
    
    # Train the QAE model.
    sum_loss = 0
    for batch in range(batches):
        x = x_train[BATCH_SIZE * batch:min(BATCH_SIZE * (batch + 1), len(x_train))]
        
        with tf.GradientTape() as t1, tf.GradientTape() as t2:
            # encoded =  model.encoder(x)
            # y_pred = model.decoder(encoded)
            y_pred = model(x)

            loss = tf.reduce_mean(tf.square(y_pred - x))

            grad_vqc = t1.gradient(loss, model.vqc_e.trainable_variables + \
                                        model.vqc_d.trainable_variables)
            model.quantum_optimizer.apply_gradients(zip(grad_vqc, model.vqc_e.trainable_variables + \
                                        model.vqc_d.trainable_variables))

            grad_cls = t2.gradient(loss, model.cls_e.trainable_variables + \
                                        model.cls_d.trainable_variables)
            model.classical_optimizer.apply_gradients(zip(grad_cls, model.cls_e.trainable_variables + \
                                        model.cls_d.trainable_variables))
        
        sum_loss += loss
        print('Batch {}/{} Loss {:.4f}'.format(batch, batches, loss), end='\r')

    avg_loss = sum_loss/batches

    # Run test samples.
    with tf.GradientTape() as t1, tf.GradientTape() as t2:
        y_pred = model(x_test)
        test_loss = tf.reduce_mean(tf.square(y_pred - x_test))

    et = time.time() - start_time
    et = str(datetime.timedelta(seconds=et))[:-7]
    print('Elapsed {}\t Epoch {}/{} [Train Loss: {:.4f}]\t [Test Loss: {:.4f}]'.format(et, epoch+1, epochs, avg_loss.numpy(), test_loss.numpy()))