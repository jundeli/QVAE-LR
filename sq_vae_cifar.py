import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import math

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers

from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

import time
import datetime
import pickle

data = tf.keras.datasets.cifar10.load_data()
data = data[0][0][:1797,:,:,:]
images = []
for img in data:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(gray)
data = np.array(images).reshape((1797, -1)).numpy()

# data normalization
norm = lambda a: np.array([i/17 for i in a])
data = np.array([norm(i) for i in data])

train_samples = 1400
test_samples = data.shape[0]-train_samples
x_train = data[:train_samples]
x_test = data[-test_samples:]

n_features = x_train.shape[1]

latent_dim = 16 #int(math.log(n_features, 2))
batch_size = 32
patches = 8
quantum_e = True
quantum_d = False

n_single_features = n_features // patches

if quantum_e and quantum_d:
    MODEL_SAVE_DIR = "model/sq-vae/patch%s" % patches
elif quantum_e and not quantum_d:
    MODEL_SAVE_DIR = "model/sq-vae-e/patch%s" % patches
elif not quantum_e and quantum_d:
    MODEL_SAVE_DIR = "model/sq-vae-d/patch%s" % patches
else:
    MODEL_SAVE_DIR = "model/vae"
    
MODEL_NAME = "latent-%s" % latent_dim

model_spec_name = "%s-model" % MODEL_NAME
model_rslt_name = "%s-results.pickle" % MODEL_NAME

model_save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

model_ckpt_path = os.path.join(model_save_path, "learned-model")
model_spec_path = os.path.join(model_save_path, model_spec_name)
model_rslt_path = os.path.join(model_save_path, model_rslt_name)

n_qubits = int(math.log(n_single_features, 2))
qml.enable_tape()
dev = qml.device("default.qubit.tf", wires=n_qubits)

@qml.qnode(dev, interface='tf', diff_method='backprop')
def qnode_e(inputs, weights):
    qml.templates.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize = True)
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

@qml.qnode(dev, interface='tf', diff_method='backprop')
def qnode_d(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    return qml.probs(wires=[i for i in range(n_qubits)])

weight_shapes_e = {"weights": (3, n_qubits, 3)}
weight_shapes_d = {"weights": (3, n_qubits, 3)}

qlayer_e = qml.qnn.KerasLayer(qnode_e, weight_shapes_e, output_dim=latent_dim)
qlayer_d = qml.qnn.KerasLayer(qnode_d, weight_shapes_d, output_dim=n_features)

class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.optimizer = tf.keras.optimizers.Adam(0.001)
        
        self.fc_mu = layers.Dense(latent_dim)
        self.fc_var = layers.Dense(latent_dim)
        if quantum_e:
            self.qlayers_e = []
            for i in range(patches):
                self.qlayers_e.append(qml.qnn.KerasLayer(qnode_e, weight_shapes_e, output_dim=n_qubits))

        self.encoder = tf.keras.Sequential([
              layers.Dense(512, activation='relu'),
              layers.Dense(256, activation='relu'),
              layers.Dense(64)
            ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(n_features)
        ])
    
    def encode(self, x):
        if quantum_e:
            split_x = tf.split(x, num_or_size_splits=patches, axis=-1)
            for i in range(patches):
                patch_x = self.qlayers_e[i](split_x[i])
                if i == 0:
                    result = patch_x
                else:
                    result = tf.concat([result, patch_x], -1)
        else:
            result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]
    
    def decode(self, z):
        x = self.decoder(z)
        return x
        
    def reparameterize(self, mu, logvar):
        std = tf.math.exp(0.5 * logvar)
        eps = tf.random.normal(std.shape)
        return eps * std + mu
    
    def call(self, x):
        [mu, log_var] = self.encode(x)
        z = self.reparameterize(mu, log_var)
        decoded = self.decode(z)

        return [decoded, x, mu, log_var]
    
    def loss_function(self, *args):
        recons = args[0]
        x = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = batch_size / train_samples
        recons_loss =tf.reduce_mean((x - recons)**2, axis=-1, keepdims=True)
        kld_loss = tf.math.reduce_mean(-0.5 * tf.math.reduce_sum(1 + log_var - mu ** 2 - 
                                tf.math.exp(log_var), axis=-1, keepdims=True), axis=-1, keepdims=True)
        loss = recons_loss + kld_weight * kld_loss

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
    
    def sample(self, noise):
        samples = self.decode(noise)
        return samples
    
model = Autoencoder(latent_dim)

recons_losses = []
losses = []
fakes = []
times = []

print('Start training...')
start_time = time.time()
EPOCHS = 20
for epoch in range(EPOCHS):    
    batch_losses = []
    batch_recons_losses = []
    noise = np.random.normal(size=[6, latent_dim]).astype(np.float32)
    fake = model.sample(noise)
    fakes.append(fake)
    
    epoch_time = time.time()
    batches = len(x_train) // batch_size
    for batch in range(batches):
        x = x_train[batch_size * batch:min(batch_size * (batch + 1), len(x_train))]
        
        with tf.GradientTape() as t1, tf.GradientTape() as t2, tf.GradientTape() as t3:
            results =  model(tf.cast(x, tf.float32))
            loss = tf.reduce_mean(model.loss_function(*results)['loss'])
            batch_losses.append(loss)
            batch_recons_losses.append(tf.reduce_mean(model.loss_function(*results)['Reconstruction_Loss']))

            if quantum_e:
                grad_enc = t1.gradient(loss, 
                                       model.qlayers_e[0].trainable_variables + 
                                       model.qlayers_e[1].trainable_variables + 
                                       model.qlayers_e[2].trainable_variables + 
                                       model.qlayers_e[3].trainable_variables + 
                                       model.qlayers_e[4].trainable_variables + 
                                       model.qlayers_e[5].trainable_variables + 
                                       model.qlayers_e[6].trainable_variables + 
                                       model.qlayers_e[7].trainable_variables
                                      )
                model.optimizer.apply_gradients(zip(grad_enc,
                                                   model.qlayers_e[0].trainable_variables + 
                                                   model.qlayers_e[1].trainable_variables + 
                                                   model.qlayers_e[2].trainable_variables + 
                                                   model.qlayers_e[3].trainable_variables + 
                                                   model.qlayers_e[4].trainable_variables + 
                                                   model.qlayers_e[5].trainable_variables + 
                                                   model.qlayers_e[6].trainable_variables + 
                                                   model.qlayers_e[7].trainable_variables
                                                   ))
            else:
                grad_enc = t1.gradient(loss, model.encoder.trainable_variables)
                model.optimizer.apply_gradients(zip(grad_enc, model.encoder.trainable_variables))
        
            grad_dec = t2.gradient(loss, model.decoder.trainable_variables)
            model.optimizer.apply_gradients(zip(grad_dec, model.decoder.trainable_variables))
            
            grad_z = t3.gradient(loss, model.fc_mu.trainable_variables+model.fc_var.trainable_variables)
            model.optimizer.apply_gradients(zip(grad_z, model.fc_mu.trainable_variables + 
                                                model.fc_var.trainable_variables))

        print('Epoch {} Batch {}/{}\tLoss {:.4f}'.format(epoch+1, batch, batches, loss.numpy()), end='\r')
    
    epoch_loss = np.mean(batch_losses)
    losses.append(epoch_loss)
    epoch_recons_loss = np.mean(batch_recons_losses)
    recons_losses.append(epoch_recons_loss)
    epoch_t = time.time() - epoch_time
    times.append(epoch_t)
    
    et = time.time() - start_time
    et = str(datetime.timedelta(seconds=et))[:-7]
    print('Elapsed {}\t Epoch {}/{} \tLoss {:.4f}'.format(et, epoch+1, EPOCHS, epoch_loss))
    
    with open(model_rslt_path, "wb") as f:
        pickle.dump((epoch_loss, epoch_recons_loss, epoch_t, fakes), f)
        
    model.save_weights(model_ckpt_path + str(epoch))
    