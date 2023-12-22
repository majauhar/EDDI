import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
import pdb
import numpy as np

def fc_uci_decoder(z,  obs_dim,  activation=tf.nn.sigmoid): #only output means since the model is N(m,sigmaI) or bernouli(m)
#     mean = np.array([3.61352356e+00, 1.13636364e+01, 1.11367787e+01, 6.91699605e-02,
#  5.54695059e-01, 6.28463439e+00, 6.85749012e+01, 3.79504269e+00,
#  9.54940711e+00, 4.08237154e+02, 1.84555336e+01, 3.56674032e+02,
#  1.26530632e+01, 2.25328063e+01])
    
#     std = np.array([8.59304135e+00, 2.32993957e+01, 6.85357058e+00, 2.53742935e-01,
#  1.15763115e-01, 7.01922514e-01, 2.81210326e+01, 2.10362836e+00,
#  8.69865112e+00, 1.68370495e+02, 2.16280519e+00, 9.12046075e+01,
#  7.13400164e+00, 9.18801155e+00])
    mean = np.array([6.85478767e+00, 2.78241119e-01, 3.34191507e-01, 6.39141486e+00,
 4.57723561e-02, 3.53080849e+01, 1.38360657e+02, 9.94027376e-01,
 3.18826664e+00, 4.89846876e-01, 1.05142670e+01, 5.87790935e+00])
    
    std = np.array([8.43782079e-01, 1.00784259e-01, 1.21007450e-01, 5.07153999e+00,
 2.18457377e-02, 1.70054011e+01, 4.24937260e+01, 2.99060158e-03,
 1.50985184e-01, 1.14114183e-01, 1.23049494e+00, 8.85548162e-01])
    
    x = layers.fully_connected(z, 50, scope='fc-01')
    x = layers.fully_connected(x, 100, scope='fc-02')
    # x = layers.fully_connected(x, obs_dim, activation_fn=tf.nn.sigmoid,
    #                            scope='fc-final')
    # pdb.set_trace()
    x = layers.fully_connected(x, obs_dim, activation_fn=None,
                            scope='fc-final')
    x = (x - mean) / (std)

    return x, None

def fc_uci_encoder(x, latent_dim, activation=None):
    e = layers.fully_connected(x, 100, scope='fc-01')
    e = layers.fully_connected(e, 50, scope='fc-02')
    e = layers.fully_connected(e, 2 * latent_dim, activation_fn=activation,
                               scope='fc-final')

    return e

def PNP_fc_uci_encoder(x, K, activation=None):
    e = layers.fully_connected(x, 100, scope='fc-01')
    e = layers.fully_connected(e, 50, scope='fc-02')
    e = layers.fully_connected(e, K, scope='fc-final')

    return e


