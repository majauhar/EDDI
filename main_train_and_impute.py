'''
EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE
This code implememts partial VAE (PNP) part demonstrated on a UCI dataset.

To run this code:
python main_train_and_impute.py  --epochs 3000  --latent_dim 10 --p 0.99 --data_dir your_directory/data/boston --output_dir your_directory/model

possible arguments:
- epochs: number of epochs.
- latent_dim: size of latent space of partial VAE.
- p: upper bound for artificial missingness probability. For example, if set to 0.9, then during each training epoch, the algorithm will
  randomly choose a probability smaller than 0.9, and randomly drops observations according to this probability.
  Our suggestion is that if original dataset already contains missing data, you can just set p to 0.
- batch_size: mini batch size for training. default: 100
- iteration: iterations (number of minibatches) used per epoch. set to -1 to run the full epoch.
  If your dataset is large, please set to other values such as 10.
- K: dimension of the feature map (h) dimension of PNP encoder. Default: 20
- M: Number of MC samples when perform imputing. Default: 50
- data_dir: Directory where UCI dataset is stored.
- output_dir: Directory where the trained model will be stored and loaded.

Other comments:
- We assume that the data is stored in an excel file named d0.xls,
   and we assume that the last column is the target variable of interest (only used in active learning)
   you should modify the load data section according to your data.
- Note that this code assumes a Gaussian noise real valued data. You may need to modify the likelihood function for other types of data.
- In preprocessing, we chose to squash the data to the range of 0 and 1. Therefore our decoder output has also been squashed
  by a sigmoid function. If you wish to change the preprocessing setting, you may also need to change the decoder setting accordingly.
  This can be found in coding.py.

File Structure:
- main functions:
  main_train_impute.py: implements the training of partial VAE (PNP) part demonstrated on a UCI dataset.
  main_active_learning.py: implements the EDDI active learning strategy, together with a global single ordering strategy based on partial VAE demonstrated on a UCI dataset
                           it will also generate a information curve plot.
- decoder-encoder functions: coding.py
- partial VAE class:p_vae.py
- training-impute functions: train_and_test_functions.py
- training-active learning functions:active_learning_functions.py
- active learning visualization: boston_bar_plot.py, this will visualize the decision process of eddi on Boston Housing data.
- data: data/boston/d0.xls

'''
### load models and functions
from train_and_test_functions import *
#### Import libraries
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from random import sample
import pandas as pd
import sklearn.preprocessing as preprocessing
import pdb
plt.switch_backend('agg')
tfd = tf.contrib.distributions

### load data
# Data = pd.read_excel(UCI + '/d0.xls')
# Data = Data.values # as_matrix()
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
Data = np.array(pd.read_csv(url, low_memory=False, sep=';'))
# pdb.set_trace()
# missing_data = np.load('MAR-same-inter_column-7_p-0.2_nan-0.18_depcol-3_dep-7913.npy')
# missing_data = np.load('MAR-same-random_column-7_p-0.5_nan-0.33_depcol-3_p1-0.17_p2-0.21_p3-0.26_dep-91112.npy')
# missing_data = np.load('MAR-same-sum_column-7_p-0.2_nan-0.24_depcol-3_dep-71213.npy')
# missing_data = np.load('MCAR_column-7_p-0.2_nan-0.2.npy')
# missing_data = np.load('MNAR-notsame-data_housing_mnar_7_p-0.1_maxdep-5_nan-0.13.npy')
# missing_data = np.load('MNAR-same-inter_column-7_p-0.2_nan-0.20_depcol-4_misdepcol-45_dep-811.npy')
# missing_data = np.load('MNAR-same-nn_column-7_p-0.5_nan-0.20_depcol-4_misdepcol-03_depcol-1013_threshold-0.4.npy')

missing_data_temp = np.load('Wine_0.7.npy')
missing_data = np.empty((4898, 12))
for i in range(4898):
    missing_data[i] = np.append(missing_data_temp[i], Data[i][11])

### data preprocess
# max_Data = 1  #
# min_Data = 0  #
# Data_std = (Data - Data.min(axis=0)) / (Data.max(axis=0) - Data.min(axis=0))
# Data = Data_std * (max_Data - min_Data) + min_Data

# different normalisation
mean = np.mean(Data, axis=0)
std = np.std(Data, axis=0)

# print(np.array(mean), np.array(std))

# exit()
Data = (Data - mean) / std
# Data = Data - np.mean(Data, axis=0)
# Data = Data / np.std(Data, axis=0)

# Data = (Data-Data.mean(0))/np.std(Data,0)
#Mask = np.ones(Data.shape) # This is a mask indicating missingness, 1 = observed, 0 = missing.
# missing_mask = np.random.rand(*Data.shape) <0.7
missing_mask = np.invert(np.isnan(missing_data)).astype(float)
# missing_mask = np.isnan(data).astype(float)
# this UCI data is fully observed. you should modify the set up of Mask if your data contains missing data.

### split the data into train and test sets
#Data_train, Data_test, mask_train, mask_test = train_test_split(
#        Data, Mask, test_size=0.1, random_state=rs)
Data_train = Data
mask_train = missing_mask
Data_test = Data
print('number of 0s (missing):', np.sum(1-mask_train))
print('number of 1s (observed) ', np.sum(mask_train))

### Train the model and save the trained model.
vae = train_p_vae(Data_train,mask_train, args.epochs, args.latent_dim, args.batch_size,args.p, args.K,args.iteration)

### Test imputating model on the test set
## Calculate test ELBO of observed test data (will load the pre-trained model). Note that this is NOT imputing.
#tf.reset_default_graph()
#test_loss = test_p_vae_marginal_elbo(Data_test,mask_test, args.latent_dim,args.K)
## Calculate imputation RMSE (conditioned on observed data. will load the pre-trained model)
## Note that here we perform imputation on a new dataset, whose observed entries are not used in training.
## this will under estimate the imputation performance, since in principle all observed entries should be used to train the model.
tf.reset_default_graph()
Data_ground_truth = Data_test
#mask_obs = np.array([bernoulli.rvs(1 - 0.3, size=Data_ground_truth.shape[1]*Data_ground_truth.shape[0])]) # manually create missing data on test set
#mask_obs = mask_obs.reshape(Data_ground_truth.shape)
mask_obs= missing_mask
Data_observed = Data_ground_truth*mask_obs

mask_target = 1-mask_obs # During test time, we use 1 to indicate missingness for imputing target.
# This line below is optional. Turn on this line means that we use the new comming testset to continue update the imputing model. Turn off this linea means that we only use the pre-trained model to impute without futher updating the model.
# vae = train_p_vae(Data_ground_truth,mask_obs, args.epochs, args.latent_dim, args.batch_size,0, args.K,args.iteration)
tf.reset_default_graph()
# Note that by default, the model calculate RMSE averaged over different imputing samples from partial vae.
RMSE,X_fill_mean_eddi = impute_p_vae(Data_observed,mask_obs,Data_ground_truth,mask_target,args.latent_dim,args.batch_size,args.K,args.M)
# Alternatively, you can also first compute the mean of partial vae imputing samples, then calculated RMSE.
# Diff_eddi=X_fill_mean_eddi*mask_target - Data*mask_target
Diff_eddi=X_fill_mean_eddi - Data

print('test impute RMSE eddi (estimation 2):', np.sqrt(np.sum(Diff_eddi ** 2 * mask_target)/np.sum(mask_target)))




