#!/usr/bin/env python
# coding: utf-8

# ## Environment setup

# In[7]:


import glob, os, time
from datetime import datetime
from functools import partial
import importlib

import numpy as np

import tensorflow as tf


# ## Networks definitions and adaptations

# In[8]:


from architecture_definitions import *

dir_postfix = get_classic_nn_dir_postfix() 
    
print_Classic_NN()  


# ### Training data set preparation

# In[9]:


import io_functions as io
importlib.reload(io)

batchSize = 4096
nEpochs = 1

#trainDataDir = "/scratch_ssd/akalinow/ProgrammingProjects/MachineLearning/OMTF/data/18_12_2020/"   
trainDataDir = "/home/kbunkow/cms_data/OMTF_data_2020/18_12_2020/"
trainFileNames = glob.glob(trainDataDir+'OMTFHits_pats0x0003_oldSample_files_*_chunk_0.tfrecord.gzip')

trainDataDir = "/scratch_cmsse/alibordi/data/training/"
trainFileNames = glob.glob(trainDataDir+'*OneOverPt*tfrecord.gzip')

dataset = io.get_Classic_NN_dataset(batchSize, nEpochs, trainFileNames, isTrain=True)

print("dataset", dataset)


# ### Model definition

# In[10]:


import model_functions as models
importlib.reload(models)

import io_functions as io
importlib.reload(io)

networkInputSize = 2 * np.sum(io.getFeaturesMask()) + 1
loss_fn = 'mape'
loss_fn = loss_MAPE_MAE

model = models.get_Classic_NN(networkInputSize=networkInputSize, loss_fn=loss_fn)
model.summary()


# ### The training loop

# In[11]:


from keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)


# In[ ]:


import time
from datetime import datetime
import os

current_time = datetime.now().strftime("%Y_%b_%d_%H_%M_%S")
print("Training start. Current Time =", current_time)

nEpochs = 150

log_dir = "logs/fit/" + current_time + dir_postfix
job_dir = "training/" + current_time + dir_postfix

checkpoint_path = job_dir + "/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=5085)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=(10, 20))
early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=5, verbose=1)
callbacks = [tensorboard_callback, cp_callback, early_stop_callback]

model.save_weights(checkpoint_path.format(epoch=0))

start_time = time.time()
model.fit(dataset.take(10),
          epochs=nEpochs, shuffle=True,
          callbacks=callbacks,
          validation_data=dataset.take(10))
end_time = time.time()

model.save(job_dir, save_format='tf')

current_time = datetime.now().strftime("%Y_%b_%d_%H_%M_%S")
print("Training end. Current Time =", current_time)
print("Total training time:", end_time - start_time, "seconds")


# In[ ]:





# In[ ]:




