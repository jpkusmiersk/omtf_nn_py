import tensorflow as tf
from tensorflow import keras
import numpy as np
#from keras import backend
import math
import matplotlib.pyplot as plt

import LutInterLayer

#import numpy as np

#tf.random.set_seed(1234) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,
rng = np.random.RandomState(2021)
#rng = np.random.RandomState()


###################################################################################################

#test

# luts = tf.Variable([[[1, 2, 3],
#                      [4, 5, 6]], 
#                     [[11, 12, 13],
#                      [14, 15, 16]]])
#
# print ("luts", luts)
# print ("luts.shape", luts.shape)
#
# print ("luts[0]", luts[0])
# print ("luts[1]", luts[1])
#
# print("tf.version.VERSION", tf.version.VERSION)
#
# inputs = tf.Variable([[0, 1],
#                       [1, 2],
#                       [2, 1],
#                       [1, 0]]) 
#
# print ("inputs", inputs)
#
# get_out_values(inputs, luts)

#exit()


layer0_lut_size = 4 
num_inputs = 4 
layer0_num_outputs = 4

layer1_lut_size = 32
layer1_num_outputs = 4

layer2_lut_size = 32

eventCnt = 10000


print("starting generating data")    

input_data = rng.uniform(size=(eventCnt, num_inputs), low= -(layer0_lut_size-1)/2., high = (layer0_lut_size - 1 - 0.001)/2.)

y = tf.Variable(initial_value = tf.zeros(shape = [eventCnt, 1], dtype = tf.float32 ))

for i in range(0, eventCnt) :
    val = 0.
    #for j in range(0, num_inputs):
    #    val += input_data[i, j] * input_data[i, j] / 16.
    #y[i].assign(math.sin(val) )
    if input_data[i, 0] > input_data[i, 1] :
        y[i].assign(input_data[i, 2])
    else : 
        y[i].assign(input_data[i, 3])
    #print("y[", i, "] ", y[i])

print("input data generated")    
y = tf.constant(y)

#print("\ninput_data", input_data)

learning_rate=0.02
#optimizer = tf.keras.optimizers.SGD(learning_rate=1., momentum=0., )
#optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

initial_learning_rate = learning_rate
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
initial_learning_rate,
decay_steps=1600/4,
decay_rate=0.99,
staircase=False)

#optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0., )
#optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule) #learning_rate
#optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule)

#layer0_result = layer0(input = input_data)
#layer1_result = layer1(input = layer0_result)

rng = np.random.RandomState() # to have different seeds for initializer

lut_nn = True

model = keras.Sequential()

if lut_nn :
    #initializer = tf.keras.initializers.TruncatedNormal(mean = 0, stddev = layer1_lut_size/40., seed = 1234)
    initializer = LutInterLayer.LutInitializerLinear(maxLutVal = (layer1_lut_size-2)/layer0_num_outputs *20, initSlopeMin = 0.001, initSlopeMax = 4, lutRangesCnt = 1, reLu = False)
    layer0 = LutInterLayer.LutInterLayer("layer0", lut_size = layer0_lut_size, num_inputs = num_inputs, num_outputs = layer0_num_outputs, initializer = initializer)
    #layer0 = tf.keras.layers.Dense(layer0_num_outputs)
    
    #initializer = tf.keras.initializers.TruncatedNormal(mean = 0, stddev = 1/4., seed = 1234)
    initializer = LutInterLayer.LutInitializerLinear(maxLutVal = (layer2_lut_size-2)/layer1_num_outputs *20, initSlopeMin = 0.001, initSlopeMax = 0.2, lutRangesCnt = 1, reLu = False)
    layer1 = LutInterLayer.LutInterLayer("layer1", lut_size = layer1_lut_size, num_inputs = layer0_num_outputs, num_outputs = layer1_num_outputs, initializer = initializer)
    
    #initializer = tf.keras.initializers.TruncatedNormal(mean = 0, stddev = 1/4., seed = 1234)
    initializer = LutInterLayer.LutInitializerLinear(maxLutVal = layer0_lut_size * 4, initSlopeMin = 0.001, initSlopeMax = 1, lutRangesCnt = 1, reLu = False)
    layer2 = LutInterLayer.LutInterLayer("layer2", lut_size = layer2_lut_size, num_inputs = layer1_num_outputs, num_outputs = 1, initializer = initializer)
    #layer2 = tf.keras.layers.Dense(1) #, activation='sigmoid'
    
    print("building model")
    model.add(tf.keras.Input(shape = [num_inputs]))
    model.add(layer0)
    model.add(layer1)
    model.add(layer2)

else :
    print("building model")
    model.add(tf.keras.Input(shape = [num_inputs]))
    model.add(tf.keras.layers.Dense(4, activation='relu') )
    model.add(tf.keras.layers.Dense(4, activation='relu') )
    model.add(tf.keras.layers.Dense(1)) #, activation='sigmoid'
     
   
model(input_data)

model.compile(optimizer=optimizer, loss='mse')

model.summary() 

model.fit(input_data, y, epochs=1500, shuffle=True, batch_size = 100)

print("model.evaluate")


if lut_nn :
    for layer in model.layers:
        if isinstance(layer, LutInterLayer.LutInterLayer):
            layer.write_lut_hist =  True   

model.evaluate(input_data,  y, verbose=2)

#print("layer0.luts_float", layer0.luts_float)
#print("layer1.luts_float", layer1.luts_float)

output_array = model(input_data)
print("inputData.shape", input_data.shape)
print("output_array.shape", output_array.shape)

#print("inputData\n", inputData)
#print("output_array\n", output_array)
#print("y", y) 

#for i in range(0, eventCnt) :
#    print(y[i], output_array[i], math.fabs(y[i] - output_array[i]))

plotDir = "C:\\tf_logs\\plots\\"
    
plt.style.use('_mpl-gallery')
plt.rcParams['axes.labelsize'] = 6 
plt.rcParams['ytick.labelsize'] = 6 
plt.rcParams['xtick.labelsize'] = 6 
# plot
fig1, ax1 = plt.subplots(2, 2, figsize=(10, 10))

ax1[0, 0].scatter(y, output_array, linewidth=1.0, s=2)


#ax1[0, 1].scatter(input_data[: , 0], y, linewidth=1.0, s=9)

#ax1[1, 1].scatter(input_data[: , 0], output_array, linewidth=1.0, s=9)

ax1[0, 1].scatter(input_data[: , 0], input_data[: , 1], c = y, linewidth=1.0, s=4)

ax1[1, 1].scatter(input_data[: , 0], input_data[: , 1], c = output_array, linewidth=1.0, s=4)

plt.savefig(plotDir + "output.png", bbox_inches="tight")

#ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
#       ylim=(0, 8), yticks=np.arange(1, 8))

def plot_luts(lut_layer) :
    fig2, ax2 = plt.subplots(8, 4, sharex=True) #(y,x)
    
    lut_xlables =  list(range(lut_layer.luts_float.shape[2]))
    
    #print("lut_layer.entries_hist", lut_layer.entries_hist)
    #luts_float[num_outputs][num_inputs][lut_size]
    for i_out in range(0, min(ax2.shape[1], lut_layer.luts_float.shape[0])) :
        for i_in in range(0, min(ax2.shape[0], lut_layer.luts_float.shape[1])) :
            ax2[i_in, i_out].plot(lut_xlables, lut_layer.luts_float[i_out][i_in], linewidth=1.0)
            
            ax_entries = ax2[i_in, i_out].twinx() 
            ax_entries.scatter(lut_xlables, lut_layer.entries_hist[i_in], linewidth=1.0, s=9, color='tab:red')
            ax2[i_in, i_out].set_title('lut_layer.luts_float[' + str(i_out) + '][' + str(i_in) + ']', fontdict={'fontsize': 6})

if lut_nn :
    #plot_luts(layer0)
    #plot_luts(layer1)
    for layer in model.layers:
        if isinstance(layer, LutInterLayer.LutInterLayer):
            fig = layer.plot_luts(plotDir)    
            plt.close(fig)


#plt.margins(0.2)
#plt.subplots_adjust(hspace=0.4, wspace=0.17, left = 0.035, bottom = 0.045, top = 0.95)
#plt.show()



