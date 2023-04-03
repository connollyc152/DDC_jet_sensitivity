import tensorflow as tf
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import concatenate
from tensorflow.keras import regularizers
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D
import SETTINGS
settings = SETTINGS.EXPERIMENTS
import cmasher as cmr
tmap = cmr.viola #fusion_r

data_loc = "CHANGE TO LOCATION OF DATA"

def MakeContourPlot(data, title, y, pf):
    plt.figure(figsize=(7.5,6), dpi=300)
    X, Y = np.meshgrid(y, pf)
    maxi = np.max(np.abs(data))
    levels = np.linspace(-maxi, maxi, 17)
    cs = plt.contourf(X, Y, data ,cmap=tmap,levels = levels)
    plt.ylim(np.max(pf),np.min(pf))
    plt.colorbar(cs, format='%.02f', label = "")
    plt.title(title)
    plt.show()

def mae(test, predict):
    return np.mean(np.abs(np.ndarray.flatten(test)-np.ndarray.flatten(predict)))

def DeltaAO(AO_data, lag):
    AO_array = []
    for m in np.arange(0, (int(np.size(AO_data)) - lag), 1):
        AO_array.append((AO_data[m + lag] - AO_data[m]))
    AO_array = np.array(AO_array)
    return(AO_array)

def OpenAndCombineFiles(files):
    f = nc.Dataset(data_loc + "/runs_averaged/OneHemisphere_run" + str(files[0]) + "Runningavg" + str(run_avg) + "_lat.nc",'r')
    AO = np.array(f['output_ts'][:])
    temp = np.array(f['de_ts'][:,5:,:])
    y = np.array(f['y'][:])
    pf = np.array(f['pf'][5:])
    
    temp_data = temp
    AO_data = AO
    AO_labels = AO
      
    for file in files[1:]:
        f = nc.Dataset(data_loc +  "/runs_averaged/OneHemisphere_run" + str(file) + "Runningavg" + str(run_avg) + "_lat.nc",'r')
        AO = np.array(f['output_ts'][:])
        temp = np.array(f['de_ts'][:,5:,:])

        temp_data = np.row_stack((temp_data, temp))
        AO_data = np.concatenate((AO_data, AO))
        AO_labels = np.concatenate((AO_labels, AO))

    temp_data = np.array(temp_data)
    AO_data = np.array(AO_data)
    AO_labels = np.array(AO_labels)
    
    temp_data = temp_data[:-Nshift]
    AO_data = DeltaAO(AO_data,Nshift)
    AO_labels = AO_labels[:-Nshift]
    print("done Opening data")
    return(temp_data, AO_data, AO_labels, y, pf)

def RegressLossExpSigma(y_true, y_pred):    
    # network predictions
    mu = y_pred[:,0]
    sigma = y_pred[:,1]
    
    # normal distribution defined by N(mu,sigma)
    norm_dist = tfp.distributions.Normal(mu,sigma)
    # compute the log as the -log(p)
    loss = -norm_dist.log_prob(y_true[:,0])    
    return tf.reduce_mean(loss, axis=-1) 

class Exponentiate(keras.layers.Layer):
    """Custom layer to exp the sigma and tau estimates inline."""

    def __init__(self, **kwargs):
        super(Exponentiate, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.exp(inputs)

def compile_model(x_train, x_train2, y_train, settings):
    SEED = settings['seed']

    inputA = tf.keras.Input(shape = (25, 32, 1,))
    inputB = tf.keras.Input(shape = (1,))
    
    layers = inputA
    
    for blocks, filters, kernel, stride in zip(settings['CNN_blocks'], settings['CNN_filters'], settings['CNN_kernals'], settings['CNN_strides']):
        for layer in np.arange(0,blocks):
            layers = Conv2D(filters, activation=settings['activation_conv'],
                            kernel_size=(kernel, kernel), strides = (stride,stride), padding = settings['padding'],
                            kernel_initializer=keras.initializers.RandomNormal(seed=SEED))(layers)
        layers = tf.keras.layers.AveragePooling2D((2,2))(layers)

    layers = tf.keras.layers.Flatten()(layers)
    layers = concatenate([layers, inputB])

    for hidden, ridge, dropout in zip(settings['Dense_layers'], settings['ridge'], settings['dense_dropout']):
        layers = Dense(hidden, activation=settings['activation_dense'],
                       kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.00, l2=ridge),
                       bias_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                       kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED))(layers)
        layers = tf.keras.layers.Dropout(dropout)(layers)
        
    LOSS = RegressLossExpSigma
    metrics = ["mean_absolute_error"]


    mu_z_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
        name="mu_z_unit",
    )(layers)
    
    log_sigma_z_unit = tf.keras.layers.Dense(
        units=1,
        activation="linear",
        use_bias=True,
        bias_initializer=tf.keras.initializers.Zeros(),
        kernel_initializer=tf.keras.initializers.Zeros(),
        name="log_sigma_z_unit",
    )(layers)

    sigma_unit = Exponentiate(
        name="sigma_unit",
    )(log_sigma_z_unit)
    
    output_layer = tf.keras.layers.concatenate([mu_z_unit, sigma_unit], axis=1)

    model = tf.keras.models.Model([inputA, inputB], output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=settings['LR']), 
                  loss=LOSS, 
                  metrics=metrics,
                 )
            
    
    return model

def HistogramWHeightNormBySample(AO, error, bins_n, title, label):
    mini = np.min(AO)
    maxi = np.max(AO)
    bins = np.linspace(mini,maxi,bins_n)
    n = np.histogram(AO, bins = bins, weights=error)
    n2 = np.histogram(AO, bins = bins)

    n2[0][n2[0]==0] = 1
    values = n[0] / n2[0]

    n, bins1, patches = plt.hist(bins[:-1], weights=values, bins = bins, label=label)
    
    plt.xlabel("AO")
    plt.ylabel("MAE")
    plt.legend()
    plt.title(title)
    return(n,bins1)

def CreateAverageDeltaArray(bins, n, init_state):
    array = np.empty(shape=(np.size(init_state)))
    array.fill(9999)

    for s, sample in enumerate(init_state[:]):
        n2, bins1  = np.histogram(sample, bins = bins)
        
        if sample >= bins[-1:][0]:
            array[s] = list(n[-1:])[0]
            
        if sample <= bins[:1][0]:
            array[s] = list(n[:1])[0]

        if (sample < bins[-1:][0]) and (sample > bins[:1][0]):
            array[s] = list(n[n2==1])[0]

    array = np.array(array)
    return(array)   
     
####################3
modelsavepath = "/Users/cconn/Documents/DDC_jetsensitivity_CNN/CNN/models/"
experiment = ["FINAL_data"]

for exp in experiment:
    expsettings = settings[exp]
    Nshift =  expsettings['Nshift']
    run_avg = expsettings['run_avg']
    seed = expsettings['seed']
    np.random.seed(seed)
    tf.random.set_seed(seed)
    modelSaveName = (modelsavepath + expsettings['experiment'])
    es = EarlyStopping(monitor='val_loss',patience = expsettings['earlyStopping'])
    checkpoint = ModelCheckpoint(modelSaveName + ".h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=settings[exp]['checkpoint'])
    
    dataInTraino1, dataOutTraino1, AO_labelsTo1, y, pf = OpenAndCombineFiles([1, 2]) 
    dataInTraino2, dataOutTraino2, AO_labelsTo2, y, pf = OpenAndCombineFiles([11, 12]) 
    dataInTraino = np.row_stack((dataInTraino1, dataInTraino2))
    dataOutTraino = np.concatenate((dataOutTraino1, dataOutTraino2))
    AO_labelsTo = np.concatenate((AO_labelsTo1, AO_labelsTo2))
    
    weights = np.empty(shape = int(np.size(AO_labelsTo)))
    weights.fill(1)
    if expsettings['sampleweight'] == "dtdt_large":
        datain_mean_value = (np.max(np.abs(dataInTraino), axis = 0)) 
        temp_field_std = np.full_like(dataInTraino[0,:,:], datain_mean_value)
        dataInTraino_norm = (dataInTraino)/temp_field_std
             
        maxi = []
        mini = []
        for sample in dataInTraino_norm:
            maxi.append((np.max(sample)) + 2)
            mini.append((np.min(sample)) + 2)
        mag = (np.abs(mini)) + (maxi)
        weights = (mag) 
        weights[weights <= 1] = 1 
        plt.title("weights")
        plt.hist(weights)
        plt.show()
    if expsettings['sampleweight'] == "output":
        weights = np.abs(dataOutTraino) * 2
        weights[weights <= 1] = 1 
        plt.title("weights")
        plt.hist(weights)
        plt.show()
        plt.scatter(dataOutTraino,  weights, label='training')#, cmap='Reds', c=PredictedUncertainty, s = 5)
        plt.show()
    
    dataInValo1, dataOutValo1, AO_labelsVo1, y, pf = OpenAndCombineFiles(['3'])
    dataInValo2, dataOutValo2, AO_labelsVo2, y, pf = OpenAndCombineFiles(['13'])
    dataInValo = np.row_stack((dataInValo1, dataInValo2))
    dataOutValo = np.concatenate((dataOutValo1, dataOutValo2))
    AO_labelsVo = np.concatenate((AO_labelsVo1, AO_labelsVo2))

    label_mean = np.mean(AO_labelsTo, axis = 0)
    label_std = np.std(AO_labelsTo, axis = 0)
    AO_labelsTo_orig = AO_labelsTo
    AO_labelsTo = (AO_labelsTo - label_mean)/label_std
    AO_labelsVo = (AO_labelsVo - label_mean)/label_std
  
    temp_field_mean = np.full_like(dataInTraino[0,:,:], 0)
    temp_field_std = np.full_like(dataInTraino[0,:,:], .1)
    MakeContourPlot(dataInTraino[0,:,:], "", y, pf)
    dataInTrainostand = (dataInTraino - temp_field_mean)/temp_field_std
    dataInValostand = (dataInValo - temp_field_mean)/temp_field_std

    dataInTrainostand = dataInTrainostand.reshape(int(np.size(AO_labelsTo)), np.size(pf), np.size(y), 1)
    dataInValostand = dataInValostand.reshape(int(np.size(AO_labelsVo)),np.size(pf),np.size(y),1)

    out_mean = np.mean(dataOutTraino, axis = 0)
    out_std = np.std(dataOutTraino, axis = 0)
    dataOutTraino = (dataOutTraino - out_mean)/out_std
    dataOutValo = (dataOutValo - out_mean)/out_std
         
    dataOutTraino_2 = np.empty(shape = (np.size(dataOutTraino),2))
    dataOutTraino_2.fill(0)
    dataOutTraino_2[:,0] = dataOutTraino
    dataOutValo_2 = np.empty(shape = (np.size(dataOutValo),2))
    dataOutValo_2.fill(0)
    dataOutValo_2[:,0] = dataOutValo        
   
    if expsettings['SAVESTANDARDIZE']:
        ts = nc.Dataset(data_loc + "/standardize_files/standardizingFile_Nshift"+ str(Nshift) +"_"+ str(run_avg) +"runavg" + str(expsettings['TrainingFiles'])+ ".nc", 'w' , format='NETCDF4')
        ts_labelmean = ts.createDimension('ts_labelmean',1)
        ts_labelstd = ts.createDimension('ts_labelstd',1)
        ts_outmean = ts.createDimension('ts_outmean',1)
        ts_outstd = ts.createDimension('ts_outstd',1)
        ts_y = ts.createDimension('ts_y',np.size(y))

        ts_pf = ts.createDimension("ts_pf", np.size(pf))
         
        ts_labelmean = ts.createVariable('labelmean','f4',('ts_labelmean'))
        ts_labelstd = ts.createVariable('labelstd','f4',('ts_labelstd'))
        ts_outmean = ts.createVariable('outmean','f4',('ts_outmean'))
        ts_outstd = ts.createVariable('outstd','f4',('ts_outstd'))
        ts_inmean = ts.createVariable('ts_inmean','f4',('ts_pf','ts_y'))
        ts_instd = ts.createVariable('ts_instd','f4',('ts_pf','ts_y'))
        
        ts_labelmean[:] = label_mean
        ts_labelstd[:] = label_std
        ts_outmean[:] = out_mean
        ts_outstd[:] = out_std
        ts_inmean[:,:] = temp_field_mean
        ts_instd[:,:] = temp_field_std
    
        ts.close()
#######################################

    model = compile_model(dataInTrainostand, AO_labelsTo, dataOutTraino_2, expsettings)
    model.summary()

    print("Amount of training data " + str(np.size(AO_labelsTo)))
    print("Amount of validation data " + str(np.size(AO_labelsVo)))
    if expsettings['sampleweight'] == None:
        history = model.fit([dataInTrainostand, AO_labelsTo], dataOutTraino_2, batch_size = expsettings['batch_size'], 
                            epochs=expsettings['epochs'], shuffle=True, verbose=1, validation_data = ([dataInValostand,AO_labelsVo], dataOutValo_2),callbacks=[es,checkpoint]) 
    else:
        print("Weights")
        history = model.fit([dataInTrainostand, AO_labelsTo], dataOutTraino_2, batch_size = expsettings['batch_size'], 
                            epochs=expsettings['epochs'], shuffle=True, verbose=1, validation_data = ([dataInValostand,AO_labelsVo], dataOutValo_2),callbacks=[es,checkpoint], sample_weight = weights) 

    plt.figure(figsize = (16,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'],label = 'training', c="blue")
    plt.plot(history.history['val_loss'],label = 'val_loss' , c="orange")
    plt.xlabel('epoch')
    plt.legend(loc = "upper right")
    plt.show()

    plt.figure(figsize = (16,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'],label = 'training', c="blue")
    plt.plot(history.history['val_loss'],label = 'val_loss' , c="orange")
    plt.xlabel('epoch')
    plt.ylim(0,1.5)
    plt.legend(loc = "upper right")
    plt.show()

    dataInTesto, dataOutTesto, AO_labelsTo, y, pf = OpenAndCombineFiles([4])
    AO_labelsTostand = (AO_labelsTo - label_mean) / label_std
    dataOutTestostad = (dataOutTesto - out_mean)/out_std
    dataInTestostand = (dataInTesto - temp_field_mean) / temp_field_std
    dataInTestostand = dataInTestostand.reshape(int(np.size(AO_labelsTo)), 25, 32, 1)
    
    yTrainPred = model.predict([dataInTestostand,AO_labelsTostand])
    yTrainPredValue = (yTrainPred[:,0] * out_std) + out_mean
    yTrainPredUncertainty = yTrainPred[:,1]

    plt.figure(figsize=(6.5,6))
    plt.plot(dataOutTesto ,yTrainPredValue,'o',markersize=1, label='training', color = "blue")
    plt.plot([-10,10],[-10,10], color = 'gray')
    plt.ylim(-10,10)
    plt.xlim(-10,10)
    plt.legend()
    plt.xlabel('actual')
    plt.ylabel('prediction')
    plt.show()
    
    plt.hist(yTrainPredUncertainty)
    plt.show()
    
    plt.figure(figsize=(6.5,6))
    cs = plt.scatter(dataOutTesto,  yTrainPredValue, label='training', cmap='Reds', c=yTrainPredUncertainty, s = 5)
    plt.colorbar(cs)
    plt.plot([-10,10],[-10,10], color = 'gray')
    plt.ylim(-10,10)
    plt.xlim(-10,10)
    plt.legend()
    plt.xlabel('actual')
    plt.ylabel('prediction')
    plt.show()
    
