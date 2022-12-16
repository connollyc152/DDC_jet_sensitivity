import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import netCDF4 as nc
import tensorflow as tf
import seaborn as sns
import scipy.stats as stats
import keras
import tensorflow_probability as tfp
import math as m
import matplotlib.cm as cm

import sys
sys.path.insert(0, '/Users/cconn/Documents/JetResponse_to_dTdt_PAPER/Code/AnalyzeModels')
import Plots as p
import PredictPlots as pp
sys.path.insert(0, '/Users/cconn/Documents/DDC_jetsensitivity_CNN/CNN')
import SETTINGS

from scipy.odr import *

data_loc = "/Users/cconn/Documents/DDC_jetsensitivity_CNN/CESM_DDC/data"

label_font = {'fontname':'Noto Sans JP', 'size':'12', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
title_font = {'fontname':'Not o Sans JP', 'size':'18', 'color':'black', 'weight':'bold',
              'verticalalignment':'bottom'}
axis_font = {'fontname':'Noto Sans JP', 'size':'13', 'color':'black', 'weight':'normal'}
colorbar_font = {'family':'Noto Sans JP', 'size':'10', 'color':'black', 'weight':'light'}
legend_font = {'family':'Noto Sans JP', 'size':'12', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
axisTick_font = {'family':'Noto Sans JP', 'size':'10', 'color':'black', 'weight':'light'}

#CREATE COLORBAR###################
lower = plt.cm.RdBu_r(np.linspace(0,.49, 49))
white = plt.cm.RdBu_r(np.ones(2)*0.5)
upper = plt.cm.RdBu_r(np.linspace(0.51, 1, 49))
colors = np.vstack((lower, white, upper))
tmap = matplotlib.colors.LinearSegmentedColormap.from_list('terrain_map_white', colors)
tmap_dis = matplotlib.colors.LinearSegmentedColormap.from_list('terrain_map_white', colors, N = 15)
# tmap = cmr.guppy_r #fusion_r
tmap_uncertainty = "bone" #cmr.savanna_r
tmap_uncertainty_dis = cm.get_cmap(tmap_uncertainty,lut=15)

ModelColor = "black"
SteadyEvolutionColor = "#35D6E8"
PersistenceColor = "orange"


def CreateGaussianBlob(array, y_loc, pf_loc, y_gsize, pf_gsize, anom):
    HEATING_blab = np.empty(shape = (pf_size, y_size))
    HEATING_blab.fill(0)
    for l, lat in enumerate(y):
        for h, height in enumerate(pf): 
            HEATING_blab[h,l] = HEATING_blab[h,l] + ((m.exp(-((lat - y_loc)**2/(y_gsize)**2) - ((height - pf_loc)**2/(pf_gsize)**2))) * (anom))   
    return(array + HEATING_blab)

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

def mae(test, predict): 
    return np.abs((test-predict))

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

def HistogramWHeightNormBySample_COUNT(AO, error, bins_n, title, label):
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
    return(n,bins1,n2[0])

def DeltaAO(AO_data, lag):
    AO_array = []
    for mi in np.arange(0, (int(np.size(AO_data)) - lag), 1):
        AO_array.append((AO_data[mi + lag] - AO_data[mi]))
    AO_array = np.array(AO_array)
    return(AO_array)

n = 1000
def OpenAndCombineFiles(files):
    f = nc.Dataset(data_loc + "/runs_averaged/OneHemisphere_run" + str(files[0]) + "Runningavg" + str(run_avg) + "_lat.nc",'r')
    print("opening file " + "OneHemisphere_run" + str(files[0]) + "Runningavg" + str(run_avg) + "_lat.nc")
    AO = np.array(f['output_ts'][:n])
    temp = np.array(f['de_ts'][:n,5:,:])
    y = np.array(f['y'][:])
    pf = np.array(f['pf'][5:])
    
    temp_data = temp
    AO_data = AO
    AO_labels = AO
      
    for file in files[1:]:
        f = nc.Dataset(data_loc + "/runs_averaged/OneHemisphere_run" + str(file) + "Runningavg" + str(run_avg) + "_lat.nc",'r')
        print("opening file " + str(file))
        AO = np.array(f['output_ts'][:n])
        temp = np.array(f['de_ts'][:n,5:,:])

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

def plot_pits(x_val, onehot_val, model_shash):
    clr_shash = 'tab:blue'
    bins, hist_shash, D_shash, EDp_shash = compute_pit(onehot_val, x_data=x_val,model_shash=model_shash)
    bins_inc = bins[1]-bins[0]
    bin_add = bins_inc/2
    bin_width = bins_inc*.98
    plt.figure(dpi = 300)
    plt.bar(hist_shash[1][:-1] + bin_add,
             hist_shash[0],
             width=bin_width,
             color=clr_shash,
             label='SHASH',
            )
    plt.axhline(y=.1,
                linestyle='--',
                color='k',
                linewidth=2.,
               )
    yticks = np.around(np.arange(0,1,.05),2)
    plt.yticks(yticks,yticks)
    plt.ylim(0,.2)
    plt.xticks(bins,np.around(bins,1))
    plt.text(0.,np.max(plt.ylim())*.99,
             'SHASH D: ' + str(np.round(D_shash,4)) + ' (' + str(np.round(EDp_shash,4)) +  ')',
             color='black',#clr_shash,
             verticalalignment='top',
             fontsize=12)
    plt.xlabel('probability integral transform')
    plt.ylabel('probability')
    plt.show()

def compute_pit(onehot_data, x_data, model_shash):
    bins = np.linspace(0, 1, 11)
    shash_pred = model_shash.predict(x_data)
    mu_pred = shash_pred[:,0]
    sigma_pred = shash_pred[:,1]
    norm_dist = tfp.distributions.Normal(mu_pred,sigma_pred)
    F = norm_dist.cdf(onehot_data)
    pit_hist = np.histogram(F,
                              bins,
                              weights=np.ones_like(F)/float(len(F)),
                             )
    # pit metric from Bourdin et al. (2014) and Nipen and Stull (2011)
    # compute expected deviation of PIT for a perfect forecast
    B   = len(pit_hist[0])
    D   = np.sqrt(1/B * np.sum( (pit_hist[0] - 1/B)**2 ))
    EDp = np.sqrt( (1.-1/B) / (onehot_data.shape[0]*B) )
    return bins, pit_hist, D, EDp


modelsavepath = "/Users/cconn/Documents/DDC_jetsensitivity_CNN/CNN/models/"
experiment = "smalltrial" #################################
settings = SETTINGS.EXPERIMENTS[experiment]
seed = settings['seed']
np.random.seed(seed)
tf.random.set_seed(seed)
Nshift =  settings['Nshift']
run_avg = settings['run_avg']

#############################################
model_name = modelsavepath + settings['experiment'] + '.h5'
new_model = tf.keras.models.load_model(model_name, custom_objects={"Exponentiate": Exponentiate, "RegressLossExpSigma": RegressLossExpSigma})
new_model.summary()
tf.keras.utils.plot_model(new_model, 'model.png', show_shapes=True, dpi=100)

ODRreg = True
OPENTRAININGDATA = False
AVERAGECHANGE = True
SAVEFIGURES = False
figuresavepath = "/Users/cconn/Documents/DDC_jetsensitivity_CNN/saved_figures/"

###PLOTS######
SCATTER = True
BASELINESCATTER = True
SCATTERCONTOUR = True
SCATTERUNCERTAINTY = True

# PREDICTEDSIGN = True
SAMPLELIST = True
sample_list = [100, 300,3750,8640,1753, 1900] #list(np.arange(104,300,1)) #65672,3726,1278,9385,13098,14000

PITHISTOGRAM = True
# BOXANDWHISKER = True

HISTWITHINERROR = True
HISTERROR_BASEDONLABELS = True
HISTERROR_BASEDONDELTAAO = True
HISTVIOLIN = True

###PredictPLots##########
GAUSSIANBLAB = True
PULLINGSAMPLESADDGAUSIAN = True
GAUSSIANBLABLINEAR = True
TESTGAUSSIANS = True

CNN_AE_COUNT = True
POLESURFACESIGN = True
PDF_OF_PREDICTED = True

dataInTest, dataOutTest, AO_labelsT, y, pf = OpenAndCombineFiles([4,5,6,7,8,9,10]) #,5,6,7,8,9,10
dataInTest2, dataOutTest2, AO_labelsT2, y, pf = OpenAndCombineFiles([14,15,16,17,18,19,110 ]) #,15,16,17,18,19,110 
y_size = np.size(y)
pf_size = np.size(pf)
dataInTest = np.row_stack((dataInTest, dataInTest2))
dataOutTest = np.concatenate((dataOutTest, dataOutTest2))
AO_labelsT = np.concatenate((AO_labelsT, AO_labelsT2))
print("Amount of data currently used: " + str(np.size(AO_labelsT)))

f = nc.Dataset(data_loc + "/standardize_files/standardizingFile_Nshift"+ str(Nshift) +"_"+ str(run_avg) +"runavg" + str(settings['TrainingFiles'])+ ".nc",'r')
print("Opening " +  data_loc + "/standardize_files/standardizingFile_Nshift"+ str(Nshift) +"_"+ str(run_avg) +"runavg" + str(settings['TrainingFiles'])+ ".nc")
AO_labels_mean, AO_labels_std, dataOut_mean, dataOut_std = f['labelmean'][0], f['labelstd'][0], f['outmean'][0], f['outstd'][0]
in_mean, in_std = f['ts_inmean'][:,:], f['ts_instd'][:,:]
f.close()

AO_labelsT_stand = (AO_labelsT - AO_labels_mean)/AO_labels_std 
dataInTest_stand = (dataInTest - in_mean)/in_std
dataOutTest_stand = (dataOutTest - dataOut_mean)/dataOut_std

dataInTestF = dataInTest_stand.reshape(-1, pf_size, y_size, 1)
print("PREDICTING")
PredictedData = new_model.predict([dataInTestF,AO_labelsT_stand])
PredictedValue = (PredictedData[:,0] * dataOut_std) + dataOut_mean
PredictedUncertainty = PredictedData[:,1]

if OPENTRAININGDATA:
    ("OPENING TRAINING DATA")
    dataInTrainTr, dataOutTrainT, AO_labelsTr, y, pf = OpenAndCombineFiles(['1','2']) 
    dataInTrainTr2, dataOutTrainT2, AO_labelsTr2, y, pf = OpenAndCombineFiles(['11','12']) 
    dataInTrainTr = np.row_stack((dataInTrainTr, dataInTrainTr2))
    dataOutTrainT = np.concatenate((dataOutTrainT, dataOutTrainT2))
    AO_labelsTr = np.concatenate((AO_labelsTr, AO_labelsTr2))
    
    AO_labelsTr_stand = (AO_labelsTr - AO_labels_mean)/AO_labels_std
    dataInTrainTr_stand = (dataInTrainTr - in_mean)/in_std
    del(AO_labelsTr)

#Create Persistence Baseline
perpred = np.empty(shape = (int(np.size(AO_labelsT))))
perpred.fill(0)

if AVERAGECHANGE:
    #Create Average Evolution Baseline
    print("CALCUALTING AVERAGE CHANGE")
    dataInTrainT, dataOutTrainT, AO_labelsTT, y, pf = OpenAndCombineFiles([1, 2])
    dataInTrainT2, dataOutTrainT2, AO_labelsTT2, y, pf = OpenAndCombineFiles([11, 12])
    dataInTrainT = np.row_stack((dataInTrainT, dataInTrainT2))
    dataOutTrainT = np.concatenate((dataOutTrainT, dataOutTrainT2))
    AO_labelsTT = np.concatenate((AO_labelsTT, AO_labelsTT2))
    
    n, bins = HistogramWHeightNormBySample(AO_labelsTT, dataOutTrainT, 100, "Average True AO from Data", "data")
    average_Delta = CreateAverageDeltaArray(bins, n, AO_labelsT)#[:,0]
    plt.show()

if ODRreg:    
    def f(B, x):
        return B[0]*x + B[1]
    
    linear = Model(f)
    mydata = RealData(dataOutTest, PredictedValue, sy=PredictedUncertainty)
    myodr = ODR(mydata, linear, beta0=[1., 2.])
    myoutput = myodr.run()
    myoutput.pprint()
    slope = myoutput.beta[0]
    y_intercept = myoutput.beta[1]

if PITHISTOGRAM:
    plot_pits(x_val=[dataInTestF,AO_labelsT_stand], onehot_val=dataOutTest_stand, model_shash=new_model)    
    
############################
#####CALCULATIONS###########
############################

bins = np.arange(26, 52, 2)
min_bin = 26
max_bin = 50
n_bins = (np.size(bins))

if CNN_AE_COUNT: 
    #Counts how many time the CNN is better than average evolution
    bins = np.linspace(np.min(AO_labelsT),np.max(AO_labelsT),n_bins) 
    bins = [27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
    MAE_array_per_CNN = mae(PredictedValue,dataOutTest)
    MAE_array_per_SE = mae(average_Delta,dataOutTest)
    CNN_W = 0
    CNN_L = 0
    for pr, pred in enumerate(MAE_array_per_CNN):
        if MAE_array_per_CNN[pr] > MAE_array_per_SE[pr]:
            CNN_L = CNN_L + 1
        if MAE_array_per_CNN[pr] < MAE_array_per_SE[pr]:
            CNN_W = CNN_W + 1
    for b, bin_i in enumerate(bins[:-1]):
        CNN_W2 = 0
        CNN_L2 = 0
        bin_data_CNN = MAE_array_per_CNN[(bins[b + 1] > AO_labelsT)&(AO_labelsT >= bins[b])]
        bin_data_SE = MAE_array_per_SE[(bins[b + 1] > AO_labelsT)&(AO_labelsT >= bins[b])]
        print(np.shape(bin_data_CNN))
        for pr, pred in enumerate(bin_data_CNN):
            if bin_data_CNN[pr] > bin_data_SE[pr]:
                CNN_L2 = CNN_L2 + 1
            if bin_data_CNN[pr] < bin_data_SE[pr]:
                CNN_W2 = CNN_W2 + 1 
        print("CNN is better than SE " + str(CNN_W2 - CNN_L2))  
        
    ###PERCENT CORRECT SIGN
    PredictedValue_sign = PredictedValue/np.abs(PredictedValue)
    average_Delta_sign = average_Delta/np.abs(average_Delta)
    dataOutTest_sign = dataOutTest/np.abs(dataOutTest)
    CNN_correct_sign = 0
    SE_correct_sign = 0
    for pr, pred in enumerate(PredictedValue_sign):
        if PredictedValue_sign[pr] == dataOutTest_sign[pr]:
            CNN_correct_sign = CNN_correct_sign + 1
        if average_Delta_sign[pr] == dataOutTest_sign[pr]:
            SE_correct_sign = SE_correct_sign + 1
    for b, bin_i in enumerate(bins[:-1]):
        CNN_correct_sign2 = 0
        SE_correct_sign2 = 0
        bin_data_CNN = PredictedValue_sign[(bins[b + 1] > AO_labelsT)&(AO_labelsT >= bins[b])]
        bin_data_SE = average_Delta_sign[(bins[b + 1] > AO_labelsT)&(AO_labelsT >= bins[b])]
        bin_data_truth = dataOutTest_sign[(bins[b + 1] > AO_labelsT)&(AO_labelsT >= bins[b])]
        for pr, pred in enumerate(bin_data_truth):
            if bin_data_CNN[pr] == bin_data_truth[pr]:
                CNN_correct_sign2 = CNN_correct_sign2 + 1
            if bin_data_SE[pr] == bin_data_truth[pr]:
                SE_correct_sign2 = SE_correct_sign2 + 1 
        print("CNN is correct more by " + str(CNN_correct_sign2 - SE_correct_sign2))
    print(CNN_correct_sign)
    print(SE_correct_sign)
    
if SCATTER:
    p.ScatterPlot(PredictedValue, dataOutTest, "Prediction Versus Truth")
    if SAVEFIGURES: 
        plt.savefig(figuresavepath + "ScatterPlot.png")
    plt.show()

if SCATTERUNCERTAINTY:
    plt.figure(figsize=(6.5,6))
    cs = plt.scatter(dataOutTest,  PredictedValue, label='training', cmap='Reds', c=PredictedUncertainty, s = 5)
    plt.colorbar(cs)
    plt.plot([-10,10],[-10,10], color = 'gray')
    plt.ylim(-10,10)
    plt.xlim(-10,10)
    plt.legend()
    plt.xlabel('actual')
    plt.ylabel('prediction')
    plt.show()

    
if SCATTERCONTOUR:
    p.TwoD_PDF(PredictedValue, dataOutTest)
   
if BASELINESCATTER:
    predicted_list = [PredictedValue, average_Delta, perpred]
    title_list = ["CNN", "Average Evolution", "Persistence"]
    p.ThreeByOnePanelScatter(dataOutTest, predicted_list, title_list)
    
    if SAVEFIGURES: 
        plt.savefig(figuresavepath + "CombinedScattering.png")    
    
if HISTVIOLIN:
    print(np.min(AO_labelsT),np.max(AO_labelsT))
    shift = (bins[1] - bins[0])/2
    MAE_array_per = mae(PredictedValue,dataOutTest) 
    bin_data_dict = {-1:1}
    
    fig, ax1 = plt.subplots(figsize=(12,5), dpi=400)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    MAE_array_per = mae(perpred,dataOutTest)
    p.HistogramWHeightNormBySample(AO_labelsT, MAE_array_per, bins, '', "Persistence", PersistenceColor, 2, True)
    
    MAE_array_per = mae(average_Delta,dataOutTest)
    p.HistogramWHeightNormBySample(AO_labelsT, MAE_array_per, bins, '', "Average Evolution", SteadyEvolutionColor, 2, False)
    
    MAE_array_per = mae(PredictedValue,dataOutTest)
    p.HistogramWHeightNormBySample(AO_labelsT, MAE_array_per, bins, '', "CNN", ModelColor, 2, False)
      
   
    plt.title("Error based on Initial Latitude", **title_font)
    plt.xlabel("Initial Latitude", labelpad=5, **axis_font)
    plt.ylabel("Mean Absolute Error", **axis_font)
    
    for b, bin_i in enumerate(bins[:-1]):
        bin_data = MAE_array_per[(bins[b + 1] > AO_labelsT)&(AO_labelsT >= bins[b])]
        bin_data_dict[str(b)] = None
        bin_data_dict[str(b)] = (MAE_array_per[(bins[b + 1] > AO_labelsT)&(AO_labelsT >= bins[b])])
        if np.size(bin_data) == 0:
            continue
        else:
            violin_parts = plt.violinplot(positions = [(bin_i + shift)], dataset = [bin_data_dict[str(b)]], widths = [(shift * 2)], showmedians=True)
            violin_parts['cmins'].set_visible(False)
            violin_parts['cmaxes'].set_visible(False)
            violin_parts['cbars'].set_visible(False)
            violin_parts['cmedians'].set_visible(False)
            for pc in violin_parts['bodies']:
                pc.set_facecolor('grey')
                pc.set_linewidth(0)
    
    plt.ylim(0,7.2) 
    handles, labels = plt.gca().get_legend_handles_labels()

    order = [2,1,0]

    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
               loc='upper right', prop={'family':'Assistant', 'size':'12'})
    
    x1, y1 = [42.4, 42.4], [0,.1]
    plt.plot(x1, y1, c = "black",  alpha=0.3, linewidth = .5)    
    
    if SAVEFIGURES: 
        plt.savefig(figuresavepath + "Violin.png")           
    plt.show()
    
                
if HISTERROR_BASEDONDELTAAO:
    fig, ax1 = plt.subplots(figsize=(8,5), dpi=400)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    MAE_array_per = mae(perpred,dataOutTest)
    p.HistogramWHeightNormBySample(AO_labelsT, MAE_array_per, bins, '', "Persistence", PersistenceColor, 2, False)

    MAE_array_per = mae(average_Delta,dataOutTest)
    p.HistogramWHeightNormBySample(AO_labelsT, MAE_array_per, bins, '', "Steady Evolution", SteadyEvolutionColor, 2, False)

    # MAE_array_per = mae(PredictedValue,dataOutTest)
    MAE_array_per = mae(PredictedValue,dataOutTest)
    p.HistogramWHeightNormBySample(AO_labelsT, MAE_array_per, bins, '', "CNN", ModelColor, 2, True)
    plt.title("Average Absolute Error based on Initial Latitude", **title_font)
    plt.xlabel("Initial Latitude", labelpad=5, **axis_font)
    
    ax2 = ax1.twinx()
    ax2.set_ylim(0,2.5)
    ax2.set_xlim(24,51)
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    sns.kdeplot(ax=ax2, data = AO_labelsT, color = 'grey', linewidth=1.5)
    if SAVEFIGURES: 
        plt.savefig(figuresavepath + "BaselinesInitial.png")
    plt.show()

if HISTERROR_BASEDONLABELS:
    bins = np.arange(-10, 10, 2)
    fig, ax1 = plt.subplots(figsize=(8,5),dpi=400)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    MAE_array_per = mae(perpred,dataOutTest)
    p.HistogramWHeightNormBySample(dataOutTest, MAE_array_per[:], bins, '', "Persistence", PersistenceColor, 2, False)

    MAE_array_per = mae(average_Delta,dataOutTest)
    p.HistogramWHeightNormBySample(dataOutTest, MAE_array_per[:], bins, '', "Steady Evolution", SteadyEvolutionColor, 2, False)

    MAE_array_per = mae(PredictedValue ,dataOutTest)
    p.HistogramWHeightNormBySample(dataOutTest, MAE_array_per[:], bins, '', "CNN", ModelColor, 2, False)
    plt.title("Average Absolute Error based on Truth", **title_font)
    plt.xlabel("True Latitude", labelpad=5, **axis_font)
    
    ax2 = ax1.twinx()
    ax2.set_ylim(0,2.5)
    ax2.set_xlim(-11, 11)
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    sns.kdeplot(ax=ax2, data = dataOutTest, color = 'grey', linewidth=1.5)
    if SAVEFIGURES: 
        plt.savefig(figuresavepath + "BaselinesTruth.png")
    plt.show()
        
if GAUSSIANBLAB:
    print(np.max(AO_labelsT_stand))
    print(np.min(AO_labelsT_stand))
    

    returnedArray = pp.GaussianBlab(.25, new_model, y, pf, 0 , y_size, pf_size, dataOut_mean, dataOut_std, in_mean, in_std, 0)
    # returnedArray = pp.GaussianBlab(.75, new_model, y, pf, 0 , y_size, pf_size, dataOut_mean, dataOut_std, in_mean, in_std, 0)
    # returnedArray = pp.GaussianBlab(.5, new_model, y, pf, 0 , y_size, pf_size, dataOut_mean, dataOut_std, in_mean, in_std, 0)
    # returnedArray = pp.GaussianBlab(.25, new_model, y, pf, 0 , y_size, pf_size, dataOut_mean, dataOut_std, in_mean, in_std, 0)

    if SAVEFIGURES: 
        plt.savefig(figuresavepath + "MovingGaussianBlob.png")
    plt.show()
        
if GAUSSIANBLABLINEAR:
    anomalies = [-1,-.8,-.6,-.4,-.2,.2,.4,.6,.8,1]
    loc = np.empty(shape = (pf_size, y_size))
    latSize = 15
    heightSize = 200
    label = np.empty(shape=(1))
    label.fill(0)
    y_points = [12, 75, 45, 12, 75]
    pf_points = [850, 850, 550, 300, 300]
    
    loc1_array = np.empty(shape=(np.size(anomalies)))
    loc2_array = np.empty(shape=(np.size(anomalies)))
    loc3_array = np.empty(shape=(np.size(anomalies)))
    loc4_array = np.empty(shape=(np.size(anomalies)))
    loc5_array = np.empty(shape=(np.size(anomalies)))
    
    plt.figure(figsize=(7.5,6), dpi=300)
    cs = plt.scatter(y_points, pf_points, color = ["#E61B00", "#010FE6", "#0AE0E6", "#DE00E6", "#E6E100"], s = 100)
    plt.ylim(np.max(pf),np.min(pf))
    plt.xlim(np.min(y),np.max(y))
    
    loc.fill(0)
    loc = CreateGaussianBlob(loc, y_points[0],  pf_points[0], latSize, heightSize, 1)
    xs = plt.contour(y, pf, loc, colors="#E61B00", levels = [0, .5, .75])#, linestyles='dashed')
    plt.clabel(xs, inline=True, fontsize=10)
    
    loc.fill(0)
    loc = CreateGaussianBlob(loc, y_points[1],  pf_points[1], latSize, heightSize, 1)
    xs = plt.contour(y, pf, loc, colors='#010FE6', levels = [0, .5, .75])#, linestyles='dashed')
    plt.clabel(xs, inline=True, fontsize=10)
    
    loc.fill(0)
    loc = CreateGaussianBlob(loc, y_points[2],  pf_points[2], latSize, heightSize, 1)
    xs = plt.contour(y, pf, loc, colors='#0AE0E6', levels = [0, .5, .75])#, linestyles='dashed')
    plt.clabel(xs, inline=True, fontsize=10)
    
    loc.fill(0)
    loc = CreateGaussianBlob(loc, y_points[3],  pf_points[3], latSize, heightSize, 1)
    xs = plt.contour(y, pf, loc, colors='#DE00E6', levels = [0, .5, .75])#, linestyles='dashed')
    plt.clabel(xs, inline=True, fontsize=10)
    
    loc.fill(0)
    loc = CreateGaussianBlob(loc, y_points[4],  pf_points[4], latSize, heightSize, 1)
    xs = plt.contour(y, pf, loc, colors='#E6E100', levels = [0, .5, .75])#, linestyles='dashed')
    plt.clabel(xs, inline=True, fontsize=10)
    plt.show()
    
    for a, anom in enumerate(anomalies):
        loc.fill(0)
        loc = CreateGaussianBlob(loc, y_points[0],  pf_points[0], latSize, heightSize, anom)
               
        HEATING_blab_stand1 = (loc - in_mean)/ in_std
        HEATING_blabF1 = HEATING_blab_stand1.reshape(-1, 25, 32, 1)
        
        predi = new_model.predict([HEATING_blabF1,label])
        loc1_array[a] = (predi[:,0] * dataOut_std) + dataOut_mean
        
        ##############################
        loc.fill(0)
        loc = CreateGaussianBlob(loc, y_points[1],  pf_points[1], latSize, heightSize, anom)
               
        HEATING_blab_stand1 = (loc - in_mean)/ in_std
        HEATING_blabF1 = HEATING_blab_stand1.reshape(-1, 25, 32, 1)
        
        predi = new_model.predict([HEATING_blabF1,label])
        loc2_array[a] = (predi[:,0] * dataOut_std) + dataOut_mean

        ##############################
        loc.fill(0)
        loc = CreateGaussianBlob(loc, y_points[2],  pf_points[2], latSize, heightSize, anom)
               
        HEATING_blab_stand1 = (loc - in_mean)/ in_std
        HEATING_blabF1 = HEATING_blab_stand1.reshape(-1, 25, 32, 1)
        
        predi = new_model.predict([HEATING_blabF1,label])
        loc3_array[a] = (predi[:,0] * dataOut_std) + dataOut_mean
        
        ##############################
        loc.fill(0)
        loc = CreateGaussianBlob(loc, y_points[3],  pf_points[3], latSize, heightSize, anom)
               
        HEATING_blab_stand1 = (loc - in_mean)/ in_std
        HEATING_blabF1 = HEATING_blab_stand1.reshape(-1, 25, 32, 1)
        
        predi = new_model.predict([HEATING_blabF1,label])
        loc4_array[a] = (predi[:,0] * dataOut_std) + dataOut_mean
    
        ##############################
        loc.fill(0)
        loc = CreateGaussianBlob(loc, y_points[4],  pf_points[4], latSize, heightSize, anom)
               
        HEATING_blab_stand1 = (loc - in_mean)/ in_std
        HEATING_blabF1 = HEATING_blab_stand1.reshape(-1, 25, 32, 1)
        
        predi = new_model.predict([HEATING_blabF1,label])
        loc5_array[a] = (predi[:,0] * dataOut_std) + dataOut_mean
    
    plt.figure(figsize=(8,2), dpi=600)
    plt.axvline(x = 4.5, color = "black",  alpha=0.7)
    plt.plot(loc3_array, color = '#0AE0E6')
    plt.plot([int(np.size(anomalies) - 1), 0],[loc3_array[-1:], loc3_array[0]], color = '#0AE0E6', linestyle='--',  alpha=0.3)
    plt.plot(loc4_array, color = '#DE00E6')
    plt.plot([int(np.size(anomalies) - 1), 0],[loc4_array[-1:], loc4_array[0]], color = '#DE00E6', linestyle='--',  alpha=0.3)
    plt.xticks(ticks = np.arange(0, np.size(anomalies), 1), labels = anomalies, **axisTick_font)
    plt.yticks(**axisTick_font)
    plt.xlim(0,(np.size(anomalies) - 1))
    plt.show()
    
    plt.figure(figsize=(8,2), dpi=600)
    plt.axvline(x = 4.5, color = "black",  alpha=0.7) 
    plt.plot(loc1_array, color = '#E61B00')
    plt.plot([int(np.size(anomalies) - 1), 0],[loc1_array[-1:], loc1_array[0]], color = "#E61B00", linestyle='--',  alpha=0.3)
    plt.plot(loc2_array, color = '#010FE6')
    plt.plot([int(np.size(anomalies) - 1), 0],[loc2_array[-1:], loc2_array[0]], color = '#010FE6', linestyle='--',  alpha=0.3)
    plt.plot(loc5_array, color = '#E6E100')
    plt.plot([int(np.size(anomalies) - 1), 0],[loc5_array[-1:], loc5_array[0]], color = '#E6E100', linestyle='--',  alpha=0.3)
    plt.xticks(ticks = np.arange(0, np.size(anomalies), 1), labels = anomalies, **axisTick_font)
    plt.yticks(**axisTick_font)
    plt.xlim(0,(np.size(anomalies) - 1))
    plt.show()    

if HISTWITHINERROR:
    errorlist = [0.05, .1, .25, .5, .75, 1]
    for errorbound in errorlist:
        plt.figure(figsize=(6.5,6))
        MAE_array_per = mae(dataOutTest,perpred)
        data_Below_error = AO_labelsT[MAE_array_per < errorbound]
        p.Histogram(data_Below_error, 10, "", "Persistence", PersistenceColor, 2)
    
        MAE_array_per = mae(dataOutTest,average_Delta)
        data_Below_error = AO_labelsT[MAE_array_per < errorbound]
        nS, bins1S, patchesS = p.Histogram(data_Below_error, 10, "", "Steady Evolution", SteadyEvolutionColor, 2)
        
        MAE_array_per = mae(dataOutTest,PredictedValue)
        data_Below_error = AO_labelsT[MAE_array_per < errorbound]
        print("How much data falls under " + str(errorbound))
        print(str(np.size(data_Below_error)))
        nT, bins1T, patchesT = p.Histogram(data_Below_error, 10, "", "CNN", ModelColor, 1)
        plt.title("How many samples had an error less than " + str(errorbound))
        plt.show()    

if SAMPLELIST:
    p.CreateSamplePlot(sample_list, dataInTest, dataOutTest, AO_labelsT, PredictedValue, y, pf, Nshift)

if TESTGAUSSIANS:
    print("TESTING GAUSSIANS")
    dtdt = np.empty(shape = (pf_size, y_size))
    label = np.empty(shape=(1))
    label.fill(0) 
    
    LAT = [30, 30, 60, 60]
    LAT_S = [15, 15, 15, 15]
    PRES = [1000, 700, 1000, 700]
    PRES_S = [200, 200,200, 200]
    Q = [.25, .25, .25, .25]

    for g, gaus in enumerate(LAT):
        dtdt.fill(0)
        dtdt = CreateGaussianBlob(dtdt, LAT[g], PRES[g], LAT_S[g], PRES_S[g], Q[g])
        
        HEATING_blab_stand1 = (dtdt - in_mean)/ in_std
        HEATING_blabF1 = HEATING_blab_stand1.reshape(-1, pf_size, y_size, 1)
    
        predi = new_model.predict([HEATING_blabF1,label])
        print("Prediction from gaussian " + str(g))
        print((predi[:,0] * dataOut_std) + dataOut_mean)
        print(predi[:,1])

############################
#####HEATINGS###############
############################ 
EXPLORE_SINGLE_HEATING_MAGNITUDE = True
EXPLORE_TWO_HEATING_MAGNITUDE = True 
TwoHeatingNonlinear = True
LINEARTWO = True
TUGOFWAREXP = True
POLARPATTERNLINE = True
Doyeon_Kim = True 
BUTLER2010 = True

def TwoHeatingSmall(y, pf, initloc):
    print("COMPLETING EXPLORE TWO HEATING MAGNITUE")
    ADD_DDC_DATA = False
    Trials_x = [-1.2, -1.1, -1, -.9, -.8, -.7, -.6, -.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.2] #np.arange(-1.2, 1.4, interval_x)
    Trials_y = [-.2, -.15, -.10, -.05, 0, .05, .1, .15, .2]#np.arange(-.2, .21, interval_y)
    HEATING_pred = np.empty(shape=(np.size(Trials_y), np.size(Trials_x)))
    HEATING_pred.fill(0)
    HEATING_uncertainty = np.empty(shape=(np.size(Trials_y), np.size(Trials_x)))
    HEATING_uncertainty.fill(0)
    HEATING_blab = np.empty(shape = (pf_size, y_size))
    for TH, tropic_heating in enumerate(Trials_y):
        for PH, polar_heating in enumerate(Trials_x):
            HEATING_blab.fill(0)
            for l, lat in enumerate(y):
                for h, height in enumerate(pf): 
                    HEATING_blab[h,l] = HEATING_blab[h,l] + ((m.exp(-((Arctic_lat - lat)**2/(Arctic_latSize)**2) - ((Arctic_height - height)**2/(Arcitc_heightSize)**2))) * (polar_heating))
                    HEATING_blab[h,l] = HEATING_blab[h,l] + ((m.exp(-((Tropic_lat - lat)**2/(Tropic_latSize)**2) - ((Tropic_height - height)**2/(Tropic_heightSize)**2))) * (tropic_heating))
            
            HEATING_blab_stand = (HEATING_blab - in_mean)/ in_std
            HEATING_blabF = HEATING_blab_stand.reshape(-1, 25, 32, 1)
            label = np.empty(shape=(1))
            label.fill(initloc)
            
            predi = new_model.predict([HEATING_blabF,label])
            predvalue = predi[:,0]
            HEATING_pred[TH, PH] = (predvalue * dataOut_std) + dataOut_mean
            # title = "tH: " + str(tropic_heating) + " pH: " + str(polar_heating) + " pred: " + str(HEATING_pred[TH, PH])
            # print(title)
            # MakeContourPlot(HEATING_blab, title, y, pf)
            HEATING_uncertainty[TH, PH] = predi[:,1]
 
    Trials_x2 = Trials_x #- (interval_x/2)
    Trials_y2 = Trials_y #- (interval_y/2)
    X, Y = np.meshgrid(Trials_x2, Trials_y2)
    
    plt.figure(figsize=(8,2), dpi=600)
    cs = plt.pcolormesh(X, Y, HEATING_pred, cmap=tmap_dis, vmin = -4, vmax = 4)#, vmin = -maxi, vmax = maxi)#, vmin = -4, vmax = 4)
    # plt.title("Predicted Jet Shift (combined before pred)", **title_font)
    # plt.colorbar(cs, label = "")
    plt.xticks(**axisTick_font)
    plt.yticks(**axisTick_font)
    plt.xlabel("Polar Anom", **axis_font)
    plt.ylabel("Tropical Anom", **axis_font)
    
    if ADD_DDC_DATA:
       f = nc.Dataset("/Users/cconn/Documents/CESM2_2HS/data/Averaged_HeatingFiles/DeltaFromHeating_small.nc",'r') 
       ddc_heating = f["heatingScenerios"][:,:]
       x = ddc_heating[:,1] 
       y = ddc_heating[:,2] 
       
       plt.scatter(x, y, c = ddc_heating[:,0], cmap=tmap_dis, vmin = -5, vmax = 5, edgecolors='black', s = 100, linewidth = .2)#, facecolors='none', edgecolors=ddc_heating[:,0])
    if SAVEFIGURES: 
        plt.savefig(figuresavepath + "TwoCompetingHeating.png")
    plt.show()

    
    plt.figure(figsize=(8,2), dpi=600)
    cs = plt.pcolormesh(X, Y, HEATING_uncertainty, cmap=tmap_uncertainty_dis, vmin = 0)#, vmin = 0.6, vmax = 1)#, vmin = -maxi, vmax = maxi)#, vmin = -4, vmax = 4)
    # plt.title("Scynorio Uncertainty", **title_font)
    # plt.colorbar(cs, label = "")
    plt.xticks(**axisTick_font)
    plt.yticks(**axisTick_font)
    plt.xlabel("Polar Anom", **axis_font)
    plt.ylabel("Tropical Anom", **axis_font)
    plt.show()
    return(HEATING_pred)

Tropic_lat = 0
Tropic_height = 300
Tropic_latSize = 27
Tropic_heightSize = 125
Arctic_lat = 90
Arctic_height = 1000
Arctic_latSize = 16
Arcitc_heightSize = 250
    
if EXPLORE_TWO_HEATING_MAGNITUDE:
    heatingField = TwoHeatingSmall(y, pf, 0)

if TwoHeatingNonlinear:
    print("TwoHeatingNonlinear")
    Trials_x = [-1.2, -1.1, -1, -.9, -.8, -.7, -.6, -.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.2] #np.arange(-1.2, 1.4, interval_x)
    Trials_y = [-.2, -.15, -.10, -.05, 0, .05, .1, .15, .2]#np.arange(-.2, .21, interval_y)

    Tropical_response = np.empty(shape = np.size(Trials_y))
    Polar_response = np.empty(shape = np.size(Trials_x))
    
    dtdt = np.empty(shape = (pf_size, y_size))
    label = np.empty(shape=(1))
    label.fill(0) 
    
    for PH, polar_heating in enumerate(Trials_x):
        dtdt.fill(0)
        dtdt = CreateGaussianBlob(dtdt, Arctic_lat, Arctic_height, Arctic_latSize, Arcitc_heightSize, polar_heating)

        HEATING_blab_stand1 = (dtdt - in_mean)/ in_std
        HEATING_blabF1 = HEATING_blab_stand1.reshape(-1, 25, 32, 1)
    
        predi = new_model.predict([HEATING_blabF1,label])
        Polar_response[PH] = (predi[:,0] * dataOut_std) + dataOut_mean
        
    for TH, tropical_heating in enumerate(Trials_y):
        dtdt.fill(0)
        dtdt = CreateGaussianBlob(dtdt, Tropic_lat, Tropic_height, Tropic_latSize, Tropic_heightSize, tropical_heating)

        HEATING_blab_stand1 = (dtdt - in_mean)/ in_std
        HEATING_blabF1 = HEATING_blab_stand1.reshape(-1, 25, 32, 1)
    
        predi = new_model.predict([HEATING_blabF1,label])
        Tropical_response[TH] = (predi[:,0] * dataOut_std) + dataOut_mean 
        
    HEATING_added = np.empty(shape=(np.size(Trials_y), np.size(Trials_x)))
    for xT, XT in enumerate(Trials_x):
        HEATING_added[:,xT] = Tropical_response
    for yT, YT in enumerate(Trials_y):
        HEATING_added[yT, :] = HEATING_added[yT, :] + Polar_response  
    
    Trials_x2 = Trials_x #- (interval_x/2)
    Trials_y2 = Trials_y #- (interval_y/2)
    X, Y = np.meshgrid(Trials_x2, Trials_y2)
    
    plt.figure(figsize=(8,2), dpi=600)
    cs = plt.pcolormesh(X, Y, HEATING_added, cmap=tmap_dis, vmin = -4, vmax = 4)#, vmin = -maxi, vmax = maxi)#, vmin = -4, vmax = 4)
    # plt.title("Predicted jet Shift (combined after pred)", **title_font)
    # plt.colorbar(cs, label = "")
    plt.xticks(**axisTick_font)
    plt.yticks(**axisTick_font)
    plt.xlabel("Polar Anom", **axis_font)
    plt.ylabel("Tropical Anom", **axis_font)

    SUB = heatingField - HEATING_added
    
    plt.figure(figsize=(8,2), dpi=600)
    cs = plt.pcolormesh(X, Y, SUB, cmap=tmap_dis, vmin = -.7, vmax = .7)#, vmin = -maxi, vmax = maxi)#, vmin = -4, vmax = 4)
    # plt.title("Difference", **title_font)
    # plt.colorbar(cs, label = "")
    plt.xticks(**axisTick_font)
    plt.yticks(**axisTick_font)
    plt.xlabel("Polar Anom", **axis_font)
    plt.ylabel("Tropical Anom", **axis_font)
    
if BUTLER2010:
    print("BUTLER2010")
    anomalies = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
    lat_loc = 0
    lat_shape = [27, 27, 13.5, 27]
    pf_loc = [300, 300, 300, 500]
    pf_shape  = [125, 75, 125, 125]

    label = np.empty(shape=(1))
    label.fill(0)     
    dtdt = np.empty(shape = (pf_size, y_size))
    predictedarray = np.empty(shape=(np.size(anomalies), np.size(lat_shape)))
    predictedarray.fill(0)
    uncertaintyarray = np.empty(shape=(np.size(anomalies), np.size(lat_shape)))
    uncertaintyarray.fill(0)
    for a, anom in enumerate(anomalies):
        dtdt.fill(0)
        dtdt = CreateGaussianBlob(dtdt, lat_loc, pf_loc[0], lat_shape[0], pf_shape[0], anom)
        # p.MakeContourPlot_outline(dtdt, "Tropical pattern a", y, pf, 'cornflowerblue', [0, .5, .75])
        # p.MakeContourPlot(dtdt, "Tropical pattern a", y, pf)
        
        HEATING_blab_stand1 = (dtdt - in_mean)/ in_std
        HEATING_blabF1 = HEATING_blab_stand1.reshape(-1, 25, 32, 1)
    
        predi = new_model.predict([HEATING_blabF1,label])
        predictedarray[a, 0] = (predi[:,0] * dataOut_std) + dataOut_mean
        uncertaintyarray[a, 0] = predi[:,1]
        #************************
        
        dtdt.fill(0)
        dtdt = CreateGaussianBlob(dtdt, lat_loc, pf_loc[1], lat_shape[1], pf_shape[1], anom)
        # p.MakeContourPlot_outline(dtdt, "Tropical pattern b", y, pf, 'crimson', [0, .5, .75])
        # p.MakeContourPlot(dtdt, "Tropical pattern b", y, pf)
        
        HEATING_blab_stand1 = (dtdt - in_mean)/ in_std
        HEATING_blabF1 = HEATING_blab_stand1.reshape(-1, 25, 32, 1)
    
        predi = new_model.predict([HEATING_blabF1,label])
        predictedarray[a, 1] = (predi[:,0] * dataOut_std) + dataOut_mean
        uncertaintyarray[a, 1] = predi[:,1]
        #************************
        
        dtdt.fill(0)
        dtdt = CreateGaussianBlob(dtdt, lat_loc, pf_loc[2], lat_shape[2], pf_shape[2], anom)
        # p.MakeContourPlot_outline(dtdt, "Tropical pattern c", y, pf, 'blueviolet', [0, .5, .75])
        # p.MakeContourPlot(dtdt, "Tropical pattern c", y, pf)
        
        HEATING_blab_stand1 = (dtdt - in_mean)/ in_std
        HEATING_blabF1 = HEATING_blab_stand1.reshape(-1, 25, 32, 1)
    
        predi = new_model.predict([HEATING_blabF1,label])
        predictedarray[a, 2] = (predi[:,0] * dataOut_std) + dataOut_mean
        uncertaintyarray[a, 2] = predi[:,1]
        #************************
        
        dtdt.fill(0)
        dtdt = CreateGaussianBlob(dtdt, lat_loc, pf_loc[3], lat_shape[3], pf_shape[3], anom)
        # p.MakeContourPlot_outline(dtdt, "Tropical pattern d", y, pf, 'seagreen', [0, .5, .75])
        # p.MakeContourPlot(dtdt, "Tropical pattern d", y, pf)
        
        HEATING_blab_stand1 = (dtdt - in_mean)/ in_std
        HEATING_blabF1 = HEATING_blab_stand1.reshape(-1, 25, 32, 1)
    
        predi = new_model.predict([HEATING_blabF1,label])
        predictedarray[a, 3] = (predi[:,0] * dataOut_std) + dataOut_mean
        uncertaintyarray[a, 3] = predi[:,1]
        # sys.exit()
        #************************
    plt.figure(figsize=(8,2), dpi=600)
    plt.axvline(x = 4, color = "black",  alpha=0.3)
    plt.plot(predictedarray[:,0], color = 'cornflowerblue', label = "a)")
    # plt.plot([int(np.size(anomalies) - 1), 0],[predictedarray[-1:,0], predictedarray[0,0]], color = 'turquoise', linestyle='--',  alpha=0.3, label = "a)")
    plt.plot(predictedarray[:,1], color = 'crimson', label = "b)")
    # plt.plot([int(np.size(anomalies) - 1), 0],[predictedarray[-1:,1], predictedarray[0,1]], color = 'orange', linestyle='--',  alpha=0.3, label = "b)")
    plt.plot(predictedarray[:,2], color = 'blueviolet', label = "c)")
    # plt.plot([int(np.size(anomalies) - 1), 0],[predictedarray[-1:,2], predictedarray[0,2]], color = 'orchid', linestyle='--',  alpha=0.3, label = "c)")
    plt.plot(predictedarray[:,3], color = 'seagreen', label = "d)")
    # plt.plot([int(np.size(anomalies) - 1), 0],[predictedarray[-1:,3], predictedarray[0,3]], color = 'green', linestyle='--',  alpha=0.3, label = "d)")    
    plt.legend()
    plt.xticks(ticks = np.arange(0, np.size(anomalies), 1), labels = anomalies, **axisTick_font)
    plt.yticks(**axisTick_font)
    plt.xlim(0,(np.size(anomalies) - 1))
    plt.show()

 