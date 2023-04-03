# CESM dry dynamical core jet sensitivity

Using a CNN to learn the forced response of the CESM dry dynamical core jet stream to a temperature tendency.

## Tensorflow Code

Code was written in matplotlib 3.5.1, netcdf4 1.5.7, numpy 1.21.4, python 3.7.7, scipy 1.7.3, seaborn, 0.11.2, tensorflow 2.8.0, and tensorflow-probability 0.15.0

## Files

**Prep_data_Cruns.py** : Script used to process the zonally averaged temperature and zonally averaged zonal winds from the control run into a smoothed zonally averaged temperature tendency and a jet location and shift. 

**Prep_Data_HE.py** : Script used to process the zonally averaged zonal winds from the forced heating experiemetns into a true jet shift.

**SETTINGS.py** : holds the settings for the convolutional neural network

**CNN_predictLat_uncertainty.py** : Code used to train the CNN.

**CNN_predictLat_uncertainty_data.py** : Code used to train the CNN that works with available data.

**CNN_model.h5** : CNN model

**Datasheet_for_Earth_Datasets.pdf** : datasheet for available data.

**AnalyzeModels/main.py** : main script used to evaluate the trained network. Makes all subplots in (DOI/link). Calls Plots.py and PredictedPlots.py to create figures.

**AnalyzeModels/main_data.py** : main script used to evaluate the trained network. Unlike main.py, this file works easily with available data. Makes all subplots in (DOI/link). Calls Plots.py and PredictedPlots.py to create figures.

**AnalyzeModels/Plots.py** : Holds functions used to plot data

**AnalyzeModels/PredictedPlots.py** : Holds functions that predict and create figures based around the network predicting a forced response

**AnalyzeModels/compareCNN_DDC.py** : Creates Figure 7 from (DOI/link) comparing the predicted jet shift from the neural network to the true jet shift from the CESM dry dynamical core.
