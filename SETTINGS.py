"""
Created on Mon Apr 18 11:53:36 2022

@author: cconn
"""

EXPERIMENTS = {
    "FINAL" : { #STOPPED AT 111, reset to 101
        "experiment" : "FINALS",
          "seed" : 300,
          "CNN_blocks" : [2, 2],
          "CNN_filters" : [32, 64],
          "CNN_kernals" : [3, 3],
          "CNN_strides" : [1, 1],
          "CNN_dropout" : 0,
          "Dense_layers" : [500, 500, 500],
          "activation_conv" : "tanh",
          "activation_dense" : "tanh",
          "ridge" : [0.,.0,.0],
          "dense_dropout" : [0.3,0.3,0.3],
          "earlyStopping" : 10,
          "checkpoint" : 1,
           "TrainingFiles" : [1, 2, 11, 12],
           "ValidationFiles" : [3, 13],
          "SAVESTANDARDIZE" : True,
          "epochs" : 1000,
          "batch_size" : 256,
          "padding" : "same",
          "LR" : 10e-7,
          "Nshift" : 360,
          "run_avg" : 240,
          "sampleweight" : None
    },
    "smalltrial" : { #STOPPED AT 111, reset to 101
        "experiment" : "smalltrial",
          "seed" : 300,
          "CNN_blocks" : [1, 1],
          "CNN_filters" : [8, 16],
          "CNN_kernals" : [3, 3],
          "CNN_strides" : [1, 1],
          "CNN_dropout" : 0,
          "Dense_layers" : [50, 50, 50],
          "activation_conv" : "tanh",
          "activation_dense" : "tanh",
          "ridge" : [0.,.0,.0],
          "dense_dropout" : [0.3,0.3,0.3],
          "earlyStopping" : 10,
          "checkpoint" : 1,
           "TrainingFiles" : [1, 2, 11, 12],
           "ValidationFiles" : [3, 13],
          "SAVESTANDARDIZE" : True,
          "epochs" : 1000,
          "batch_size" : 256,
          "padding" : "same",
          "LR" : 10e-6,
          "Nshift" : 360,
          "run_avg" : 240,
          "sampleweight" : None
    },
}