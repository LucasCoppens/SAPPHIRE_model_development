### SAPPHIRE model builder

This repository contains python code that allows the user to develop neural network classifiers for promoter prediction. The code can be modified and tailored to one's needs using the information below. Execute the script 'train_network.py' to create new models.

#### Required input files
A genbank file, a file with coordinates of transcription start sites (TSSs) and a file with coordinates of background sequences for training are required to use the script 'train_network.py'. Input should be formated like the examples found in the data folder. Input file names should be specified at the top of the file 'train_network.py'.

#### Network architecture
The architecture of choice can be specified in line 114. The CNN architecture was found to be the best performing one for promoter prediction in Pseudomonas and Salmonella. One of five provided architectures (code in architectures.py) can be chosen:
- fully connected neural network (build_fully_connected_NN)
- convoluational neural network (CNN) (build_CNN)
- recurrent neural network (RNN) (build_LSTM)
- CNN into RNN (build_CNN_LSTM)
- RNN into CNN into RNN (build_LSTM_CNN_LSTM) 

#### New model location
Every new model will be saved as 'models/model_TSS.h5'. This can be modified in line 130.

#### Class weights 
Class weights for training are programmed to compensate for different amounts of TSS and background data for training. This can be modified in line 97.

