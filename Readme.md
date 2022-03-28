Instructions for executing the code for the regularized VAE model for DNA-regularized nanocluster design as described in:

F. Moomtaheen, M. Killeen, J. Oswald, A. Gonzàlez-Rosell, P. Mastracco, A. Gorovits, S. Copp, and P. Bogdanov, "DNA-Stabilized Silver Nanocluster Design via Regularized Variational Autoencoders", Under review at SIGKDD, 2022

Please, send questions to pbogdanov@albany.edu


## Requirements

The majority of the implementation uses python. For truncated sampling we utilize the mvtnorm package in R.

* Python requirements: documented in file requirements.txt (information on installing Python modules here https://docs.python.org/3/installing/index.html) 

* If you encounter an error when installing rpy2, please refer to the following link for install instructions: https://rpy2.github.io/doc/latest/html/overview.html 

* R requirements: tmvtnorm package, mvtnorm package, Matrix package, stats64 package, gmm package, sandwich package, corpcor package (information on installing R packages can be found here https://www.rdocumentation.org/packages/utils/versions/3.6.2/topics/install.packages) 

* CUDA: Cuda compilation tools, release 10.1, V10.1.243 (Optional)


## Training Instructions

1. Install dependencies detailed in ```requirements.txt```
2. Edit hyperparameters.json hyperparameters to those you wish to train a model with (a sample training data can be found in ```synthetic-data-and-model-info```, it is named ```training-data.csv```)
3. Run single-run.py
OUTPUT: The resulting trained model will be saved in the ```models``` folder and a .npz file containing metrics regarding the model training process will be saved in the ```runs``` folder



## Sampling Instructions

1. Install dependencies detailed in ```requirements.txt``` including those for R listed above
2. Open sampling-parameters.json and edit parameter values to those you wish to sample with
3. Execute sampleSequences.py
OUTPUT: The resulting sampled sequences will be saved along with other information regarding the samples in ```data-for-sampling/past-samples-with-info/samples-time.time()```. The file in this folder will be named ```generated-sequences```.
<br><br>

Parameter JSON Files:
1. hyperparameters.json:
* batch size: batch size for training
* number of epochs:  total number of epochs for training
* alpha: value of alpha hyperparameter
beta: value of beta hyperparameter
* gamma: value of gamma hyperparamter
* delta: value of delta hyperparamter
* dropout: value of dropout hyperparamter
* dumber of latent dimensions: represents |z| in paper, dimensionality of latent space
* number of LSTM layers: Number of layers present in LSTM
* hidden size: h/2 in paper, dimensionality of hidden state in LSTM
* number of linear dimensions: w in paper, linear width of fully connected encoding layer
* training percentage: percent of data used for training, (1 - training percentage) for validation
* weighting: boolean flag for weighting in calculation of regularization loss
* path to training data: path indicating where training data is located

2. sampling-parameters.json:
* Number of samples: number of samples to be extracted via sampling process (duplicate samples removed, so actual number of samples will likely be less)
* Wavelength Proxy Threshold: Indicates samples wavelength proxy values should be higher than n percent of the training data (if this value is 90, samples wavelength proxy values >= 90% of training data’s values)
* LII Proxy Threshold: Indicates samples LII proxy values should be higher than n percent of the training data (if this value is 90, samples LII proxy values >= 90% of training data’s values)
* Original data path: Path to data used to train the model
* Model path: Path to model to be used for sampling
* Path to R: Path to location of R installation on local machine

## Directory Contents

* training-data.csv: Synthetic training data used to train the model
* synthetic-data-and-model-info: Directory containing training-data.csv, model trained
using training-data.csv and optimal model hyperparameters, and .npz containing info for
training-data.csv (needed in sampling-parameters.json for sampling procedure)
* requirements.txt: Python dependencies which need to be installed in order to train models and sample sequences (install all dependencies at once using ```python -m pip install -r requirements.txt```
* hyperparameters.json: JSON file indicating hyperparameters used for model training
* sampling-parameters.json: JSON file indicating parameters used for sampling sequences
* sampleSequences.py: Driver script for sequence sampling process
* truncatedSampling.R: R script for conducting sampling of truncated latent distribution
* filter_sampled_sequences.py: Script used to remove repeated sequences and sequences found in training data from sampled sequences
* plotRun.py: Script used in generating metric plots for training
* single-run.py: Driver script for training process
* sequenceModel.py: Defines structure for model architecture
* sequenceTrainer.py: Defines model training process
* sequenceDataset.py: Processes data for training
* probabilityBin.py: Groups training data into probability bins
* process-data.py: Script used to generate .npz of training data needed for sampling process
* utils/helpers.py: Contains helper functions for training process
* utils/model.py: Superclass for sequenceModel.py
* utils/trainer.py: Superclass for sequenceTrainer.py
* data-for-sampling: Directory containing data needed and data produced through training process
* runs: Folder for storing .npz files containing metrics of previously trained models
* models: Folder for storing .pt model files for previously trained models
* graphs: Folder for storing plots indicating training metrics for previously trained models
* data-for-sampling/past-samples-with-info: Directory containing all sampled sequences and related information
* data-for-sampling/processed-data-files: Directory containing .npz file(s) used in sampling process

## Acknowledgements

This builds upon the implementation provided for : “Attribute-based Regularization of Latent Spaces for Variational Auto-Encoders” by Ashis Pati, Alexander Lerch
 

