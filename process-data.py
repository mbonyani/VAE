import numpy as np
import pandas as pd
import time
import sequenceDataset as sd

def process_data_file(path_to_dataset: str, sequence_length=10):
    """Takes in a filepath to desired dataset and the sequence length of the sequences for that dataset,
    saves .npz file with arrays for one hot encoded sequences, array of wavelengths and array of local
    integrated intensities"""
    data = sd.SequenceDataset(path_to_dataset, sequence_length)
    ohe_sequences = data.transform_sequences(data.dataset['Sequence'].apply(lambda x: pd.Series([c for c in x])).
                                             to_numpy())  #One hot encodings in the form ['A', 'C', 'G', 'T']
    Wavelen = np.array(data.dataset['Wavelen'])
    LII = np.array(data.dataset['LII'])


    file_path = f"./data-for-sampling/processed-data-files/clean-data-base-{time.time()}.npz"

    np.savez(file_path, Wavelen=Wavelen, LII=LII,
             ohe=ohe_sequences)

process_data_file(path_to_dataset='training-data.csv')

        