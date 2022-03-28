def write_unique(path_to_sequences: str):
    sequences_dict = {}

    with open(path_to_sequences, 'r+') as f:
        f.readline()  # Omits the header row for the names of columns
        read_data = f.readlines()
        for i, line in enumerate(read_data):
            split = line.split(',')
            seq = split[0]
            sequences_dict[seq] = seq
        f.truncate(0)

    with open(path_to_sequences, 'r+') as f:
        f.write("Sequence,Wavelen,LII\n")
        keys = sequences_dict.keys()
        for key in keys:
            f.write(key)


def fill_training_data_dict(path_to_data: str):
    data_set_dict = {}

    with open(path_to_data, 'r+') as f:
        f.readline()  # Omits the header row for the names of columns
        read_data = f.readlines()
        for line in read_data:
            split = line.split(',')
            seq = split[0]
            data_set_dict[seq] = seq

    return data_set_dict


def remove_duplicate(data_set_dict: dict, path_to_sequences: str):
    file_contents = None
    with open(path_to_sequences, 'r+') as f:
        file_contents = f.readlines()
        f.truncate(0)
    
    with open(path_to_sequences, 'r+') as f:
        f.write(file_contents[0])
        file_contents = file_contents[1:]

        for line in file_contents:
            split = line.split(',')
            seq = split[0]
            try:
                if data_set_dict.get(line) is None:
                    f.write(seq)
            except:
                print("ERROR ENCOUNTERED")

# import pandas as pd

# df = pd.read_csv('data-for-sampling/past-samples-with-info/samples-1645480525.773994/detailed-sequences')

# with open('training-data.csv', 'w+') as f:
#     f.write("Sequence,Wavelen,LII\n")
#     for i, _ in enumerate(df.loc[:,'Sequence Generated']):
#         f.write(f"{df.loc[i,'Sequence Generated']},{df.loc[i, 'Wavelen Encoded']},{df.loc[i, 'LII Encoded']}\n")