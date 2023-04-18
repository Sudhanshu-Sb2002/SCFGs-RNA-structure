# a list of functions for reading and writing files
import os
import pickle
import pandas as pd
import numpy as np

def get_data_set_paths(DATA_SET_PATH):
    x = {}

    x['humanTNRA'] = os.path.join(DATA_SET_PATH, "hg19-tRNAs", "hg19-mature-tRNAs.fa")

    x['strand'] = os.path.join(DATA_SET_PATH, "RNA_STRAND_data", "RNA_STRAND_data", "all_ct_files")

    return x

def read_all_RNA_strand_data(data_set_path, reread=False,reload_path=None):
    # first we go through all files in the directory
    seqs = {}


    if reread:
        files = os.listdir(data_set_path)
        # we will store all the data in a list

        for file in files:
            try:
                prefix = file.split("_")[0]
                if prefix not in seqs:
                    seqs[prefix] = []
                # each file is a .ct file conncect format for RNA secondary structure
                with open(os.path.join(data_set_path, file)) as f:
                    #  first few lines are comments starting with #
                    #  then there is a line whose first element is the number of nucleotides-n
                    #  then there is a table with n rows 6 columns( seperated by spaces) which we read with pandas
                    lines = f.readlines()
                    # find the line with the number of nucleotides
                    n = 0
                    i = 0
                    for i in range(len(lines)):
                        if lines[i][0] != "#":
                            n = int(lines[i].split()[0])
                            break
                    # read the table
                    df = pd.read_csv(os.path.join(data_set_path, file), delim_whitespace=True, skiprows=i + 1, nrows=n,
                                     header=None)
                    # read the sequence
                    temp = np.array(df[1].values, dtype=str)
                    indices = np.array(df[0].values, dtype=int)
                    pairings = np.array(df[4].values, dtype=int)
                    seqs[prefix].append(np.array([temp, indices, pairings]))
                    print(file, end="\t")
            except:
                print("error in file", file)
        pickle.dump(seqs, open(reload_path, "wb"))
    else:
        if reload_path is None:
            reload_path=os.path.join(data_set_path, "data.pkl")
        seqs = pickle.load(open(reload_path, "rb"))
    return refine_strands(seqs)


def refine_strands(seqs):
    # if the key 'SPR' is there, we remove it
    if 'SPR' in seqs:
        del seqs['SPR']
    keys_to_delete = []
    for key, value in seqs.items():
        if value is None or len(value) == 0:
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del seqs[key]

    key_to_index = {}
    for i, key in enumerate(seqs.keys()):
        key_to_index[key] = i
    return seqs, key_to_index