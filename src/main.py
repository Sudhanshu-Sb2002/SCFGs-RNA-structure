DATA_SET_PATH = "..\\datasets"
TMP_SAVE_PATH='..\\tmp_data_files'

from  SCFG.CFG_sparse import CFG
from read_write import get_data_set_paths,read_all_RNA_strand_data
from grammar_trainer import train_grammar,create_grammar
from string_utils import estimate_base_pair_freqs
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def main():
    data_set_paths = get_data_set_paths(DATA_SET_PATH)
    freq_path = os.path.join(TMP_SAVE_PATH, 'freqs.npz')
    tmp_strands=os.path.join(TMP_SAVE_PATH,'strands.pkl')

    seqs,key_to_index = read_all_RNA_strand_data(data_set_paths['strand'], reread=False,reload_path=tmp_strands)
    sub_train_data_set='RFA'
    data_set_index=key_to_index[sub_train_data_set]

    non_terminals = ["S", "L", "F"]
    terminals = ["A", "C", "G", "U"]
    grammar= create_grammar(non_terminals, terminals)
    single_freq, double_freq = estimate_base_pair_freqs(CFG.convert_strings_to_int,grammar.terminal_dict, seqs, freq_path=freq_path,recompute=False)

    # now we train the grammar on RFA dataset
    grammar = train_grammar(grammar, seqs[sub_train_data_set], single_freq[data_set_index], double_freq[data_set_index])


if __name__ == '__main__':
    main()