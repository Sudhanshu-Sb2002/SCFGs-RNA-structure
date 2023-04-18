DATA_SET_PATH = "..\\datasets"
TMP_SAVE_PATH='..\\tmp_data_files'

from  SCFG.CFG_sparse import CFG
from read_write import get_data_set_paths,read_all_RNA_strand_data,write_prediction_actual
from grammar_trainer import train_grammar,create_grammar,train_test_validate_split,test_grammar,compare_results
from string_utils import estimate_base_pair_freqs
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def main():
    freq_path = os.path.join(TMP_SAVE_PATH, 'freqs.npz')
    tmp_strands = os.path.join(TMP_SAVE_PATH, 'strands.pkl')
    grammar_save_path = os.path.join(TMP_SAVE_PATH, 'grammar.pkl')
    parses_save_path = os.path.join(TMP_SAVE_PATH, 'parses.pkl')


    data_set_paths = get_data_set_paths(DATA_SET_PATH)
    seqs,key_to_index = read_all_RNA_strand_data(data_set_paths['strand'], reread=False,reload_path=tmp_strands)
    data_set='data'
    data_set_index=key_to_index[data_set]

    non_terminals = ["S", "L", "F"]
    terminals = ["A", "C", "G", "U"]
    grammar= create_grammar(non_terminals, terminals)
    single_freq, double_freq = estimate_base_pair_freqs(CFG.convert_strings_to_int,grammar.terminal_dict, seqs, freq_path=freq_path,recompute=False)

    # now we train the grammar on RFA dataset
    retrain=False
    train, val, test = None, None, None
    if retrain:

        train,val,test=train_test_validate_split(seqs[data_set],0.5,0.25,300)
        grammar = train_grammar(grammar, train, val, single_freq[data_set_index], double_freq[data_set_index])
        # dump [train,val,test, grammar] to a file using pickle

        with open(grammar_save_path, 'wb') as f:
            pickle.dump([train,val,test,grammar], f)
    else:

        with open(grammar_save_path, 'rb') as f:
            train,val,test,grammar = pickle.load(f)
    tr_rule1, e_rule1, r_rule1, et_rule1 = np.copy(grammar.rules[0]), np.copy(grammar.rules[1]), np.copy(
        grammar.rules[2]), np.copy(grammar.rules[3])
    tr_rule1[0, 1, 0] = 0.869
    tr_rule1[2, 1, 0] = 0.212
    e_rule1[1] = 0.895 * single_freq[data_set_index]
    r_rule1[0, 1] = 0.131
    et_rule1[2, :, 2, :] = 0.788 * double_freq[data_set_index]
    et_rule1[1, :, 2, :] = 0.105 * double_freq[data_set_index]
    grammar.rules[0], grammar.rules[1], grammar.rules[2], grammar.rules[3] = tr_rule1, e_rule1, r_rule1, et_rule1

    # now it is time to verify this against the actual data
    parse_p, parses=test_grammar(grammar, test,recompute=True,prev_file=parses_save_path)

    # write the best predictions to a file to visualize
    pairings,TF_PN=compare_results(test, parses, parse_p)
    best=best_index(TF_PN)
    write_prediction_actual(TMP_SAVE_PATH,test[best][0],test[best][2].astype(int),pairings[best],terminals)


def best_index(TP_FN):
    positivity = TP_FN[:,2, 0] / (TP_FN[:,2, 0] + TP_FN[:,2, 1])
    return np.argmax(positivity)

if __name__ == '__main__':
    main()