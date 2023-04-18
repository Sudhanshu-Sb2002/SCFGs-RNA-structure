import os
import pickle
import numpy as np


def estimate_base_pair_freqs(convert_strings_to_int,terminal_dict, string_dict,freq_path, recompute=False):
    if not recompute and os.path.exists(freq_path):
        data = np.load(freq_path)
        single_freq = data['single_freq']
        double_freq = data['double_freq']
        return single_freq, double_freq
    intstrings = []
    for data_set_name, strings in string_dict.items():
        if strings is not None:
            intstrings.append(convert_strings_to_int(strings, terminal_dict, data_set_name))

    loop_base_freq = np.zeros((len(intstrings), len(terminal_dict)))
    stem_base_pair_freq = np.zeros((len(intstrings), len(terminal_dict), len(terminal_dict)))
    set_no = 0

    for key, value in string_dict.items():
        for s in range(len(intstrings[set_no])):
            for i in range(len(intstrings[set_no][s])):
                # check if the base is in a loop or in a stem, by checking if the base is paired (3rd column is not 0)

                if int(string_dict[key][s][2, i]) == 0:
                    loop_base_freq[set_no][intstrings[set_no][s][i]] += 1
                else:
                    c1 = intstrings[set_no][s][i]
                    try:
                        c2 = intstrings[set_no][s][int(string_dict[key][s][2, i]) - 1]
                        stem_base_pair_freq[set_no][c1, c2] += 1
                        stem_base_pair_freq[set_no][c2, c1] += 1
                    except IndexError:
                        print("Index error", key, end=' ')
                        continue

        set_no += 1
    for k in range(len(loop_base_freq)):
        loop_base_freq[k] /= np.sum(loop_base_freq[k])
        for i in range(len(terminal_dict)):
            for j in range(len(terminal_dict)):
                stem_base_pair_freq[k][i, j] = stem_base_pair_freq[k][i, j] + stem_base_pair_freq[k][j, i]
                stem_base_pair_freq[k][j, i] = stem_base_pair_freq[k][i, j]
        stem_base_pair_freq[k] /= np.sum(stem_base_pair_freq[k])

    np.savez(freq_path, single_freq=loop_base_freq, double_freq=stem_base_pair_freq)
    return loop_base_freq, stem_base_pair_freq