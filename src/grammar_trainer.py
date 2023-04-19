import numpy as np
from SCFG.CFG_sparse import CFG, get_pairings_from_parse_tree
import pickle
from numba import njit, prange
import numba as nb


def train_test_validate_split(strands, train_p, test_p, max_length):
    small_strings = np.array([strand.shape[1] < max_length for strand in strands])
    strands_to_pass = []
    for i in range(len(small_strings)):
        if small_strings[i]:
            strands_to_pass.append(strands[i])
    indices = np.arange(len(strands_to_pass))
    np.random.shuffle(indices)
    train_indices = indices[:int(train_p * len(indices))]
    test_indices = indices[int(train_p * len(indices)):int((train_p + test_p) * len(indices))]
    val_indices = indices[int((train_p + test_p) * len(indices)):]
    train = [strands_to_pass[i] for i in train_indices]
    test = [strands_to_pass[i] for i in test_indices]
    val = [strands_to_pass[i] for i in val_indices]
    return train, val, test


def train_grammar(grammar, train, val, single_freq, double_freq):
    # now we train the grammar
    # first print out the  single and double frequencies

    grammar.assign_random_probablities(single_freq, double_freq)

    tr_rule1, e_rule1, r_rule1, et_rule1 = np.copy(grammar.rules[0]), np.copy(grammar.rules[1]), np.copy(
        grammar.rules[2]), np.copy(grammar.rules[3])
    tr_rule1[0, 1, 0] = 0.869
    tr_rule1[2, 1, 0] = 0.212
    e_rule1[1] = 0.895 * single_freq
    r_rule1[0, 1] = 0.131
    et_rule1[2, :, 2, :] = 0.788 * double_freq
    et_rule1[1, :, 2, :] = 0.105 * double_freq
    grammar.rules[0], grammar.rules[1], grammar.rules[2], grammar.rules[3] = tr_rule1, e_rule1, r_rule1, et_rule1
    grammar.inside_out_driver(train, val, single_freq, double_freq, n_starts=10)
    return grammar


def create_grammar(non_terminals, terminals):
    # first we create a grammar object

    grammar = CFG(non_terminals[0], non_terminals, terminals, InferGeneralRuleOnly=True)
    # The rule types are S->LS|L, F-> dFd|LS, L->s|dFd
    # ["Transition", "Emission", "Replace", "Emmission-Transtion"]
    rules = []
    rules.append(("S", 0, (1, 0)))
    rules.append(("S", 2, [1]))
    rules.append(("F", 0, (1, 0)))
    rules.append(("L", 1, None))
    for i in range(len(terminals)):
        for j in range(len(terminals)):
            rules.append(("F", 3, (i, 2, j)))
            rules.append(("L", 3, (i, 2, j)))
    # now we add the rules to the grammar
    for rule in rules:
        grammar.activate_rules(rule)
    # we estimate single and double counts for the rules

    # now we assign random probabilities to the rules
    return grammar


def test_grammar(grammar: CFG, test_strings, recompute=False, prev_file=None):
    # first we print the rules
    grammar.grammar_print_rules()
    test = grammar.convert_strings_to_int(test_strings, grammar.terminal_dict)
    n_nonterm = len(grammar.nonterminal_dict)
    tr_rule, e_rule, r_rule, et_rule = grammar.rules[0], grammar.rules[1], grammar.rules[2], grammar.rules[3]
    inner_vals, traceback_vals = None, None
    # now we test the grammar using the CYK algorithm to get the most probable parse tree
    if recompute:
        inner_vals = [np.zeros((len(s), len(s), len(grammar.nonterminal_dict)), dtype=float) for s in test]
        traceback_vals = [np.zeros((len(s), len(s), len(grammar.nonterminal_dict), 4), dtype=int) for s in test]
        for s in range(len(test)):
            inner_vals[s], traceback_vals[s] = grammar.CYK_algorithm(test[s], tr_rule, e_rule, r_rule, et_rule,
                                                                     grammar.rule_present, n_nonterm, inner_vals[s],
                                                                     traceback_vals[s])
        # save using pickle in prev_file
        with open(prev_file, "wb") as f:
            pickle.dump(([inner_vals, traceback_vals]), f)
    else:
        with open(prev_file, "rb") as f:
            inner_vals, traceback_vals = pickle.load(f)
    return inner_vals, traceback_vals


def compare_results(test, traceback_vals, inner_vals):
    print()
    string_lengths = [s.shape[1] for s in test]
    pairings = get_pairings_from_parse_tree(traceback_vals, string_lengths)
    comparisons = np.zeros((len(test), 3, 4))
    # first a simple comparison of pairings, we will check the presene and absence of pairs
    for s in range(len(test)):
        comparisons[s] = comparison(test[s][2].astype(int), pairings[s], )
    for measure in range(3):
        TP_percent = np.sum(comparisons[:, measure, 0]) / np.sum(
            comparisons[:, measure, 0] + comparisons[:, measure, 1])
        TN_percent = np.sum(comparisons[:, measure, 3]) / np.sum(
            comparisons[:, measure, 2] + comparisons[:, measure, 3])
        correct_percent = (np.sum(comparisons[:, measure, 0]) + np.sum(comparisons[:, measure, 3])) / np.sum(
            comparisons[:, measure, :])
        print(f"Measure {measure}\t ")
        print(f"TP%: {TP_percent:.2f} TN%: {TN_percent:.2f} Correct%: {correct_percent:.2f}")
        # also calculate PPV and NPV
        PPV = np.sum(comparisons[:, measure, 0]) / np.sum(comparisons[:, measure, 0] + comparisons[:, measure, 2])
        NPV = np.sum(comparisons[:, measure, 3]) / np.sum(comparisons[:, measure, 1] + comparisons[:, measure, 3])
        print(f"PPV: {PPV:.2f} NPV: {NPV:.2f}")
        # also calculate the % of strings >80% correct using metric 2
        correct = (comparisons[:, measure, 0] + comparisons[:, measure, 3]) / np.sum(comparisons[:, measure, :],
                                                                                     axis=-1)
        print(f" % of strings >75% correct: {np.sum(correct > 0.75) / len(correct):.2f}")

    return pairings, comparisons


@njit(nb.i4[:, ::1](nb.i4[::1], nb.i4[::1]), cache=True, parallel=True)
def comparison(pairings, predictions):
    # TP FN FP TN
    TF_PN = np.zeros((3, 4), dtype=nb.i4)
    for i in prange(len(pairings)):
        if pairings[i] > 0:
            if predictions[i] > 0:
                # true positive
                TF_PN[0][0] += 1
                if predictions[i] == pairings[i]:
                    # true positive
                    TF_PN[1][0] += 1
                else:
                    # false negative
                    TF_PN[1][1] += 1
                if predictions[i] - 1 <= pairings[i] <= predictions[i] + 1:
                    # true positive
                    TF_PN[2][0] += 1
                else:
                    # false negative
                    TF_PN[2][1] += 1
            else:
                # false negative
                TF_PN[0][1] += 1
                TF_PN[1][1] += 1
                TF_PN[2][1] += 1

        else:
            if predictions[i] > 0:
                # false positive
                TF_PN[0][2] += 1
                TF_PN[1][2] += 1
                TF_PN[2][2] += 1
            else:
                # true negative
                TF_PN[0][3] += 1
                TF_PN[1][3] += 1
                TF_PN[2][3] += 1
    return TF_PN
