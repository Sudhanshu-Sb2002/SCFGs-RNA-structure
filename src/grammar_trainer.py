import numpy as np
from SCFG.CFG_sparse import CFG

def train_grammar(grammar, strands, single_freq, double_freq,max_length=200):
    # now we train the grammar
    # first print out the  single and double frequencies

    print("Single Frequencies in Loop Region")
    for i in range(len(single_freq)):
        print(f"{grammar.terminals[i]} : {single_freq[i]:.2f}")
    print("Double Frequencies in Stem Region")
    print( f"Base:\t{grammar.terminals[0]:<10}{grammar.terminals[1]:<10}{grammar.terminals[2]:<10}{grammar.terminals[3]:<10}")
    print()
    for i in range(len(double_freq)):
        print(f"{grammar.terminals[i]:<5}", end="\t")
        print(f"{double_freq[i, 0]:<10.2f}{double_freq[i, 1]:<10.2f}{double_freq[i, 2]:<10.2f}{double_freq[i, 3]:<10.2f}")

    grammar.assign_random_probablities(single_freq, double_freq)

    strings_smaller_than_200=np.array([strand.shape[1]<max_length for strand in strands])
    strands_to_pass=[]
    for i in range(len(strings_smaller_than_200)):
        if strings_smaller_than_200[i]:
            strands_to_pass.append(strands[i])
    tr_rule1, e_rule1, r_rule1, et_rule1 = np.copy(grammar.rules[0]), np.copy(grammar.rules[1]), np.copy(grammar.rules[2]), np.copy(grammar.rules[3])
    tr_rule1[0, 1, 0] = 0.869
    tr_rule1[2, 1, 0] = 0.212
    e_rule1[1] = 0.895 * single_freq
    r_rule1[0, 1] = 0.131
    et_rule1[2, :, 2, :] = 0.788 * double_freq
    et_rule1[1, :, 2, :] = 0.105 * double_freq
    grammar.rules[0], grammar.rules[1], grammar.rules[2], grammar.rules[3] = tr_rule1, e_rule1, r_rule1, et_rule1
    grammar.inside_out_driver(strands_to_pass, single_freq, double_freq,n_starts=20)
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