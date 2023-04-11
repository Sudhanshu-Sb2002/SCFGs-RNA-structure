import numpy as np
from numba import njit, prange
import numba as nb
import os
import sys

sys.setrecursionlimit(40000)
DATA_SET_PATH = "../../datasets"
import warnings

# turn off NumbaPendingDeprecationWarning
warnings.filterwarnings("ignore", category=nb.errors.NumbaPendingDeprecationWarning)


class CFG:
    def __init__(self, start, terminals, nonterminals, individual_probability=True):
        self.start = start
        self.terminals = terminals
        self.nonterminals = nonterminals
        # This SCFG is sparse and not in chomsky normal form. THerefore the amount of rules is not fixed, but lesser
        # rules are of three types ( A-> BC, A-> a, A->B, A->aBc, where capital letters are nonterminals and small letters are terminals)
        # we will just call these type of rules as Transition, Emission, Replace, and ET (Emission and Transition)
        self.terminal_dict = {self.terminals[i]: i for i in range(len(self.terminals))}
        self.nonterminal_dict = {self.nonterminals[i]: i for i in range(len(self.nonterminals))}
        # for each non terminal, specify the type of rules it can have
        self.nt_rules = np.zeros((len(self.nonterminals), 4))  # intially  no rules are applicable
        self.rules = self.init_rules()
        self.individual_probability = individual_probability

    def init_rules(self):
        # create rules for each type
        Tr = np.zeros((len(self.nonterminals), len(self.nonterminals), len(self.nonterminals)))
        Er = np.zeros((len(self.nonterminals), len(self.terminals)))
        Rr = np.zeros((len(self.nonterminals), len(self.nonterminals)))
        ETr = np.zeros((len(self.nonterminals), len(self.terminals), len(self.nonterminals), len(self.nonterminals)))
        # set each element to -np.nan
        Tr.fill(np.nan)
        Er.fill(np.nan)
        Rr.fill(np.nan)
        ETr.fill(np.nan)

        return [Tr, Er, Rr, ETr]

    def activate_rules(self, rule):
        # here rule a is a tuple of (non terminal, ruletype, list of indices)
        nonterminal, ruletype, indices = rule
        self.nt_rules[self.nonterminal_dict[nonterminal], ruletype] = 1
        if indices is None:
            # set all rules of this type to 0
            self.rules[ruletype][nonterminal].fill(0)
        else:
            # fill only those indices with 0
            self.rules[ruletype][nonterminal][indices] = 0

    def assign_random_probablities(self):
        counts = np.zeros((len(self.nonterminals), 4))
        for v in range(self.nonterminals):
            # count the total number of non nan values
            for rule in range(4):
                if self.nt_rules[v, rule] == 1:
                    counts[v, rule] = np.count_nonzero(~np.isnan(self.rules[rule][v]))
            # generate a probability distribution of length counts[v]
            dist = np.random.dirichlet(np.ones(int(counts[v])), size=1)[0]
            # fill the probabilities
            for rule in range(4):
                if self.nt_rules[v, rule] == 1:
                    self.rules[rule][v][~np.isnan(self.rules[rule][v])] = dist[:int(counts[v, rule])]
                    dist = dist[int(counts[v, rule]):]
        return

    @staticmethod
    @njit(cache=True)
    def inside_algorithm(string, rules, L, n_non_terminals, alpha):




