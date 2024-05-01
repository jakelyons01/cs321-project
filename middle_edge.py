"""
Finds middle edge in an alignment in linear space

Jake Lyons and Jaime Robinson
"""

import sys
import numpy as np
from Bio.Align import substitution_matrices
sub_mat = substitution_matrices.load("BLOSUM62")
from enum import IntEnum

INDEL = -5


def read(filename):
    #read two amino acid strings from file
    with open(filename, "r") as file: 
       seq1 = list(file.readline().strip()) 
       seq2 = list(file.readline().strip()) 

    return seq1, seq2

"""
plan:
    using sliding window:
        compute score of immediate child
        Save backtracking pointers
        (then copy col2 info into col1 and repeat)

        return the (score? and) the start and end nodes for the edge
"""

def middle_edge(seq1, seq2):
    #finds middle edge in alignment graph
    n = len(seq1)
    m = len(seq2)
    score = np.zeros((2, n+1), dtype=np.int_)
    back = np.zeros((2, n), dtype=np.int_)

    #initialize left edge of graph
    for i in range(0, n):
        score[0, i+1] = score[0, i] + INDEL

    top_half = m/2 if m%2 == 0 else m//2 +1

    for i in range(0, top_half):
        #copy row1 to row0 if not first iteration
        if i != 0:
            np.copyto(score[0], score[1])
            #NOTE: don't need to reset values of row1 to 0, we will simply overwrite them

        #now run loop to calculate score in row1 over and over again

