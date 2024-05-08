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

class Back(IntEnum):
    MAT = 0
    VRT = 1
    HRZ = 2

def read(filename):
    #read two amino acid strings from file
    with open(filename, "r") as file: 
       seq1 = list(file.readline().strip()) 
       seq2 = list(file.readline().strip()) 

    return seq1, seq2


def copy_next_col(matrix):
    #copies col1 into col0 for given nx2 matrix
    trans = np.transpose(matrix)
    np.copyto(trans[0], trans[1])
    return np.transpose(trans)

def middle_edge(seq1, seq2):
    #finds middle edge in alignment graph
    n = len(seq1)
    m = len(seq2)
    score = np.zeros((n+1, 2), dtype=np.int_)
    back = np.zeros((n, 2), dtype=np.int_)

    #initialize left edge of graph
    for i in range(0, n):
        score[i+1, 0] = score[i, 0] + INDEL

    #initialize top row
    score[0, 1] = score[0, 0] + INDEL

    #top_half = int(n/2 if n%2 == 0 else n//2 +1)
    top_half = int(m//2 +1)

    #iterate window
    for i in range(1, top_half+1): #iterates first half of n
        #copy row1 to row0
        if i != 1:
            score = copy_next_col(score)
            score[0,1] = score[0,0] + INDEL #account for indel
            back = copy_next_col(back)

        for j in range(1, n+1):
            #do scoring and fill in backtrack
            score[j, 1] = max(
                    score[j-1, 1] + INDEL,
                    score[j, 0] + INDEL,
                    score[j-1, 0] + sub_mat[(seq1[j-1], seq2[i-1])]
                    )

            if score[j, 1] == score[j-1, 1] + INDEL:
                back[j-1, 0] = Back.VRT

            elif score[j, 1] == score[j, 0] + INDEL:
                back[j-1, 0] = Back.HRZ

            elif score[j, 1] == score[j-1, 0] + sub_mat[(seq1[j-1], seq2[i-1])]:
                back[j-1, 0] = Back.MAT

    return back, score


if __name__ == "__main__":
    seq1, seq2 = read(sys.argv[1])
    rev = False
    if len(seq1) >= len(seq2):
        back, score = middle_edge(seq1, seq2)
        n = len(seq2)
    else:
        rev = True
        back, score = middle_edge(seq2, seq1)
        n = len(seq1)
    #top_half = int(n/2 if n%2 == 0 else n//2 +1)
    top_half = int(n//2 +1)
    
    score_trans = np.transpose(score)
    longest = np.argmax(score_trans[1])
    child = (longest, top_half)
    back_ptr = back[longest-1][1]
    parent=()
    
    if back_ptr == Back.MAT:
        parent = (longest -1, top_half -1) 

    elif back_ptr == Back.VRT:
        parent = (longest -1, top_half)

    elif back_ptr == Back.HRZ:
        parent = (longest, top_half-1) 

    print(score)
    if rev:
        print(parent[::-1], child[::-1])
    else:
        print(parent, child)
