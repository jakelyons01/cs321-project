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

"""
plan:
    using sliding window:
        compute score of immediate child
        Save backtracking pointers
        (then copy col2 info into col1 and repeat)

        return the (score? and) the start and end nodes for the edge
        
        NOTE: middle node = node in middle column with most letters consumed
"""

def copy_next_col(matrix):
    #copies col1 into col0 for given nx2 matrix
    trans = np.transpose(matrix)
    np.copyto(trans[0], trans[1])
    return np.transpose(trans)

def middle_edge(seq1, seq2):
    #finds middle edge in alignment graph
    n = len(seq1)
    m = len(seq2)
    score = np.zeros((m+1, 2), dtype=np.int_)
    back = np.zeros((m, 2), dtype=np.int_)

    #initialize left edge of graph
    for i in range(0, m):
        score[i+1, 0] = score[i, 0] + INDEL

    top_half = int(n/2 if n%2 == 0 else n//2 +1)

    for i in range(0, top_half+1):
        #copy row1 to row0 if not first iteration
        if i != 0:
            score = copy_next_col(score)
            back = copy_next_col(back)

        for j in range(1, m+1):
            #do scoring and fill in backtrack

            score[j, 1] = max(
                    score[j, 0  ] + INDEL,
                    score[j-1, 1] + INDEL,
                    score[j-1, 0] + sub_mat[(seq1[i-1], seq2[j-1])]
                    )
            
            if score[j, 1] == score[j, 0] + INDEL:
                back[j-1, 1] = Back.VRT

            elif score[j, 1] == score[j-1, 1] + INDEL:
                back[j-1, 1] = Back.HRZ

            elif score[j, 1] == score[j-1, 0] + sub_mat[(seq1[i-1], seq2[j-1])]:
                back[j-1, 1] = Back.MAT

    return back, score


if __name__ == "__main__":
    seq2, seq1 = read(sys.argv[1])
    back, score = middle_edge(seq1, seq2)
    n = len(seq1)
    top_half = int(n/2 if n%2 == 0 else n//2 +1)
    #print("top half:", top_half)
    #print("score:\n", score)
    #print("back:\n", back)
    
    score_trans = np.transpose(score)
    longest = np.argmax(score_trans[1])
    child = "("+ str(top_half) + ", " + str(longest) +")"
    back_pointer = back[longest-1][1]
    parent=""
    
    if back_pointer == Back.MAT:
        parent = "("+ str(top_half -1) + ", " + str(longest -1) +")"

    elif back_pointer == Back.VRT:
        parent = "("+ str(top_half) + ", " + str(longest -1) +")"

    elif back_pointer == Back.HRZ:
        parent = "("+ str(top_half-1) + ", " + str(longest) +")"

        
    #build output string: (i, j) (k, l) where the first connects to the second
    print(parent, child)