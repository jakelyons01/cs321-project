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


# def copy_next_col(matrix):
#     #copies col1 into col0 for given nx2 matrix
#     trans = np.transpose(matrix)
#     np.copyto(trans[0], trans[1])
#     return np.transpose(trans)

def middle_edge(seq1, seq2):
    #finds middle edge in alignment graph
    n = len(seq1)
    m = len(seq2)
    score = np.zeros((n+1, 2), dtype=np.int_)
    back = np.zeros((n+1, 2), dtype=np.int_)

    #top_half = int(n/2 if n%2 == 0 else n//2 +1)
    #top_half = m//2
    #print(top_half)

    #initialize left edge of graph
    col = 0
    back[0,0] = Back.VRT
    for i in range(1, n+1):
        score[i, 0] = score[i-1, 0] + INDEL
        back[i, 0] = Back.VRT
        
    #initialize top row
    #score[0, 1] = score[0, 0] + INDEL

    #iterate window
    for j in range(1, m+1):
        col = j % 2 # Switch columns on each iteration
        #copy row1 to row0
#         if i != 1:
#             score = copy_next_col(score)
#             score[0,1] = score[0,0] + INDEL #account for indel
#             back = copy_next_col(back)

        score[0, col] = score[0, col-1] + INDEL
        back[0, col] = Back.HRZ
        
        for i in range(1, n+1):
            # do scoring and fill in backtrack
            # order matches the indices in the backtrack ENUM so
            # we can set pointers from argmax
            scores = (
                score[i-1, col-1] + sub_mat[(seq1[i-1], seq2[j-1])],
                score[i-1, col] + INDEL,
                score[i, col-1] + INDEL,
            )
            score[i, col] = max(scores)
            back[i, col] = np.argmax(scores)
#             if score[j, 1] == score[j-1, 1] + INDEL:
#                 back[j-1, 0] = Back.VRT
# 
#             elif score[j, 1] == score[j, 0] + INDEL:
#                 back[j-1, 0] = Back.HRZ
# 
#             elif score[j, 1] == score[j-1, 0] + sub_mat[(seq1[j-1], seq2[i-1])]:
#                 back[j-1, 0] = Back.MAT

    return back[:,col], score[:,col]


if __name__ == "__main__":
    seq1, seq2 = read(sys.argv[1])
    
    m = len(seq2)
    top_half = m // 2
    
    _, fs_score = middle_edge(seq1, seq2[:top_half])
    back, ts_score = middle_edge(seq1[::-1], seq2[top_half:][::-1])
    
    # ts_score will be reversed
    longest = np.argmax(fs_score + ts_score[::-1])
    start = (longest, top_half)
    
    back_ptr = back[len(back)-longest-1]  # back will be reversed
    if back_ptr == Back.MAT:
        end = (longest+1, top_half+1) 
    elif back_ptr == Back.VRT:
        end = (longest+1, top_half)
    elif back_ptr == Back.HRZ:
        end = (longest, top_half+1) 
    
    print(start, end)
    
#     top_half = int(n//2 +1)
#     
#     fs_score = np.transpose(fs_score)
#     ts_score = np.transpose(ts_score)
#     longest = np.argmax(fs_score[1] + ts_score[0][::-1])
#     child = (longest, top_half)
#     back_ptr = back[longest-1][1]
#     parent=()
#     
#     if back_ptr == Back.MAT:
#         parent = (longest -1, top_half -1) 
# 
#     elif back_ptr == Back.VRT:
#         parent = ( longest -1, top_half)
# 
#     elif back_ptr == Back.HRZ:
#         parent = (longest, top_half-1) 
# 
#     print(parent, child)
