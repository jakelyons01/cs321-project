"""
Linear Space Global Alignment 

Jake Lyons and Jaime Robinson
"""

from middle_edge import middle_edge
import sys
import numpy as np
from enum import IntEnum

INDEL = -5

class Back(IntEnum):
    MAT = 0
    VRT = 1
    HRZ = 2

"""
IMPLEMENT FROM PSEUDO CODE ON vol1 p. 276
"""
def read(filename):
    #read two amino acid strings from file
    with open(filename, "r") as file: 
       seq1 = list(file.readline().strip()) 
       seq2 = list(file.readline().strip()) 

    return seq1, seq2

def get_mid_edge(top, bottom, left, right, seq1, seq2):
    #finds middle edge and middle node given information
    #call middle_edge.middle_edge(seq1, seq2)
    #parse output to make it useful

    if len(seq1) >= len(seq2):
        back, score = middle_edge(seq1[top:bottom+1], seq2[left:right+1])
        n = len(seq2)
    else:
        rev = True
        back, score = middle_edge(seq2[left:right+1], seq1[top:bottom+1])
        n = len(seq1)
    #top_half = int(n/2 if n%2 == 0 else n//2 +1)
    top_half = int(n//2 +1)
    
    score_trans = np.transpose(score)
    longest = np.argmax(score_trans[1])
    child = (longest, top_half)
    back_ptr = back[longest-1][1]
    parent=()

    if back_ptr == Back.MAT:
        parent = [longest -1, top_half -1] 
        back_ptr = Back.MAT

    elif back_ptr == Back.VRT:
        parent = [longest -1, top_half]
        back_ptr = Back.VRT

    elif back_ptr == Back.HRZ:
        parent = [longest, top_half-1] 
        back_ptr = Back.HRZ

    return parent, back_ptr

def linear_space_align(top, bottom, left, right, seq1, seq2):
    #recursively finds highest-scoring path in alignment graph in linear space
    path = []
    if left >= right:
        return [Back.VRT for _ in range(bottom - top)]
    
    if top >= bottom:
        return [Back.HRZ for _ in range(right - left)]

    middle = (left + right) //2
    mid_node, mid_edge = get_mid_edge(top, bottom, left, right, seq1, seq2) 
    #print("mid_node:",mid_node)
    #print("middle:", middle)

    #RECURSIVE CALL 1: top left box
    path = linear_space_align(top, mid_node[1], left, middle, seq1, seq2)
    path.append(mid_edge)
    
    if mid_edge == Back.HRZ or mid_edge == Back.MAT:
        middle += 1

    if mid_edge == Back.VRT or mid_edge == Back.MAT:
        #mid_node <-- mid_node + 1
        mid_node[1] += 1

    #RECURSIVE CALL 2: bottom right box
    bot_right = linear_space_align(mid_node[1], bottom, middle, right, seq1, seq2)
    path += bot_right
    
    return path


def backtrack(path, seq1, seq2):
    #backtracks path on seq1, seq2
    output = [[],[]]
    i = len(seq1) -1 #tracks position in seq1
    j = len(seq2) -1 #tracks position in seq2

    for step in path[::-1]:

        if step == Back.VRT:
            #output[0] gets letter from seq1, output[1] gets dash
            output[0] = [seq1[i]] + output[0]
            output[1] = ["-"] + output[1]
            i-= 1

        elif step == Back.HRZ:
            #output[0] gets dash, output[1] gets letter from seq2
            output[0] = ["-"] + output[0]
            output[1] = [seq2[j]] + output[1]
            j-= 1

        elif step == Back.MAT:
            #both strings get letters
            output[0] = [seq1[i]] + output[0]
            output[1] = [seq2[j]] + output[1]
            i-= 1
            j-= 1

    return output

if __name__ == "__main__":
    seq1, seq2 = read(sys.argv[1])
    path = linear_space_align(0, len(seq1), 0, len(seq2), seq1, seq2) 
    print("path:", path)
    output = backtrack(path, seq1, seq2)
    print(''.join(output[0]))
    print(''.join(output[1]))
