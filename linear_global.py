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

    back, score = middle_edge(seq1[top:bottom], seq2[left:right])
    n = len(seq2)
    top_half = int(n/2 if n%2 == 0 else n//2 +1)
    print("back:\n", back)
    print("score:\n", score)
    print("left:", left)
    print("right:", right)
    print("top:", top)
    print("bottom:", bottom)

    score_trans = np.transpose(score)
    longest = np.argmax(score_trans[1])
    child = [longest, top_half]
    back_pointer = back[longest-1][1]
    parent=[]

    if back_pointer == Back.MAT:
        parent = [longest -1, top_half -1] 
        back_pointer = Back.MAT

    elif back_pointer == Back.VRT:
        parent = [longest -1, top_half]
        back_pointer = Back.VRT

    elif back_pointer == Back.HRZ:
        parent = [longest, top_half-1] 
        back_pointer = Back.HRZ

    return parent, back_pointer

def linear_space_align(top, bottom, left, right, seq1, seq2):
    #recursively finds highest-scoring path in alignment graph in linear space
    path = []
    if left == right:
        return [Back.VRT for _ in range(bottom - top)]
    
    if top == bottom:
        return [Back.HRZ for _ in range(right - left)]

    middle = (left + right) //2
    mid_node, mid_edge = get_mid_edge(top, bottom, left, right, seq1, seq2) 

    #RECURSIVE CALL 1: top left box
    path = linear_space_align(top, mid_node[1], left, middle, seq1, seq2)
    print("path:", path)
    path.append(mid_edge)
    print("path:", path)
    
    if mid_edge == Back.HRZ or mid_edge == Back.MAT:
        middle += 1

    if mid_edge == Back.VRT or mid_edge == Back.MAT:
        #mid_node <-- mid_node + 1
        mid_node[1] += 1

    #RECURSIVE CALL 2: bottom right box
    bot_right = linear_space_align(mid_node[1], bottom, middle, right, seq1, seq2)
    path.append(bot_right)
    
    return path

if __name__ == "__main__":
    seq1, seq2 = read(sys.argv[1])
    path = linear_space_align(0, len(seq1), 0, len(seq2), seq1, seq2) 
    print(path)
