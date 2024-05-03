"""
Linear Space Global Alignment 

Jake Lyons and Jaime Robinson
"""

from middle_edge import middle_edge
import sys
import numpy as np

INDEL = -5

class Back(IntEnum):
    MAT = 0
    VRT = 1
    HRZ = 2

"""
IMPLEMENT FROM PSEUDO CODE ON vol1 p. 276
"""

def get_mid_edge(top, bottom, left, right):
    #finds middle edge and middle node given information
    #call middle_edge.middle_edge(seq1, seq2)
    #parse output to make it useful

    back, score = middle_edge(seq1, seq2)
    n = len(seq2)
    top_half = int(n/2 if n%2 == 0 else n//2 +1)

    score_trans = np.transpose(score)
    longest = np.argmax(score_trans[1])
    child = (longest, top_half)
    back_pointer = back[longest-1][1]
    parent=()

    if back_pointer == Back.MAT:
        parent = (longest -1, top_half -1) 

    elif back_pointer == Back.VRT:
        parent = (longest -1, top_half)

    elif back_pointer == Back.HRZ:
        parent = (longest, top_half-1) 

    return parent, back_pointer

def linear_space_align(top, bottom, left, right):
    #recursively finds highest-scoring path in alignment graph in linear space
    if left == right:
        return #alignment formed by bottom - top vertical edges
    if top == bottom:
        return #alignment fromed by right - left horizontal edges

    middle = (len(left) + len(right)) //2
    mid_node, mid_edge = get_mid_edge(a, b, c, d) 

    #RECURSIVE CALL 1: top left box
    path = linear_space_align(top, mid_node, left, middle)
    path.append(mid_edge)
    
    if mid_edge == Back.HRZ or mid_edge == Back.MAT:
        #middle <-- middle + 1

    if mid_edge == Back.VRT or mid_edge == Back.MAT:
        #mid_node <-- mid_node + 1

    #RECURSIVE CALL 2: bottom right box
    bot_right = linear_space_align(mid_node, bottom, middle, right)
    path.append(bot_right)
    
    return path
