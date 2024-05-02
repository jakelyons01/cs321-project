"""
Linear Global Alignment 

Jake Lyons and Jaime Robinson
"""

import middle_edge
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

def middle_edge(top, bottom, left, right):
    #finds middle edge and middle node given information
    #call middle_edge.middle_edge(seq1, seq2)
    #parse output to make it useful
    return

def linear_space_align(top, bottom, left, right):
    #recursively finds highest-scoring path in alignment graph in linear space
    if left == right:
        return #alignment formed by bottom - top vertical edges
    if top == bottom:
        return #alignment fromed by right - left horizontal edges

    middle = (len(left) + len(right)) //2
    return
