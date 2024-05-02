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

def linear_space_align(top, bottom, left, right):
    #recursively finds highest-scoring path in alignment graph in linear space
    return
