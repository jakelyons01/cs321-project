"""
Linear Space Global Alignment 

Jake Lyons and Jaime Robinson
"""

from middle_edge import middle_edge
import sys
import numpy as np
from enum import IntEnum
import cython

blosum = '''
A    R    N    D    C    Q    E    G    H    I    L    K    M    F    P    S    T    W    Y    V    B    Z    X    *
A  4.0 -1.0 -2.0 -2.0  0.0 -1.0 -1.0  0.0 -2.0 -1.0 -1.0 -1.0 -1.0 -2.0 -1.0  1.0  0.0 -3.0 -2.0  0.0 -2.0 -1.0  0.0 -4.0
R -1.0  5.0  0.0 -2.0 -3.0  1.0  0.0 -2.0  0.0 -3.0 -2.0  2.0 -1.0 -3.0 -2.0 -1.0 -1.0 -3.0 -2.0 -3.0 -1.0  0.0 -1.0 -4.0
N -2.0  0.0  6.0  1.0 -3.0  0.0  0.0  0.0  1.0 -3.0 -3.0  0.0 -2.0 -3.0 -2.0  1.0  0.0 -4.0 -2.0 -3.0  3.0  0.0 -1.0 -4.0
D -2.0 -2.0  1.0  6.0 -3.0  0.0  2.0 -1.0 -1.0 -3.0 -4.0 -1.0 -3.0 -3.0 -1.0  0.0 -1.0 -4.0 -3.0 -3.0  4.0  1.0 -1.0 -4.0
C  0.0 -3.0 -3.0 -3.0  9.0 -3.0 -4.0 -3.0 -3.0 -1.0 -1.0 -3.0 -1.0 -2.0 -3.0 -1.0 -1.0 -2.0 -2.0 -1.0 -3.0 -3.0 -2.0 -4.0
Q -1.0  1.0  0.0  0.0 -3.0  5.0  2.0 -2.0  0.0 -3.0 -2.0  1.0  0.0 -3.0 -1.0  0.0 -1.0 -2.0 -1.0 -2.0  0.0  3.0 -1.0 -4.0
E -1.0  0.0  0.0  2.0 -4.0  2.0  5.0 -2.0  0.0 -3.0 -3.0  1.0 -2.0 -3.0 -1.0  0.0 -1.0 -3.0 -2.0 -2.0  1.0  4.0 -1.0 -4.0
G  0.0 -2.0  0.0 -1.0 -3.0 -2.0 -2.0  6.0 -2.0 -4.0 -4.0 -2.0 -3.0 -3.0 -2.0  0.0 -2.0 -2.0 -3.0 -3.0 -1.0 -2.0 -1.0 -4.0
H -2.0  0.0  1.0 -1.0 -3.0  0.0  0.0 -2.0  8.0 -3.0 -3.0 -1.0 -2.0 -1.0 -2.0 -1.0 -2.0 -2.0  2.0 -3.0  0.0  0.0 -1.0 -4.0
I -1.0 -3.0 -3.0 -3.0 -1.0 -3.0 -3.0 -4.0 -3.0  4.0  2.0 -3.0  1.0  0.0 -3.0 -2.0 -1.0 -3.0 -1.0  3.0 -3.0 -3.0 -1.0 -4.0
L -1.0 -2.0 -3.0 -4.0 -1.0 -2.0 -3.0 -4.0 -3.0  2.0  4.0 -2.0  2.0  0.0 -3.0 -2.0 -1.0 -2.0 -1.0  1.0 -4.0 -3.0 -1.0 -4.0
K -1.0  2.0  0.0 -1.0 -3.0  1.0  1.0 -2.0 -1.0 -3.0 -2.0  5.0 -1.0 -3.0 -1.0  0.0 -1.0 -3.0 -2.0 -2.0  0.0  1.0 -1.0 -4.0
M -1.0 -1.0 -2.0 -3.0 -1.0  0.0 -2.0 -3.0 -2.0  1.0  2.0 -1.0  5.0  0.0 -2.0 -1.0 -1.0 -1.0 -1.0  1.0 -3.0 -1.0 -1.0 -4.0
F -2.0 -3.0 -3.0 -3.0 -2.0 -3.0 -3.0 -3.0 -1.0  0.0  0.0 -3.0  0.0  6.0 -4.0 -2.0 -2.0  1.0  3.0 -1.0 -3.0 -3.0 -1.0 -4.0
P -1.0 -2.0 -2.0 -1.0 -3.0 -1.0 -1.0 -2.0 -2.0 -3.0 -3.0 -1.0 -2.0 -4.0  7.0 -1.0 -1.0 -4.0 -3.0 -2.0 -2.0 -1.0 -2.0 -4.0
S  1.0 -1.0  1.0  0.0 -1.0  0.0  0.0  0.0 -1.0 -2.0 -2.0  0.0 -1.0 -2.0 -1.0  4.0  1.0 -3.0 -2.0 -2.0  0.0  0.0  0.0 -4.0
T  0.0 -1.0  0.0 -1.0 -1.0 -1.0 -1.0 -2.0 -2.0 -1.0 -1.0 -1.0 -1.0 -2.0 -1.0  1.0  5.0 -2.0 -2.0  0.0 -1.0 -1.0  0.0 -4.0
W -3.0 -3.0 -4.0 -4.0 -2.0 -2.0 -3.0 -2.0 -2.0 -3.0 -2.0 -3.0 -1.0  1.0 -4.0 -3.0 -2.0 11.0  2.0 -3.0 -4.0 -3.0 -2.0 -4.0
Y -2.0 -2.0 -2.0 -3.0 -2.0 -1.0 -2.0 -3.0  2.0 -1.0 -1.0 -2.0 -1.0  3.0 -3.0 -2.0 -2.0  2.0  7.0 -1.0 -3.0 -2.0 -1.0 -4.0
V  0.0 -3.0 -3.0 -3.0 -1.0 -2.0 -2.0 -3.0 -3.0  3.0  1.0 -2.0  1.0 -1.0 -2.0 -2.0  0.0 -3.0 -1.0  4.0 -3.0 -2.0 -1.0 -4.0
B -2.0 -1.0  3.0  4.0 -3.0  0.0  1.0 -1.0  0.0 -3.0 -4.0  0.0 -3.0 -3.0 -2.0  0.0 -1.0 -4.0 -3.0 -3.0  4.0  1.0 -1.0 -4.0
Z -1.0  0.0  0.0  1.0 -3.0  3.0  4.0 -2.0  0.0 -3.0 -3.0  1.0 -1.0 -3.0 -1.0  0.0 -1.0 -3.0 -2.0 -2.0  1.0  4.0 -1.0 -4.0
X  0.0 -1.0 -1.0 -1.0 -2.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -2.0  0.0  0.0 -2.0 -1.0 -1.0 -1.0 -1.0 -1.0 -4.0
* -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0  1.0 
'''

INDEL = -5

class Back(IntEnum):
    MAT = 0
    VRT = 1
    HRZ = 2

def make_dict(matrix: str):
    #makes dictionary from matrix
    matrix: cython.p_char
    lines: list=[]
    keys: list=[]
    fields: list=[]
    key: str
    matrix_dict: dict={}
    strip_matrix: str

    matrix = matrix.strip()
    strip_matrix = matrix.strip()
    lines = [line.strip() for line in strip_matrix.split('\n')]
    keys = lines.pop(0).split()
    matrix_dict = {}
    for line in lines:
        fields = line.split()
        key = fields.pop(0)
        matrix_dict[key] = {}
        for i in range(len(fields)):
            matrix_dict[key][keys[i]] = float(fields[i])
    return matrix_dict

"""
IMPLEMENT FROM PSEUDO CODE ON vol1 p. 276
"""
def read(filename):
    #read two amino acid strings from file
    with open(filename, "r") as file: 
       seq1 = list(file.readline().strip()) 
       seq2 = list(file.readline().strip()) 

    return seq1, seq2

def get_mid_edge(seq1: list=[], seq2: list=[], sub_mat: dict={}):
    #finds middle edge and middle node given information
    #call middle_edge.middle_edge(seq1, seq2)
    #parse output to make it useful
    m: cython.int
    top_half: cython.int
    back: numpy.ndarray
    fs_score: numpy.ndarray
    ts_score: numpy.ndarray
    longest: cython.int
    start: tuple=()
    back_ptr: cython.int
    end: tuple()

    m = len(seq2)
    top_half = m // 2
    
    _, fs_score = middle_edge(seq1, seq2[:top_half], sub_mat)
    back, ts_score = middle_edge(seq1[::-1], seq2[top_half:][::-1], sub_mat)
    
    # ts_score will be reversed
    longest = np.argmax(fs_score + ts_score[::-1])
    start = (longest, top_half)
    
    back_ptr = back[len(back)-longest-1]  # back will be reversed
    if back_ptr == Back.MAT:
        end = (longest+1, top_half+1) 
        back_ptr = Back.MAT
    elif back_ptr == Back.VRT:
        end = (longest+1, top_half)
        back_ptr = Back.VRT
    elif back_ptr == Back.HRZ:
        end = (longest, top_half+1) 
        back_ptr = Back.HRZ

    return start, back_ptr

def linear_space_align(top: cython.int, bottom: cython.int, left: cython.int, right: cython.int, seq1: list=[], seq2: list=[], sub_mat: dict={}):
    #recursively finds highest-scoring path in alignment graph in linear space

    path: list=[]
    middle: cython.int
    mid_node: tuple=()
    mid_edge: cython.int
    mid_node_index: cython.int
    bot_right: list=[]

    path = [] 
    if left == right:
        return [Back.VRT for _ in range(bottom - top)]
    
    if top == bottom:
        return [Back.HRZ for _ in range(right - left)]

    middle = (left + right) //2
    mid_node, mid_edge = get_mid_edge(seq1[top:bottom], seq2[left:right], sub_mat) 
    mid_node_index = mid_node[0] + top

    #RECURSIVE CALL 1: top left box
    path = linear_space_align(top, mid_node_index, left, middle, seq1, seq2, sub_mat)
    path.append(mid_edge)
    
    if mid_edge == Back.HRZ or mid_edge == Back.MAT:
        middle += 1

    if mid_edge == Back.VRT or mid_edge == Back.MAT:
        #mid_node <-- mid_node + 1
        mid_node_index += 1

    #RECURSIVE CALL 2: bottom right box
    bot_right = linear_space_align(mid_node_index, bottom, middle, right, seq1, seq2, sub_mat)
    path += bot_right

    return path


def get_path(path: list=[], seq1: list=[], seq2: list=[]):
    #get_paths path on seq1, seq2

    output: list=[]
    i: cython.int
    j: cython.int
    step: cython.int

    output = [[],[]]
    i = 0 #tracks position in seq1
    j = 0 #tracks position in seq2

    for step in path:

        if step == Back.VRT:
            #output[0] gets letter from seq1, output[1] gets dash
            output[0] += [seq1[i]]
            output[1] += ["-"]
            i+= 1

        elif step == Back.HRZ:
            #output[0] gets dash, output[1] gets letter from seq2
            output[0] += ["-"]
            output[1] += [seq2[j]]
            j+= 1

        elif step == Back.MAT:
            #both strings get letters
            output[0] += [seq1[i]]
            output[1] += [seq2[j]]
            i+= 1
            j+= 1

    return output

if __name__ == "__main__":
    seq1, seq2 = read(sys.argv[1])
    sub_mat = make_dict(blosum)
    path = linear_space_align(0, len(seq1), 0, len(seq2), seq1, seq2, sub_mat) 
    output = get_path(path, seq1, seq2)
    print(''.join(output[0]))
    print(''.join(output[1]))
