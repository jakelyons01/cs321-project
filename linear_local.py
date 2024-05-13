"""
linear-space local alignment

Jake Lyons and Jaime Robinson
"""

import sys
import numpy as np
from enum import IntEnum
import cython
import linear_global

INDEL = -5
pam = '''
     A    R    N    D    C    Q    E    G    H    I    L    K    M    F    P    S    T    W    Y    V    B    Z    X    *
A  2.0 -2.0  0.0  0.0 -2.0  0.0  0.0  1.0 -1.0 -1.0 -2.0 -1.0 -1.0 -3.0  1.0  1.0  1.0 -6.0 -3.0  0.0  0.0  0.0  0.0 -8.0
R -2.0  6.0  0.0 -1.0 -4.0  1.0 -1.0 -3.0  2.0 -2.0 -3.0  3.0  0.0 -4.0  0.0  0.0 -1.0  2.0 -4.0 -2.0 -1.0  0.0 -1.0 -8.0
N  0.0  0.0  2.0  2.0 -4.0  1.0  1.0  0.0  2.0 -2.0 -3.0  1.0 -2.0 -3.0  0.0  1.0  0.0 -4.0 -2.0 -2.0  2.0  1.0  0.0 -8.0
D  0.0 -1.0  2.0  4.0 -5.0  2.0  3.0  1.0  1.0 -2.0 -4.0  0.0 -3.0 -6.0 -1.0  0.0  0.0 -7.0 -4.0 -2.0  3.0  3.0 -1.0 -8.0
C -2.0 -4.0 -4.0 -5.0 12.0 -5.0 -5.0 -3.0 -3.0 -2.0 -6.0 -5.0 -5.0 -4.0 -3.0  0.0 -2.0 -8.0  0.0 -2.0 -4.0 -5.0 -3.0 -8.0
Q  0.0  1.0  1.0  2.0 -5.0  4.0  2.0 -1.0  3.0 -2.0 -2.0  1.0 -1.0 -5.0  0.0 -1.0 -1.0 -5.0 -4.0 -2.0  1.0  3.0 -1.0 -8.0
E  0.0 -1.0  1.0  3.0 -5.0  2.0  4.0  0.0  1.0 -2.0 -3.0  0.0 -2.0 -5.0 -1.0  0.0  0.0 -7.0 -4.0 -2.0  3.0  3.0 -1.0 -8.0
G  1.0 -3.0  0.0  1.0 -3.0 -1.0  0.0  5.0 -2.0 -3.0 -4.0 -2.0 -3.0 -5.0  0.0  1.0  0.0 -7.0 -5.0 -1.0  0.0  0.0 -1.0 -8.0
H -1.0  2.0  2.0  1.0 -3.0  3.0  1.0 -2.0  6.0 -2.0 -2.0  0.0 -2.0 -2.0  0.0 -1.0 -1.0 -3.0  0.0 -2.0  1.0  2.0 -1.0 -8.0
I -1.0 -2.0 -2.0 -2.0 -2.0 -2.0 -2.0 -3.0 -2.0  5.0  2.0 -2.0  2.0  1.0 -2.0 -1.0  0.0 -5.0 -1.0  4.0 -2.0 -2.0 -1.0 -8.0
L -2.0 -3.0 -3.0 -4.0 -6.0 -2.0 -3.0 -4.0 -2.0  2.0  6.0 -3.0  4.0  2.0 -3.0 -3.0 -2.0 -2.0 -1.0  2.0 -3.0 -3.0 -1.0 -8.0
K -1.0  3.0  1.0  0.0 -5.0  1.0  0.0 -2.0  0.0 -2.0 -3.0  5.0  0.0 -5.0 -1.0  0.0  0.0 -3.0 -4.0 -2.0  1.0  0.0 -1.0 -8.0
M -1.0  0.0 -2.0 -3.0 -5.0 -1.0 -2.0 -3.0 -2.0  2.0  4.0  0.0  6.0  0.0 -2.0 -2.0 -1.0 -4.0 -2.0  2.0 -2.0 -2.0 -1.0 -8.0
F -3.0 -4.0 -3.0 -6.0 -4.0 -5.0 -5.0 -5.0 -2.0  1.0  2.0 -5.0  0.0  9.0 -5.0 -3.0 -3.0  0.0  7.0 -1.0 -4.0 -5.0 -2.0 -8.0
P  1.0  0.0  0.0 -1.0 -3.0  0.0 -1.0  0.0  0.0 -2.0 -3.0 -1.0 -2.0 -5.0  6.0  1.0  0.0 -6.0 -5.0 -1.0 -1.0  0.0 -1.0 -8.0
S  1.0  0.0  1.0  0.0  0.0 -1.0  0.0  1.0 -1.0 -1.0 -3.0  0.0 -2.0 -3.0  1.0  2.0  1.0 -2.0 -3.0 -1.0  0.0  0.0  0.0 -8.0
T  1.0 -1.0  0.0  0.0 -2.0 -1.0  0.0  0.0 -1.0  0.0 -2.0  0.0 -1.0 -3.0  0.0  1.0  3.0 -5.0 -3.0  0.0  0.0 -1.0  0.0 -8.0
W -6.0  2.0 -4.0 -7.0 -8.0 -5.0 -7.0 -7.0 -3.0 -5.0 -2.0 -3.0 -4.0  0.0 -6.0 -2.0 -5.0 17.0  0.0 -6.0 -5.0 -6.0 -4.0 -8.0
Y -3.0 -4.0 -2.0 -4.0  0.0 -4.0 -4.0 -5.0  0.0 -1.0 -1.0 -4.0 -2.0  7.0 -5.0 -3.0 -3.0  0.0 10.0 -2.0 -3.0 -4.0 -2.0 -8.0
V  0.0 -2.0 -2.0 -2.0 -2.0 -2.0 -2.0 -1.0 -2.0  4.0  2.0 -2.0  2.0 -1.0 -1.0 -1.0  0.0 -6.0 -2.0  4.0 -2.0 -2.0 -1.0 -8.0
B  0.0 -1.0  2.0  3.0 -4.0  1.0  3.0  0.0  1.0 -2.0 -3.0  1.0 -2.0 -4.0 -1.0  0.0  0.0 -5.0 -3.0 -2.0  3.0  2.0 -1.0 -8.0
Z  0.0  0.0  1.0  3.0 -5.0  3.0  3.0  0.0  2.0 -2.0 -3.0  0.0 -2.0 -5.0  0.0  0.0 -1.0 -6.0 -4.0 -2.0  2.0  3.0 -1.0 -8.0
X  0.0 -1.0  0.0 -1.0 -3.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -2.0 -1.0  0.0  0.0 -4.0 -2.0 -1.0 -1.0 -1.0 -1.0 -8.0
* -8.0 -8.0 -8.0 -8.0 -8.0 -8.0 -8.0 -8.0 -8.0 -8.0 -8.0 -8.0 -8.0 -8.0 -8.0 -8.0 -8.0 -8.0 -8.0 -8.0 -8.0 -8.0 -8.0  1.0
'''

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

class Back(IntEnum):
    MAT = 0
    VRT = 1
    HRZ = 2

def read_fasta(filename):
    #reads fasta file into array
    with open(filename, "r") as file:
        file.readline() #discards fasta info
        seq=[]
        for line in file: 
            seq += list(file.readline().strip())
    return seq

def get_taxi_edges(seq1, seq2, sub_mat):
    #finds start and end of longest path in linear space
    
    n = len(seq1)
    m = len(seq2)
    score = np.zeros((n+1, 2), dtype=np.int_)
    starts = np.zeros((n+1, 2, 2), dtype=np.int_)
    max_score = 0
    max_start = [0,0]
    max_end = [0,0]

    #initialize left edge of graph
    col = 0
    #back[0,0] = Back.VRT
    for i in range(1, n+1):
        score[i, 0] = score[i-1, 0] + INDEL
        starts[i, 0] = starts[0, 0]
        
    #iterate window
    for j in range(1, m+1):
        col = j % 2 # Switch columns on each iteration

        score[0, col] = score[0, col-1] + INDEL
        starts[0, col] = starts[0, col-1]

        if score[0, col] > max_score:
            max_score = score[0, col]
            max_start = starts[0, col]
            max_end = [i, j]

        for i in range(1, n+1):
            # do scoring and fill in backtrack
            # order matches the indices in the backtrack ENUM so
            # we can set pointers from argmax
            scores = (
                0,  
                score[i-1, col-1] + sub_mat[seq1[i-1]][seq2[j-1]],
                score[i-1, col] + INDEL,
                score[i, col-1] + INDEL,
            )
            score[i, col] = max(scores)

            if score[i, col] == 0:
                starts[i, col] = [i, j]
            elif score[i, col] == score[i-1, col-1] + sub_mat[seq1[i-1]][seq2[j-1]]:
                starts[i, col] = starts[i-1, col-1]
            elif score[i, col] == score[i-1, col] + INDEL:
                starts[i, col] = starts[i-1, col]
            elif score[i, col] == score[i, col-1] + INDEL:
                starts[i, col] = starts[i, col-1]

            if score[i, col] > max_score:
                max_score = score[i, col]
                max_start = starts[i, col]
                max_end = [i, j]

    return max_start, max_end, max_score

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

def get_path(path: list=[], seq1: list=[], seq2: list=[], start: list=[]):
    #get_paths path on seq1, seq2

    output: list=[]
    i: cython.int
    j: cython.int
    step: cython.int

    output = [[],[]]
    i = start[0] #tracks position in seq1
    j = start[1] #tracks position in seq2

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
    seq1 = read_fasta(sys.argv[1])
    seq2 = read_fasta(sys.argv[2])
    sub_mat = make_dict(pam)
    start, end, score = get_taxi_edges(seq1, seq2, sub_mat)
    path = linear_global.linear_space_align(start[0], end[0], start[1], end[1], seq1, seq2, sub_mat)
    output = get_path(path, seq1, seq2, start)
    print(score)
    print(''.join(output[0]))
    print(''.join(output[1]))
