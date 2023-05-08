
import numpy as np
import math
import os

def next_path(path_pattern):
    """
    Finds the next free path in an sequentially named list of files

    e.g. path_pattern = 'file-%s.txt':

    file-1.txt
    file-2.txt
    file-3.txt

    Runs in log(n) time where n is the number of existing files in sequence
    """
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2 # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return path_pattern % b


'''
input files
'''

'''  n upto 2**14'''
def create_matrix(n):
    assert n <= 2**14
    assert math.ceil(math.log(n,2)) == math.floor(math.log(n,2)) # n is power of 2

    size = n
    return np.random.randint(0, 10, size=(size, size))

''' long integer '''
def read_2_matrix(filename):
    filecontent    = np.fromfile(filename, dtype=int, sep=" ")

    size    = filecontent[0]
    print('size of matrix is %d * %d' %(size,size))

    a = filecontent[1:size*size+1].reshape((size,size))
    b = filecontent[size*size+1:].reshape((size,size))
    
    return (a,b)

def write_2_matrix(filename, size, a, b):
    f = open(filename, "a")
    f.write(str(size) + "\n")
    f.write(" ".join(map(str, a.flatten())) + "\n")
    f.write(" ".join(map(str, b.flatten())) + "\n")
    # increment to the latest file

def write_result(filename, size, method, time):
    
    
    pass

if __name__ == "__main__":
    size = 4
    mat = create_matrix(size)
    print(mat)
    mat2 = create_matrix(size)
    print(mat2)

    write_2_matrix(next_path("output-%s.txt"), size, mat, mat2)

    # a,b = read_2_matrix("input1_4.txt")
    # print(a)
    # print(b)



'''
Final result files

input1_128_output_StraightDivAndConq.txt
input1_128_output_StraightDivAndConqP.txt
input1_128_output_StrassenDivAndConq.txt
input1_128_output_StrassenDivAndConqP.txt

hh:mm:ss
input1_128_info_StraightDivAndConq.txt
input1_128_info_StraightDivAndConqP.txt
input1_128_info_StrassenDivAndConq.txt
input1_128_info_StrassenDivAndConqP.txt
'''










