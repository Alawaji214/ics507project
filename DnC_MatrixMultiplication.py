#

import numpy as np
import multiprocessing

#Function to add two matrices
def add_matrix(matrix_A, matrix_B, matrix_C, split_index):
    for i in range(split_index):
        for j in range(split_index):
            matrix_C[i][j] = matrix_A[i][j] + matrix_B[i][j]

#Function to multiply two matrices using divide and conquer algorithm
def straightDnC(A, B):
    size = len(A)
 
    c = [[0 for x in range(size)] for y in range(size)]
    if (size == 1):
        c[0][0] = A[0][0] * B[0][0]
 
    else:
        split_index = size // 2
 
        c00 = [[0 for x in range(split_index)] for y in range(split_index)]
        c01 = [[0 for x in range(split_index)] for y in range(split_index)]
        c10 = [[0 for x in range(split_index)] for y in range(split_index)]
        c11 = [[0 for x in range(split_index)] for y in range(split_index)]

        a00, a01, a10, a11 = split(A)
        b00, b01, b10, b11 = split(B)
 
        add_matrix(straightDnC(a00, b00),straightDnC(a01, b10),c00, split_index)
        add_matrix(straightDnC(a00, b01),straightDnC(a01, b11),c01, split_index)
        add_matrix(straightDnC(a10, b00),straightDnC(a11, b10),c10, split_index)
        add_matrix(straightDnC(a10, b01),straightDnC(a11, b11),c11, split_index)
 
        for i in range(split_index):
            for j in range(split_index):
                c[i][j] = c00[i][j]
                c[i][j + split_index] = c01[i][j]
                c[split_index + i][j] = c10[i][j]
                c[i + split_index][j + split_index] = c11[i][j]
 
    return c

# Function to multiply two matrices using divide and conquer algorithm on parallel
def straightDnCParallel(A, B, procnum, return_dict):
    size = len(A)
 
    c = [[0 for x in range(size)] for y in range(size)]
    if (size == 1):
        c[0][0] = A[0][0] * B[0][0]
 
    else:
        split_index = size // 2
 
        c00 = [[0 for x in range(split_index)] for y in range(split_index)]
        c01 = [[0 for x in range(split_index)] for y in range(split_index)]
        c10 = [[0 for x in range(split_index)] for y in range(split_index)]
        c11 = [[0 for x in range(split_index)] for y in range(split_index)]

        a00, a01, a10, a11 = split(A)
        b00, b01, b10, b11 = split(B)
    
        manager = multiprocessing.Manager()
        inner_return_dict = manager.dict()
        jobs = []

        p1 = multiprocessing.Process(target=straightDnCParallel, args=(a00, b00, 1, inner_return_dict))
        p2 = multiprocessing.Process(target=straightDnCParallel, args=(a01, b10, 2, inner_return_dict))
        p3 = multiprocessing.Process(target=straightDnCParallel, args=(a00, b01, 3, inner_return_dict))
        p4 = multiprocessing.Process(target=straightDnCParallel, args=(a01, b11, 4, inner_return_dict))
        p5 = multiprocessing.Process(target=straightDnCParallel, args=(a10, b00, 5, inner_return_dict))
        p6 = multiprocessing.Process(target=straightDnCParallel, args=(a11, b10, 6, inner_return_dict))
        p7 = multiprocessing.Process(target=straightDnCParallel, args=(a10, b01, 7, inner_return_dict))
        p8 = multiprocessing.Process(target=straightDnCParallel, args=(a11, b11, 8, inner_return_dict))
        
        jobs.append(p1)
        jobs.append(p2)
        jobs.append(p3)
        jobs.append(p4)
        jobs.append(p5)
        jobs.append(p6)
        jobs.append(p7)
        jobs.append(p8)

        for proc in jobs:
            proc.start()

        for proc in jobs:
            proc.join()
        # print(return_dict.values())

        add_matrix(inner_return_dict[1], inner_return_dict[2], c00, split_index)
        add_matrix(inner_return_dict[3], inner_return_dict[4], c01, split_index)
        add_matrix(inner_return_dict[5], inner_return_dict[6], c10, split_index)
        add_matrix(inner_return_dict[7], inner_return_dict[8], c11, split_index)
 
        for i in range(split_index):
            for j in range(split_index):
                c[i][j] = c00[i][j]
                c[i][j + split_index] = c01[i][j]
                c[split_index + i][j] = c10[i][j]
                c[i + split_index][j + split_index] = c11[i][j]
 
    return_dict[procnum] = c

#Function to split a matrix into quarters.
def split(matrix):
    """
    Splits a given matrix into quarters.
    Input: nxn matrix
    Output: tuple containing 4 n/2 x n/2 matrices corresponding to a, b, c, d
    """
    row, col = matrix.shape
    row2, col2 = row//2, col//2
    return matrix[:row2, :col2], matrix[:row2, col2:], matrix[row2:, :col2], matrix[row2:, col2:]
 
# function to multiply 2 matrices recursively using strassen algorithm
def strassen(x, y):
    """
    Computes matrix product by divide and conquer approach, recursively.
    Input: nxn matrices x and y
    Output: nxn matrix, product of x and y
    """
 
    # Base case when size of matrices is 1x1
    if len(x) == 1:
        return x * y
 
    # Splitting the matrices into quadrants. This will be done recursively
    # until the base case is reached.
    a, b, c, d = split(x)
    e, f, g, h = split(y)
 
    # Computing the 7 products, recursively (p1, p2...p7)
    p1 = strassen(a, f - h) 
    p2 = strassen(a + b, h)       
    p3 = strassen(c + d, e)       
    p4 = strassen(d, g - e)       
    p5 = strassen(a + d, e + h)       
    p6 = strassen(b - d, g + h) 
    p7 = strassen(a - c, e + f) 
 
    # Computing the values of the 4 quadrants of the final matrix c
    c11 = p5 + p4 - p2 + p6 
    c12 = p1 + p2          
    c21 = p3 + p4           
    c22 = p1 + p5 - p3 - p7 
 
    # Combining the 4 quadrants into a single matrix by stacking horizontally and vertically.
    c = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))
 
    return c


# function to multiply 2 matrices recursively using strassen algorithm on parallel
def strassenParallel(x, y, procnum, return_dict):
    """
    Computes matrix product by divide and conquer approach, recursively.
    Input: nxn matrices x and y
    Output: nxn matrix, product of x and y
    """

    # Base case when size of matrices is 1x1
    if len(x) == 1:
        return_dict[procnum] = x * y
        return
 
    # Splitting the matrices into quadrants. This will be done recursively
    # until the base case is reached.
    a, b, c, d = split(x)
    e, f, g, h = split(y)
 
    manager = multiprocessing.Manager()
    inner_return_dict = manager.dict()
    jobs = []

    # Computing the 7 products, recursively (p1, p2...p7)
    p1 = multiprocessing.Process(target=strassenParallel, args=(a       , f - h , 1, inner_return_dict))
    p2 = multiprocessing.Process(target=strassenParallel, args=(a + b   , h     , 2, inner_return_dict))
    p3 = multiprocessing.Process(target=strassenParallel, args=(c + d   , e     , 3, inner_return_dict))
    p4 = multiprocessing.Process(target=strassenParallel, args=(d       , g - e , 4, inner_return_dict))
    p5 = multiprocessing.Process(target=strassenParallel, args=(a + d   , e + h , 5, inner_return_dict))
    p6 = multiprocessing.Process(target=strassenParallel, args=(b - d   , g + h , 6, inner_return_dict))
    p7 = multiprocessing.Process(target=strassenParallel, args=(a - c   , e + f , 7, inner_return_dict))

    jobs.append(p1)
    jobs.append(p2)
    jobs.append(p3)
    jobs.append(p4)
    jobs.append(p5)
    jobs.append(p6)
    jobs.append(p7)

    for proc in jobs:
        proc.start()

    for proc in jobs:
        proc.join()
 
    # Computing the values of the 4 quadrants of the final matrix c
    c11 = inner_return_dict[5] + inner_return_dict[4] - inner_return_dict[2] + inner_return_dict[6] 
    c12 = inner_return_dict[1] + inner_return_dict[2]          
    c21 = inner_return_dict[3] + inner_return_dict[4]           
    c22 = inner_return_dict[1] + inner_return_dict[5] - inner_return_dict[3] - inner_return_dict[7] 
 
    # Combining the 4 quadrants into a single matrix by stacking horizontally and vertically.
    c = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))
 
    return_dict[procnum] = c


from matrix import read_2_matrix

if __name__ == "__main__":

    A, B = read_2_matrix("input1_4.txt")

    print(straightDnC(A,B))
    print(strassen(A,B))

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    # straightDnCParallel(A,B, 0, return_dict)
    # print(return_dict[0])

    strassenParallel(A,B, 0, return_dict)
    print(return_dict[0])

    # print(strassen(A,B))