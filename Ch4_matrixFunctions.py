# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 07:35:32 2016

@author: ross

Data Science From Scratch
Chapter 4: Linear Algebra
Matrices
"""

A = [[1, 2, 3],
     [4, 5, 6]]
     
B = [[1, 2],
     [3, 4],
     [5, 6],
     [7 ,8]]

def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0 # number of elements in first row
    return num_rows, num_cols

def get_row(A, i):
    return A[i] # A[i] is already the ith row

def get_column(A, j):
    return [A_i[j]          #jth element of row A_i
            for A_i in A]   #for each row A_i

def make_matrix(num_rows, num_cols, entry_fn):
    """returns a num_rows x num_cols matrix 
    whose (i,j)-th entry is entry_fn(i, j)"""
    return [[entry_fn(i, j) for j in range(num_cols)]
            for i in range(num_rows)] 
            # given i, create a list
            # [entry_fun(i,0),...]            
            # create one list for each i
        
def is_diagonal(i, j):
    """1's on the 'diagonal', 0's everywhere else"""
    return 1 if i == j else 0

identity_matrix = make_matrix(5, 5, is_diagonal)            