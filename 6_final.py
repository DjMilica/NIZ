import copy
import numpy as np
from scipy import linalg as la

def SVD(A, m, n, eps = 1.e-4):
    # 1. step, U = A
    U = np.array(copy.copy(A))
    # 2. step, V = I(nxn)
    V = np.eye(n) 
    # 3. step, calculate N^2 = M, s = 0, first = true
    M = 0
    s = 0
    first = True
    for i in range(m):
        for j in range(n):
            M += U[i][j] * U[i][j]

    # step 4
    a = 0
    b = 0
    c = 0
    error = 1
    while np.sqrt(s) > eps*eps*M or first == True:
        # 4.a
        s = 0
        first = False
        
        # 4.b
        for i in range(n-1):
            for j in range(i+1, n):
                squaresum = 0
                a = 0
                b = 0
                c = 0
                # determining c,s,d1,d2
                for k in range(m):
                    squaresum += U[k][i]*U[k][j]
                    a += U[k][i]*U[k][i]
                    b += U[k][j]*U[k][j]
                    c += U[k][i]*U[k][j]
                s += squaresum * squaresum

                # Jacobi rotation
                tau = ((b - a)* 1.0) / (2*c)
                t = np.sign(tau) * 1.0/(np.abs(tau) + np.sqrt(1 + tau*tau))
                cs = 1.0/np.sqrt(1 + t*t)
                sn = cs*t
                
                # update columns i and j of U
                for k in range(m):
                    tmp = U[k][i]
                    U[k][i] = cs*tmp - sn*U[k][j]
                    U[k][j] = sn*tmp + cs*U[k][j]
                    
                # update the matrix V
                for k in range(n):
                    tmp = V[k][i]
                    V[k][i] = cs*tmp - sn*V[k][j]
                    V[k][j] = sn*tmp + cs*V[k][j]
                
    # step 5
    b = np.zeros(n)
    for i in range(n):
        b[i] = np.linalg.norm(U[:,i])
        for k in range(m):
            U[k][i] = U[k][i] / b[i]

    E = np.diag(b)
    
    
    print("U is:")
    print(U)
    print("E is")
    print(E)
    print("V.T is:")
    print(V.T)
    
    print("A is:")
    print(np.dot(U, np.dot(E, V.T)))
    
def main():
   A = [ [1.0 ,3.0,2.0],
       [5.0 ,6.0,4.0],
       [7.0 ,8.0,9.0],
        [2.0,1.0,6.0]]
   U, S, V = la.svd(A, full_matrices = False)
   print(U)
   print(S)
   print(V)
   
   SVD(A, 4, 3)