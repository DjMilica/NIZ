import copy
import numpy as np
from scipy import linalg as la

#only for symetric matrices

def SVD(A, m, n, eps = 1.e-4):
    # 1. step, B = A
    B = np.array(copy.copy(A))
    # 2. step, U = I(mxn)
    U = np.eye(m,n) 
    # 3. step, V + I(nxn)
    V = np.eye(n)
    
    # 4. step, calculate N^2 = M, s = 0, first = true
    M = 0
    s = 0
    first = True
    for i in range(m):
        for j in range(n):
            M += B[i][j] * B[i][j]

    # step 5
    a = 0
    b = 0
    c = 0
    error = 1
    while np.sqrt(s) > eps*eps*M or first == True:
        s = 0
        first = False
        
        for i in range(n-1):
            for j in range(i+1, n):
                s = s + B[i][j]*B[i][j] + B[j][i]*B[j][i]
                # determining c,s,d1,d2
                a = B[i][i]
                b = B[j][j]
                c = B[i][j]

                # Jacobi rotation
                tau = ((b - a)* 1.0) / (2*c)
                t = np.sign(tau) * 1.0/(np.abs(tau) + np.sqrt(1 + tau*tau))
                cs = 1.0/np.sqrt(1 + t*t)
                sn = cs*t
                
                # update the 2 by 2 submatrix 
                B[i][i] = a - c*t
                B[j][j] = b + c*t
                B[i][j] = 0
                B[j][i] = 0
                
                # update the rest of rows and columns i and j
                for k in range(n):
                    if k != i and k != j:
                        tmp = B[i][k]
                        B[i][k] = cs*tmp - sn*B[j][k]
                        B[j][k] = sn*tmp + cs*B[j][k]
                        B[k][i] = B[i][k]
                        B[k][j] = B[j][k]
                # update the eigenvector matrix V
                for k in range(n):
                    tmp = V[k][i]
                    V[k][i] = cs*tmp - sn*V[k][j]
                    V[k][j] = sn*tmp + cs*V[k][j]
                    
                # update the eigenvector matrix U
                for k in range(n):
                    tmp = U[k][i]
                    U[k][i] = cs*tmp - sn*U[k][j]
                    U[k][j] = sn*tmp + cs*U[k][j]
                
    # step 5
    s = np.diag(B)
    S = np.eye(n)
    for i in range(n):
        S[i][i] = np.abs(s[i])
        if s[i] < 0:
            U[:,i] = -1*U[:,i]
    
    print("U is:")
    print(U)
    print("S is")
    print(S)
    print("V is:")
    print(V.T)
    
    print("A is:")
    print(np.dot(U, np.dot(S, V.T)))
    
def main():
   A = [ [1.0 ,5.0,7.0],
      [5.0 ,6.0,4.0],
      [7.0 ,4.0,9.0]]
   U, S, V = la.svd(A)
   print(U)
   print(S)
   print(V)
   
   SVD(A,3,3)
   
main()