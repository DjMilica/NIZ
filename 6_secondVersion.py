import copy
import numpy as np

def SVD(A, m, n, eps = 1.e-8):
    # 1. step, U = A
    U = np.array(copy.copy(A))
    # 2. step, V = I(nxn)
    V = np.eye(n) 
    # 3. step, calculate N^2 = M, s = 0, first = true
    M = 0
    s = 0
    first = True
    for i in range(n):
        for j in range(m):
            M += U[i][j] * U[i][j]
    # step 4
    a = 0
    b = 0
    c = 0

    while np.sqrt(s) > eps*eps*M or first == True:
        # 4.a
        s = 0
        first = False
        converge = 0
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
                tau = (b - a)* 1.0 / 2*c
                t = np.sign(tau) * 1.0/(np.abs(tau) + np.sqrt(1 + tau*tau))
                cs = 1.0/np.sqrt(1 + t*t)
                sn = c*t
                
                for k in range(n):
                    # update columns i and j of U
                    tmp = U[k][i]
                    U[k][i] = cs*tmp - sn*U[k][j]
                    U[k][j] = sn*tmp + cs*U[k][j]
                    # update the matrix V
                    tmp = V[k][i]
                    V[k][i] = cs*tmp - sn*V[k][j]
                    V[k][j] = sn*tmp + cs*V[k][j]
                
    # step 5
    b = np.zeros(n)
    for i in range(n):
        squaresum = 0
        for k in range(m):
            squaresum += U[k][i]*U[k][i]
        b[i] = np.linalg.norm(U[:,i])
        #b[i] = np.sqrt(squaresum)
        for k in range(m):
            U[k][i] = U[k][i] / b[i]
        #U[:,i] = U[:,i]*1.0/b[i]*1.0

    E = np.diag(b)
    #U = U.dot(np.linalg.inv(E))
    
    
    print("U is:")
    print(U)
    print("E is")
    print(E)
    print("V.T is:")
    print(V.T)
    
    print("A is:")
    print(np.dot(U, np.dot(E, V.T)))

def main():
   A = [ [1 ,3,2],
       [5 ,6,4],
       [7 ,8,9]]
   U, S, V = np.linalg.svd(A, compute_uv=True, full_matrices=False)
   print(U)
   print(S)
   print(V)
   
   SVD(A, 3, 3)
   
main()