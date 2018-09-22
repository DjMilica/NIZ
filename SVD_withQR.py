import copy
from scipy import linalg as la

def SVD(A, m, n):
    U = np.eye(m);
    S = A.T;
    V = np.eye(n);

    loopmax=100*max(m,n);
    loopcount=0;
    
    error = float("inf")
    
    while error > 1.e-8 and loopcount < loopmax:
        Q, S = np.linalg.qr(S.T, mode = 'complete')
        U = np.dot(U,Q)
        Q, S = np.linalg.qr(S.T, mode = 'complete')
        V = np.dot(V,Q)


        e = np.triu(S,1);
        E = np.linalg.norm(e[:]);
        F = np.linalg.norm(np.diag(S));
        if F == 0:
            F = 1
        error = E / F;
        loopcount = loopcount + 1;

#fix the signs in S
    ss = np.diag(S);
    S = np.zeros((m,n))
    for n in range(len(ss)):
        ssn = ss[n]
        S[n,n] = abs(ssn)
        if ssn < 0:
            U[:,n] = -1* U[:,n]

        
    print("U is:")
    print(U)
    print("S is")
    print(S)
    print("V.T is:")
    print(V.T)
    
    print("A is:")
    print(np.dot(U, np.dot(S, V.T))) 
    
def main():
   A = np.array([ [1.0 ,3.0,2.0],
       [5.0 ,6.0,4.0],
       [7.0 ,8.0,9.0],
             [2.0,1.0,6.0]])
   U, S, V = la.svd(A)
   print(U)
   print(S)
   print(V)
   
   SVD(A, 4, 3)
   
main()