#!/usr/bin/python3

import numpy as np

## STARTING BY DOING THE QR DECOMP ##
def qrgs(A):
	size=np.shape(A)
	m=size[0]
	n=size[1]
	sub=float()
	Q=np.zeros((m,n))
	R=np.zeros((n,n))
	Ak=np.zeros(m)
	i=0
	for k in range(n):
			Ak=A[:,k]
			for i in range(k):
				R[i,k] = dot(Q[:,i],A[:,k])
				Ak=Ak-(dot(A[:,k],Q[:,i])*Q[:,i])
			Q[:,k] = ((Ak)/magnitude(Ak))
			R[k,k]=magnitude(Ak)

	return Q,R

def magnitude(V):
	return np.sqrt(sum(x**2 for x in V))

def dot(A,B):
	dotProd=0
	if len(A) != len(B):
		return
	for i in range(len(A)):
			dotProd += A[i] * B[i]
	return dotProd

#Linear Least Squares#

def linear_least_squares(A,b):
    Q,R= qrgs(A)
    Qt = np.transpose(Q)
    bt = np.transpose(b)
    z= Qt @ bt

    size= R.shape[1]
    x = np.zeros(size)

    for i in range(size-1, -1, -1):
        x[i] = z[i]
        for j in range(i+1, size):
            x[i] -= R[i,j] * x[j]
        x[i] /= R[i,i]
    return x

def MatMul(A,B):
    n_A, m_A = A.shape
    n_B, m_B = B.shape
    if m_A==n_B:
    	c= np.zeros((n_A,m_B))
    	for i in range(n_A):
	    	for j in range(m_B):
		    	for k in range(m_A):
			    	c[i,j] += A[i,k] * B[k,j]
    	return c
    else:
        return None

def trans(T):
	return np.array(list(zip(*T)))


def eigen(A):
    tol=1e-18
    Ak=A.copy()
    Xk=diagMat(A.shape[0])
    E=[]
    for i in range(100000):
        Q, R = qrgs(Ak)
        Ak = MatMul(R,Q)
        Xk = MatMul(Xk,Q)
        for j in range(A.shape[0]):
            Xk[:,j] = Xk[:,j] / magnitude(Xk[:,j])
        if np.max(np.abs(Ak - upTri(Ak))) < tol:
            break
    for k in range(A.shape[0]):
        E.append(Ak[k,k])

    w=np.array(E)
    v=Xk
    return w, v


def diagMat(n):
	d=np.zeros((n,n))
	for i in range(n):
		d[i,i]=1
	return d

def upTri(A):
    k=0
    m, n = A.shape
    r = np.zeros(A.shape)
    for i in range(m):
        for j in range(n):
            if j>= i + k:
                r[i,j] = A[i, j]
    return r
    

def main():
    A= np.array([[2,1,1],[1,3,2],[1,0,0]])
    b=np.array([[1,1,1]])
    print(eigen(A))
if __name__ == "__main__":
	main ()

