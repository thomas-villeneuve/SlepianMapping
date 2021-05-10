import sympy as sp
from sympy.functions.special.polynomials import assoc_legendre as Pnm
import numpy as np

#Symbols
t = sp.symbols(r'\theta', real=True, nonnegative=True ) #[0,π] 
p = sp.symbols(r'\phi', real=True, nonnegative=True) #[0,2π) 

l = sp.symbols('l', real=True, nonnegative=True)
m = sp.symbols('m', real=True, integer=True)

lp = sp.symbols('l\'', real=True, nonnegative=True)
mp = sp.symbols('m\'', real=True, integer=True)

def fixIndex(l,m):
    return (l+1)**2-l+m-1

#spherical harmonic normalization coefficients
def A(l,m):
    
    if m == 0:
        return sp.sqrt(((2*l+1)*sp.factorial(l-m))/((sp.factorial(l+m))*(4*sp.pi)))
    else:
        return sp.sqrt((2*(2*l+1)*sp.factorial(l-m))/((sp.factorial(l+m))*(4*sp.pi)))
    
#associated legendre polynomials
def P(l,m):
    if m < 1:
        return Pnm(l,m,sp.cos(t))
    else: 
        return (-1)**m*Pnm(l,m,sp.cos(t))
    
#spherical harmonics
def Y(l,m):
    if m >= 0:
        return A(l,m)*P(l,m)*sp.cos(m*p)
    elif m < 0:
        return A(l,m)*P(l,m)*sp.sin(-m*p)
    
#makes an array of  the spherical harmonic fucntions 
def makeYlmVector(L):
    vec = sp.zeros((L+1)**2,1) #SH vectors have (L+1)^2 dimensions

    for l in range(L+1):
        for m in range(-l,l+1):
            vec[fixIndex(l,m)] = Y(l,m)
            
    return vec      

#inner product of angular functions 
def dot_f(f,g):
    h = sp.integrate(f*g*sp.sin(t), (p, 0, 2*sp.pi)) 
    h = sp.integrate(h,(t,0,sp.pi))
    return h 

#norm of an angular function
def norm_f(f):
    return dot_f(f,f)

#inner product in spherical harmonic basis
def dot_v(f,g):
    return sp.transpose(f)*g

#norm of spherical harmonic vector
def norm_v(f):
    return dot_v(f)

#takes a map M and writes it as a speherical harmonic vector (may not be exact)
def toVector(M,L):
    
    Y_vec = makeYlmVector(L)
    
    M_vec = dot_f(M,Y_vec)
            
    return M_vec

#write a spherical harmonic vector as a function
def toFunction(M,L):
    
    Y_vec = makeYlmVector(L)
    
    return sp.transpose(M)*Y_vec

#triangularize a symmetric matrix so that comuputations are less intensive.
#Also set to zero any diagonals that will integrate to zero eventually.
def leftMatrix(l,n):
    L = sp.zeros(n,n)
    L[l,l] = 1
    return L

def rightMatrix(l,n):
    R = sp.eye(n)
    R[0,0] = 0
    for i in range(l):
        R[i+1,i+1] = 0
        
    for j in range(n):
        if (j+l) % 2 == 0:
            R[j,j] = 0
    return R

def rightMatrix2(l,n):
    R = sp.eye(n)
    for i in range(l+1):
        R[i,i] = 0
    return R

def triangleMatrix(D,n):

    T = sp.zeros(n,n)
    
    for l in range(n):
        T = T + leftMatrix(l,n)*D*rightMatrix(l,n)
        
    return T

def un_triangleMatrix(D,n):
    
    U = sp.eye(n)*1/2
    U = U + D
    
    for l in range(n):
        U = U + sp.transpose(leftMatrix(l,n)*D*rightMatrix2(l,n))
        
    return U

#inner product of angular functions 
def D_m(L,m):
    
    n = L-np.abs(m)+1 #size of D_m matrix (L-|m|+1)
    
    vec = sp.zeros(n,1)
    
    for i in range(n): 
        
        vec[i] = A(i+m,m)*P(i+m,m) #adding in phi directly since the integral is trivial
        
    D = vec*sp.transpose(vec) #contruct the matrix of Ylm products
    
    #perform the integral on only the upper triangle, ignoring the diagonal and elements 
    #along even diagonals from the middle. This significantly reduces the number of integrals 
    D = triangleMatrix(D,n) 
    
    if m == 0:
    
        D = 2*sp.pi*sp.integrate(D*sp.sin(t),(t,0,sp.pi/2)) #add in the phi contribution explicitly

    else:
        
        D = sp.pi*sp.integrate(D*sp.sin(t),(t,0,sp.pi/2))
    
    #make the matrix symmetric again and add back in the diagonal
    D = un_triangleMatrix(D,n)
        
    return D

def eigenvector(D,i):
    if D.eigenvects()[i][2][0].normalized()[0] < 0:
        return -D.eigenvects()[i][2][0].normalized()
    else: 
        return D.eigenvects()[i][2][0].normalized()
    
def eigenvalue(D,i):
    return D.eigenvects()[i][0]

#This extends d to S but also sorts it into a vector
#with 'normal' spherical order
def extend(d,L,m):
    S = sp.zeros((L+1)**2,1)
    
    for l in range(L-np.abs(m)+1):
        S[fixIndex(l+np.abs(m),m)] = d[l]
        
    return S

def sort(C):
    for i in range(len(C)):
        for j in range(len(C)-i-1):
            if C[i][0] < C[j+i+1][0]: #list eigenvalues in in decreasing order
                temp = C[i]
                C[i] = C[j+i+1]
                C[j+i+1] = temp
                
            if C[i][0] == C[j+i+1][0]:
                
                if C[i][0] >= 1/2: #for eigenvalue >= 0 list in increasing order of m
                
                    if C[i][1] > C[j+i+1][1]: 
                        temp = C[i]
                        C[i] = C[j+i+1]
                        C[j+i+1] = temp
                        
                if C[i][0] < 1/2: #for eigenvalue < 0 list in decreasing order order of m
                
                    if C[i][1] < C[j+i+1][1]:
                        temp = C[i]
                        C[i] = C[j+i+1]
                        C[j+i+1] = temp
    return C

def slepianToHarmonic(L):
    
    C = []
    
    for m in range(L+1):
        D = D_m(L,m)
        
        for i in range(L-m+1):
            
            if m==0:
                C.append([eigenvalue(D,i),m,extend(eigenvector(D,i),L,m)])
            else:
                C.append([eigenvalue(D,i),-m,extend(eigenvector(D,i),L,-m)])
                C.append([eigenvalue(D,i),m,extend(eigenvector(D,i),L,m)])
      
    sort(C)
    
    K = sp.Matrix([])
    
    for i in range(len(C)):
        K = K.row_insert(i,sp.transpose(C[i][2]))
        
    return sp.transpose(K)