import sympy as sp
from sympy.functions.special.polynomials import assoc_legendre as Plm
from sympy.functions.special.tensor_functions import KroneckerDelta as delta
import numpy as np
from starry import Map

#Symbols
theta = sp.symbols(r'\theta', real=True, positive=True)
phi = sp.symbols(r'\phi', real=True, positive=True)

#L = sp.symbols('L', real=True, nonnegative=True)

l = sp.symbols(r'\ell', real=True, nonnegative=True)
m = sp.symbols('m', real=True, interger=True)

p = sp.symbols('p', real=True, nonnegative=True)
q = sp.symbols('q', real=True, interger=True)

#spherical harmonic normalilization
def Alm(l, m):
    return sp.sqrt(((2-delta(m,0))*(2*l+1)*sp.factorial(l-m))
                   /((sp.factorial(l+m))*(4*sp.pi)))

#spherical harmonic
def Ylm( l, m, theta, phi):
    if m < 0:
        return Alm(l,m)*(-1)*Plm(l,m,sp.cos(theta))*sp.sin(m*phi)
    else:
        return Alm(l,m)*(-1)**m*Plm(l,m,sp.cos(theta))*sp.cos(m*phi)

#slepian matrix
def D(l, m, p, q):
    return sp.integrate(sp.integrate(Ylm(l,m,theta,phi)*Ylm(p,q,theta,phi)
                                     *sp.sin(theta), (phi, 0, 2*sp.pi)), 
                                        (theta, 0, sp.pi/2))

#Generate the slepian matrices as a list of fixed order matrices
def s_matrix(L): 
 
    s_matrix = [None]*(L+1)    
            
    for m in range(L+1):
        
        m_matrix = sp.Matrix() #fixed order matrix 
        
        for n in range (L-m+1): #to combine the rows
            
            row = sp.Matrix()
            
            for o in range (L-m+1): #to create the rows 
                
                e = sp.Matrix([D(n+m,m,o+m,m)])
                row = row.col_insert(o,e) #stitch elements into a row
    
            m_matrix = m_matrix.row_insert(n,row) #stitch rows into a matrix
             
        s_matrix[m] = m_matrix #add matrix to the array
        
    #Now s_matrix[m] is the mth order slepian matrix
    return s_matrix
   
#Generate a list of all orders, eigenvalues and eigenvectors   
def s_array(L): 
    
    s_mat = s_matrix(L)
    s_array = [[None,0,None]]
    
    for m in range(L+1): #pick a matrix
    
        for n in range(L-m+1): #pick an eigenspace
            
            eival = s_mat[m].eigenvects()[n][0]
            eivec = s_mat[m].eigenvects()[n][2][0]
            eivec = sp.GramSchmidt([eivec],True)[0]
            
            if L == 1 and eivec[0] < 0:
                eivec = -eivec

            sort = bool(False)
            
            for o in range(len(s_array)): #sort by ei.val
                 if sort == False:
                    if s_array[0][1] == 0:
                        s_array[0][0] = m
                        s_array[0][1] = eival
                        s_array[0][2] = eivec
                        sort = True
                    
                    elif eival == s_array[o][1]: #if same eival sort by order m
                        
                        p,q = o,o
                        
                        if (m==0):
                            
                            while (s_array[p][1] == eival) and (s_array[p][0] < m):
                                p=p+1
                            s_array.insert(p,[m,eival,eivec])
                            
                        else:
                            while (s_array[p][1] == eival) and (s_array[p][0] < -m):
                                p=p+1
                            s_array.insert(p,[-m,eival,eivec])
                            while (s_array[q][1] == eival) and (s_array[q][0] < m):
                                q=q+1
                            s_array.insert(q,[m,eival,eivec])
                            
                        sort = True
                        
                    elif eival > s_array[o][1]:
                        if (m==0):
                            s_array.insert(o,[m,eival,eivec])
                        else:
                            s_array.insert(o,[m,eival,eivec])
                            s_array.insert(o,[-m,eival,eivec])
                        sort = True
                        
            if sort == False:
                if (m==0):
                    s_array.append([m,eival,eivec])
                else:
                    s_array.append([m,eival,eivec])
                    s_array.append([-m,eival,eivec])

    #s_array[n][0] is the order of the nth slepian (n starting from 0)
    #s_array[n][1] is the concentration/eival of the nth slepian
    #s_array[n][2] is the harmonic coefficients/eivec of the nth slepian
    return s_array
    
#Assemble the slepian functions (k starting from 1)
def Slepian(s_array,k):

    S = 0
    k=k-1
    L = int(np.sqrt(len(s_array))-1)
    m = s_array[k][0]
    
    for s in range(L-np.abs(m)+1):
        S = S + s_array[k][2][s]*Ylm(s+np.abs(m),m,theta,phi)

    return sp.simplify(S)

def printAllSlepian(s_array):
    
    for i in range (len(s_array)):
        print(Slepian(s_array,i+1))
        print('\n')
    

#creates a map using the kth slepian (k starting from 1)
def s_map(s_array,k): 

    k=k-1
    L = int(np.sqrt(len(s_array))-1)
    m = s_array[k][0]
    a_m = int(np.abs(m))
    map = Map(lmax=L)
    
    
    for l in range(L-a_m+1):
        map[a_m+l, m] = s_array[k][2][l]
        
    map.show()

#creates maps for all slepians
def s_mapAll(s_array): 

    for i in range (len(s_array)):
        s_map(s_array,i+1)
       




        
            

                
                
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
