import numpy as numerinopyndarino
import matplotlib.pyplot as ploterino
import multiprocessing as mp
from numba import jit
from IPython import get_ipython

#Naive
def naive(C, I, T): #Inputs are, C: Complex matrix, I: No. Iterations, T: Threshold value
    M = numerinopyndarino.zeros(C.shape) #Declare our output matrix M
    for col in range(C.shape[0]):  #Loop through each complex number c in our C matrix
        for row in range(C.shape[1]):

            c = C[row, col]
            z = 0 + 0j

            for i in range(I-1): #Calculate I iterations of z with formula: z_(i+1) = z_i ^2 + c
                
                z = (z**2) + c

                if (numerinopyndarino.abs(z) > T): #Calculate l(c) as the index where |z_i| > T
                    lc = i
                    break
                else:
                    lc = I

            M[row, col] = lc/I #Store lc to the output matrix M. We divide by I to map from 0-1.
    return M

#Naive Single Row
def naive_row(C, I, T): #Inputs are, C: Complex matrix, I: No. Iterations, T: Threshold value
    M = numerinopyndarino.zeros(C.shape) #Declare our output matrix M
    for col in range(C.shape[0]):  #Loop through each complex number c in our C matrix
        c = C[col]
        z = 0 + 0j

        for i in range(I-1): #Calculate I iterations of z with formula: z_(i+1) = z_i ^2 + c
            
            z = (z**2) + c

            if (numerinopyndarino.abs(z) > T): #Calculate l(c) as the index where |z_i| > T
                lc = i
                break
            else:
                lc = I

        M[col] = lc/I #Store lc to the output matrix M. We divide by I to map from 0-1.
    return M

#Vectorized
def vectorized(C, I, T): #Inputs are, C: Complex matrix, I: No. Iterations, T: Threshold value
    M = numerinopyndarino.zeros(C.shape, dtype=int) #Declare our output matrix M
    
    Z = numerinopyndarino.zeros(C.shape, dtype=complex) #Declare Z as a matrix, where we calculate z = z²+c for each point in C

    for i in range(I-1):
        Z = Z*Z + C

        M[numerinopyndarino.abs(Z) > T] = i

    M[numerinopyndarino.abs(Z) <= T] = I

    return M


#Numbda
@jit(nopython=True)
def naive_with_numba(C, I, T): #Inputs are, C: Complex matrix, I: No. Iterations, T: Threshold value
    M = numerinopyndarino.zeros(C.shape) #Declare our output matrix M
    for col in range(C.shape[0]):  #Loop through each complex number c in our C matrix
        for row in range(C.shape[1]):

            c = C[row, col]
            z = 0 + 0j

            for i in range(I-1): #Calculate I iterations of z with formula: z_(i+1) = z_i ^2 + c
                
                z = (z**2) + c

                if (numerinopyndarino.abs(z) > T): #Calculate l(c) as the index where |z_i| > T
                    lc = i
                    break
                else:
                    lc = I

            M[row, col] = lc/I #Store lc to the output matrix M. We divide by I to map from 0-1.

    return M

#Parallel using Multiprocessing
def naive_with_multiprocessing(C, I, T):
    P = mp.cpu_count() #Detect number of cores
    pool = mp.Pool(processes=P)

    print("No. cores: ", P)
    M = numerinopyndarino.zeros(C.shape) 

    results = []
    for n in range(C.shape[0]): #Loop all rows in C, and run each row in a process
        _C = C[n]
        result = pool.apply_async(naive_row, (_C, I, T,))
        results.append(result)

    pool.close()
    pool.join()

    for i in range(M.shape[0]):
        M[i] = results[i].get()

    return M

#Function for generating the complex matrix C
def generate_C_complex_matrix(pim, pre):
    C = numerinopyndarino.zeros((pim, pre), dtype=complex) #Generate the C matrix with the given dimesion pim x pre
    re = numerinopyndarino.tile(numerinopyndarino.linspace(-2, 1, pim),(pim,1)) #Generate real numbers from -2 to 1, with pim intervals (Stored in a pim x pim matrix)
    im = numerinopyndarino.tile(numerinopyndarino.linspace(1.5, -1.5, pre), (pre,1)).T #Generate imaginary numbers from -1.5 to 1.5 with pre intervals (Stored in a pre x pre matrix)
    C = re + im*1j
    return C

def main():
    #Parameters
    pim = 10
    pre = 10
    I = 100
    T = 2

    #Generate Complex matrix C
    C = generate_C_complex_matrix(pim, pre)

    #Naive Version

    #M = naive(C, I, T)
    #M = vectorized(C, I, T)

    #M = naive_with_numba(C, I, T)
    #M = naive_with_multiprocessing(C, I, T)
    get_ipython().magic('timeit naive(C, I, T)')

    ploterino.imshow(M, cmap='bone', extent=(-2, 1, -1.5, 1.5))
    ploterino.show()


if __name__ == "__main__":
    main()

