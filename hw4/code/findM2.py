'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import submission as sub
import helper

def computeM2(pts1,pts2,M,K1,K2,F):
    E = sub.essentialMatrix(F,K1,K2)
    M1 = np.eye(3,4)
    C1 = np.matmul(K1,M1)
    M2s = helper.camera2(E)
    
    min_err = -1
    for i in range(4):
        M2_pred = M2s[:,:,i]
        C2_pred = np.matmul(K2,M2_pred)
        #print(C2_pred)
        [w,err] = sub.triangulate(C1,pts1,C2_pred,pts2)
        print(err)
        if i==0:
            min_err = err
            C2 = C2_pred
            M2 = M2_pred
            P = w
        else:
            if err < min_err:
                min_err = err
                C2 = C2_pred
                M2 = M2_pred
                P = w
    return P,M2,C2,M1,C1

D = np.load("../data/some_corresp.npz")
K = np.load("../data/intrinsics.npz")
pts1 = D['pts1']
pts2 = D['pts2']
K1 = K['K1']
K2 = K['K2']
M = 640
F = sub.eightpoint(pts1,pts2,M)
P,M2,C2,M1,C1 = computeM2(pts1,pts2,M,K1,K2,F)
np.savez('q3_3.npz',M2=M2,C2=C2,P=P)