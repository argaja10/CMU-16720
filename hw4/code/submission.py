"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper
import pdb
'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    #Scaling matrix for coordinates
    T = np.array([[1./M,0.,0.],[0.,1./M,0.],[0.,0.,1.]])
    N = pts1.shape[0]
    #pts1 = np.concatenate((pts1,np.ones((N,1))),axis=1).T
    #pts2 = np.concatenate((pts2,np.ones((N,1))),axis=1).T
    #Get scaled coordinates of the correspondences (Nx3 matrices)
    #pts1_sc = np.matmul(T,pts1).T
    #pts2_sc = np.matmul(T,pts2).T
    pts1_sc = pts1/M
    pts2_sc = pts2/M
    xl,yl = pts1_sc[:,0], pts1_sc[:,1]
    xr,yr = pts2_sc[:,0], pts2_sc[:,1]
    U = np.zeros((N,9)).astype('float64')
    U[:,0] = xl*xr
    U[:,1] = yl*xr
    U[:,2] = xr
    U[:,3] = yr*xl
    U[:,4] = yr*yl
    U[:,5] = yr
    U[:,6] = xl
    U[:,7] = yl
    U[:,8] = 1
    _,_,vh = np.linalg.svd(U)
    f = vh[-1,:]
    F_scaled = np.reshape(f,(3,3))
    F_scaled = helper.refineF(F_scaled,pts1_sc[:,:],pts2_sc[:,:]) 
    F = np.matmul(np.matmul(T,F_scaled),T)
    print(F)
    np.savez('q2_1.npz',F=F,M=M)
    return F
    
'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    T = np.array([[1./M,0.,0.],[0.,1./M,0.],[0.,0.,1.]])
    N = pts1.shape[0]
    Nmin = int(0.95*N)
    thresh = 0.8
    pts1 = np.concatenate((pts1,np.ones((N,1))),axis=1).T
    pts2 = np.concatenate((pts2,np.ones((N,1))),axis=1).T
    pts1_sc = np.matmul(T,pts1).T
    pts2_sc = np.matmul(T,pts2).T
    num_iter = 100
    for i in range(num_iter):
        seven_pts_ids = np.random.choice(N,7)
        p1 = pts1_sc[seven_pts_ids,:]
        p2 = pts2_sc[seven_pts_ids,:]
        #Get scaled coordinates of the correspondences (Nx3 matrices)
        xl,yl = p1[:,0], p1[:,1]
        xr,yr = p2[:,0], p2[:,1]
        U = np.zeros((7,9))
        U[:,0] = xl*xr
        U[:,1] = yl*xr
        U[:,2] = xr
        U[:,3] = xl*yr
        U[:,4] = yl*yr
        U[:,5] = yr
        U[:,6] = xl
        U[:,7] = yl
        U[:,8] = 1
        _,_,vh = np.linalg.svd(U)
        f1 = vh[-1,:]
        f2 = vh[-2,:]
        F1 = f1.reshape((3,3))
        F2 = f2.reshape((3,3))
        fun = lambda a:np.linalg.det(a*F1 + (1-a)*F2)
        a0 = fun(0)
        a1 = (2*(fun(1)-fun(-1)))/(3) - (fun(2)-fun(-2))/(12)
        a2 = 0.5*fun(1) + 0.5*fun(-1) - fun(0)
        a3 = (fun(2)-fun(-2))/(12) - (fun(1)-fun(-1))/(6)
        roots = np.roots([a3,a2,a1,a0])
        a_root = roots[np.isreal(roots)]
        a_root = np.real(a_root)
        a_root = list(a_root)
        Farray = []
        good_enough = False
        i=1
        for root in a_root:
            F_scaled = (root*F1 + (1-root)*F2)
            F_scaled = helper.refineF(F_scaled,p1[:,:-1],p2[:,:-1])
            F = np.matmul(np.matmul(T,F_scaled),T)
            Farray.append(F)
            Fp = np.matmul(F,pts1)
            Fp = Fp/Fp[2,:]
            Fp = Fp[:-1,:]
            mag = np.sum(np.abs(Fp.T)**2,axis=-1)**(1./2)
            D = np.diag(np.matmul(pts2.T,np.matmul(F,pts1)))
            D = np.abs(D)/mag
            n = np.count_nonzero(D<thresh)    
            print(i,n)
            i += 1
            if(n>Nmin):
                good_enough = True
        if(good_enough):
            break
    np.savez('q2_2.npz',Farray=Farray,M=M,pts1=pts1[:-1,seven_pts_ids],pts2=pts2[:-1,seven_pts_ids])
    return Farray

'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    E = np.matmul(F,K2)
    E = np.matmul(K1.T,E)
    print(E)
    return E

'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    A = np.zeros((4,4))
    N = pts1.shape[0]
    P = np.empty((N,3))
    err = 0
    for i in range(N):
        pt1=  pts1[i,:].astype('float64')
        pt2 = pts2[i,:].astype('float64')
        A[0,:] = C1[0,:]
        A[1,:] = C1[1,:]
        A[2,:] = C2[0,:]
        A[3,:] = C2[1,:]
        A[0,-1] = A[0,-1]-pt1[0]
        A[1,-1] = A[1,-1]-pt1[1]
        A[2,-1] = A[2,-1]-pt2[0]
        A[3,-1] = A[3,-1]-pt2[1]
        _,eps,vh = np.linalg.svd(A)
        w_tilde = vh[-1,:]
        w_tilde = w_tilde/w_tilde[-1]
        P[i,:] = w_tilde[:-1]
        pt1_est = np.matmul(C1,w_tilde)
        pt1_est = pt1_est[:-1]/pt1_est[-1]
        pt2_est = np.matmul(C2,w_tilde)
        pt2_est=  pt2_est[:-1]/pt2_est[-1]
        err += np.linalg.norm(pt1-pt1_est)**2+np.linalg.norm(pt2-pt2_est)**2
    return P, err

'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    dist_thresh = 20
    sy, sx, _ = im2.shape
    v = np.array([x1, y1, 1])
    l = F.dot(v)
    s = np.sqrt(l[0]**2+l[1]**2)
    if s == 0:
        print('ERROR: Zero line vector in displayEpipolar')
    l = l/s
    if l[0] != 0:
        ye = y1+20
        ys = y1-20
        xe = -(l[1] * ye + l[2])/l[0]
        xs = -(l[1] * ys + l[2])/l[0]
    else:
        xe = x1+20
        xs = x1-20
        ye = -(l[0] * xe + l[2])/l[1]
        ys = -(l[0] * xs + l[2])/l[1]
    err = None
    for t in np.arange(0,1,0.01):
        x2_est = xe*t + (1-t)*xs
        y2_est = ye*t + (1-t)*ys
        if np.sqrt((x1-x2_est)**2 + (y1-y2_est)**2)<dist_thresh:
            x2_est = int(np.round(x2_est))
            y2_est = int(np.round(y2_est))
            
            diff = im2[y2_est-5:y2_est+6,x2_est-5:x2_est+6]-im1[int(y1)-5:int(y1)+6,int(x1)-5:int(x1)+6]
            diff_metric = np.linalg.norm(diff)
            if err is None:
                err = diff_metric
                x2 = x2_est
                y2 = y2_est
            else:
                if diff_metric<err:
                    err = diff_metric
                    x2 = x2_est
                    y2 = y2_est
    return x2,y2