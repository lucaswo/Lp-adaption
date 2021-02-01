import numpy as np

def lowner(a,tol):
    '''
    :param a:
    :param tol:
    :return:LOWNER Approximate Lowner ellipsoid.
   [E C]=LOWNER(A,TOL) finds an approximation of the Lowner ellipsoid
   of the points in the columns of A.  The resulting ellipsoid satisfies
       x=A-repmat(C,size(A,2)); all(dot(x,E*x)<=1)
   and has a volume of approximately (1+TOL) times the minimum volume
   ellipsoid satisfying the above requirement.

   A must be real and non-degenerate.  TOL must be positive.

   Usually you can get faster results by using only the points on
   the convex hull, e.g.:
       [E c]=lowner(a(:,unique(convhulln(a'))),tol)

   Example:
       a=randn(2,100);
       [E c]=lowner(a,0.001);
       t=linspace(0,2*pi);
       [V D]=eig(E);
       e=repmat(c,size(t))+V*diag(1./sqrt(diag(D)))*[cos(t);sin(t)];
       plot(a(1,:),a(2,:),'+',e(1,:),e(2,:),'-')

   Reference:
       Khachiyan, Leonid G.  Rounding of Polytopes in the Real Number
           Model of Computation.  Mathematics of Operations Research,
           Vol. 21, No. 2 (May, 1996) pp. 307--320.
    '''
    (n,m) = a.shape
    if n<1:
        ValueError('Input must be in one dimension or higher.')
    F = khachiyan(np.vstack((a,np.ones(shape=(1,m)))),tol)
    A = F[0:n,0:n]
    b = F[0:n,-1].reshape(n,1)
    c = np.linalg.lstsq(-A, b)[0].reshape(n,1)
    E = A/((1-c.T)@b.reshape(n,1)-F[-1,-1])

    ac = a - np.tile(c,(1,m))
    return E/(np.max(np.sum(ac*(E@ac),axis=0)))




def khachiyan(a,tol):
    (n, m) = a.shape
    if n<2:
        ValueError('n must be 2 or higher!')
    elif not np.real(a).all():
        ValueError('inputs mus be real')
    elif not (np.real(tol) and tol>0):
        ValueError('Tolerance must be positive')

    invA = m * np.linalg.inv(a@a.T)
    w = np.sum(a*(invA@a),axis=0).reshape(1,m)

    while True:
        w_r, r = np.max(w),np.argmax(w)
        f = w_r/n
        eps = f-1
        if eps<=tol:
            break
        g = eps/((n-1)*f)
        h = 1+g
        g = g/f
        b = invA@a[:,r]
        invA = h*invA-g*b@b.T
        bTa = b.T@a
        w = h*w-g*(bTa*bTa)

    return invA/w_r

'''
 Accumulated roundoff errors may cause the ellipsoid 
 to not quite cover all the points.
 Use
   E=invA/max(dot(a,invA*a,1));
 if you don't like that.
'''