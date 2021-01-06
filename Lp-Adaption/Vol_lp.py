from scipy.special import gamma

def vol_lp(N:int,r,p:int):
    if p > 100:
        return (2*r)**N
    else:
        return (2*gamma(1/p+1)*r)**N/(gamma(N/p+1))