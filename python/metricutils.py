'''
metric utils
'''
import numpy as np

def LAP(mtr):
    [nx,ny] = mtr.shape
    lap = 0
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            lap += (mtr[i+1,j]+mtr[i-1,j]+mtr[i,j+1]+mtr[i,j-1]-4*mtr[i,j])**2
    return lap

def GRA(mtr):
    gra = 0
    [nx,ny] = mtr.shape
    for i in range(1,nx):
        for j in range(1,ny):
            gra += np.sqrt((mtr[i,j]-mtr[i-1,j])**2+(mtr[i,j]-mtr[i,j-1])**2)
    return gra

def TC(mtr):
    mtr = np.reshape(mtr, (1, mtr.shape[0]*mtr.shape[1]))
    m = np.mean(mtr)
    v = np.var(mtr)
    std = np.power(v, 0.5)
    tc = np.power(std / m, 0.5)
    return tc

def mix_metric(mtr):
    '''
    in order to reduce the time
    :param mtr:
    :return:
    '''
    gra = 0
    lap = 0
    [nx,ny] = mtr.shape
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            lap += (mtr[i+1,j]+mtr[i-1,j]+mtr[i,j+1]+mtr[i,j-1]-4*mtr[i,j])**2
            gra += np.sqrt((mtr[i,j]-mtr[i-1,j])**2+(mtr[i,j]-mtr[i,j-1])**2)
    for i in range(nx-1,nx):
        for j in range(ny-1,ny):
            gra += np.sqrt((mtr[i,j]-mtr[i-1,j])**2+(mtr[i,j]-mtr[i,j-1])**2)

    mtr = np.reshape(mtr, (1, mtr.shape[0]*mtr.shape[1]))
    m = np.mean(mtr)
    v = np.var(mtr)
    std = np.power(v, 0.5)
    tc = np.power(std / m, 0.5)
    return lap,gra,tc



if __name__=="__main__":
    mtr = np.random.rand(1024, 1024)
    # lap = LAP(mtr)
    # gra = GRA(mtr)
    # tc = TC(mtr)
    lap2,gra2,tc2= mix_metric(mtr)