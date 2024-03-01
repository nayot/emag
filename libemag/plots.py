# 2D Vector Field Plot Function
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
# scienceplots package ช่วยให้สามารถปรับแต่งให้กราฟสวยขึ้น แต่สำหรับบางเครื่องที่ไม่มีระบบจะ skip คำสั่งนี้โดยอัตโนมัติ
try:
    import scienceplots
    plt.style.use(['science', 'grid', 'notebook'])
except ImportError:
    pass
# ฟังก์ชันสำหรับ plot กราฟเวคเตอร์ โดยรับฟังกชัน field_func ซึ่งใช้ในการคำนวณค่าของสนาม ณ จุดต่าง ๆ
def field_plot(field_func, num_grids=20,\
               x_min=-10, x_max=10, y_min=-10, y_max=10,\
               xlabel='x', ylabel='y', title='',\
               contour=False, cmap='plasma', streamline=False, normalising=True, normalising_factor=0.05):
    x = np.linspace(x_min, x_max, num_grids)
    y = np.linspace(y_min, y_max, num_grids)
    
    X, Y = np.meshgrid(x, y)
    
    E_field = np.vectorize(field_func, signature='(),()->(n)')(X, Y)
    
    Ex = E_field[:,:,0]
    Ey = E_field[:,:,1]
    
    E = np.sqrt(Ex**2 + Ey**2)
    
    if normalising:
        Emax = np.std(E) * 0.05
        Ex[Ex>Emax] = Emax
        Ey[Ey>Emax] = Emax
        Ex[Ex<-Emax] = -Emax
        Ey[Ey<-Emax] = -Emax
    
    ax = plt.axes()
    
    if contour:
        E = np.log(E)
        Emin = E.min()
        Emax = E.max()
        ax.contourf(X, Y, E, levels=np.linspace(Emin, Emax, 100), cmap=cmap)
    
    if streamline == True:
        ax.streamplot(X, Y, Ex, Ey)
    else:
        ax.quiver(X, Y, Ex, Ey)
    
    # Countour Plot
    
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    return ax


def fast_field_plot(field_func, num_grids=20,\
               x_min=-10, x_max=10, y_min=-10, y_max=10,\
               xlabel='x', ylabel='y', title='',\
               contour=False, cmap='plasma'):
    x = np.linspace(x_min, x_max, num_grids)
    y = np.linspace(y_min, y_max, num_grids)
    
    X, Y = np.meshgrid(x, y)
    
    # Parallel computation of E_field
    def compute_E_field(x, y):
        return field_func(x, y)
    
    E_field = np.array(Parallel(n_jobs=-1)\
                       (delayed(compute_E_field)(X[i, j], Y[i, j])\
                         for i in range(num_grids) for j in range(num_grids)))
    E_field = E_field.reshape(num_grids, num_grids, -1)
    
    Ex = E_field[:,:,0]
    Ey = E_field[:,:,1]
    E = np.sqrt(Ex**2 + Ey**2)
    
    Emax = np.std(E) * 0.05
    Ex[Ex>Emax] = Emax
    Ey[Ey>Emax] = Emax
    Ex[Ex<-Emax] = -Emax
    Ey[Ey<-Emax] = -Emax
    
    ax = plt.axes()
    
    if contour:
        E = np.log(E)
        Emin = E.min()
        Emax = E.max()
        ax.contourf(X, Y, E, levels=np.linspace(Emin, Emax, 100), cmap=cmap)
    
    ax.quiver(X, Y, Ex, Ey)
    
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    return ax


import time
def fast_field_plot2(field_func, num_grids=20,\
               x_min=-10, x_max=10, y_min=-10, y_max=10,\
               xlabel='x', ylabel='y', title='',\
               contour=False, cmap='plasma'):
    x = np.linspace(x_min, x_max, num_grids)
    y = np.linspace(y_min, y_max, num_grids)
    
    X, Y = np.meshgrid(x, y)
    
    # Parallel computation of E_field
    def compute_E_field(i, j):
        return field_func(X[i, j], Y[i, j])
    
    time1 = time.time()
    E_field_list = Parallel(n_jobs=-1)(delayed(compute_E_field)(i, j)\
                                        for i in range(num_grids) for j in range(num_grids))
    time2 = time.time()
    print(time2 - time1)

    time1 = time.time()
    E_field = np.array(E_field_list).reshape(num_grids, num_grids, -1)
    time2 = time.time()
    print(time2 -time1)

    Ex = E_field[:,:,0]
    Ey = E_field[:,:,1]
    E = np.sqrt(Ex**2 + Ey**2)
    
    Emax = np.std(E) * 0.05
    Ex[Ex>Emax] = Emax
    Ey[Ey>Emax] = Emax
    Ex[Ex<-Emax] = -Emax
    Ey[Ey<-Emax] = -Emax
    
    ax = plt.axes()
    
    if contour:
        E = np.log(E)
        Emin = E.min()
        Emax = E.max()
        ax.contourf(X, Y, E, levels=np.linspace(Emin, Emax, 100), cmap=cmap)
    
    ax.quiver(X, Y, Ex, Ey)
    
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    return ax


def fast_field_plot3(field_func, num_grids=20,\
               x_min=-10, x_max=10, y_min=-10, y_max=10,\
               xlabel='x', ylabel='y', title='',\
               contour=False, cmap='plasma'):
    x = np.linspace(x_min, x_max, num_grids)
    y = np.linspace(y_min, y_max, num_grids)
    
    X, Y = np.meshgrid(x, y)
    
    # Reshape X and Y into 1D arrays
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    # Parallel computation of E_field
    def compute_E_field(idx):
        i, j = np.unravel_index(idx, (num_grids, num_grids))
        return field_func(X_flat[idx], Y_flat[idx])
    
    print("Running in parallel ...")
    E_field_list = Parallel(n_jobs=-1)(delayed(compute_E_field)(idx)\
                                        for idx in range(num_grids*num_grids))
    
    E_field = np.array(E_field_list).reshape(num_grids, num_grids, -1)
    
    Ex = E_field[:,:,0]
    Ey = E_field[:,:,1]
    E = np.sqrt(Ex**2 + Ey**2)
    
    Emax = np.std(E) * 0.05
    Ex[Ex>Emax] = Emax
    Ey[Ey>Emax] = Emax
    Ex[Ex<-Emax] = -Emax
    Ey[Ey<-Emax] = -Emax
    
    ax = plt.axes()
    
    if contour:
        E = np.log(E)
        Emin = E.min()
        Emax = E.max()
        ax.contourf(X, Y, E, levels=np.linspace(Emin, Emax, 100), cmap=cmap)
    
    ax.quiver(X, Y, Ex, Ey)
    
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    return ax


def fast_field_plot4(field_func, num_grids=20, x_min=-10, x_max=10, y_min=-10, y_max=10, xlabel='x', ylabel='y', title='', contour=False, cmap='plasma'):
    x = np.linspace(x_min, x_max, num_grids)
    y = np.linspace(y_min, y_max, num_grids)
    
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    # Ensure field_func can be pickled if it's a lambda or local function.
    # If field_func is a complex function, consider defining it at the top level of your module.
    
    def compute_E_field(x, y):
        return field_func(x, y)
    
    print("Running in parallel ...")
    # Run computations in parallel
    results = Parallel(n_jobs=-1)(delayed(compute_E_field)(X_flat[i], Y_flat[i]) for i in range(len(X_flat)))
    
    E_field = np.array(results).reshape(num_grids, num_grids, -1)
    
    Ex, Ey = E_field[:,:,0], E_field[:,:,1]
    E = np.sqrt(Ex**2 + Ey**2)
    
    # Clipping the field strength for visualization
    Emax = np.std(E) * 0.05
    Ex = np.clip(Ex, -Emax, Emax)
    Ey = np.clip(Ey, -Emax, Emax)
    
    fig, ax = plt.subplots()
    
    if contour:
        E_log = np.log(E + 1)  # Adding 1 to avoid log(0)
        levels = np.linspace(E_log.min(), E_log.max(), 100)
        ax.contourf(X, Y, E_log, levels=levels, cmap=cmap)
    else:
        ax.quiver(X, Y, Ex, Ey)
    
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    plt.show()
    
    return ax


def compute_field_component(field_func, x, y):
    """
    Wrapper function to compute the field component for a single grid point.
    """
    return field_func(x, y)

def fast_field_plot5(field_func, num_grids=20,
               x_min=-10, x_max=10, y_min=-10, y_max=10,
               xlabel='x', ylabel='y', title='',
               contour=False, cmap='plasma', n_jobs=-1):
    x = np.linspace(x_min, x_max, num_grids)
    y = np.linspace(y_min, y_max, num_grids)
    
    X, Y = np.meshgrid(x, y)
    
    # Parallel computation of E_field
    E_field = np.array(Parallel(n_jobs=n_jobs)(
        delayed(compute_field_component)(field_func, X[i, j], Y[i, j]) for i in range(num_grids) for j in range(num_grids)
    )).reshape(num_grids, num_grids, 2)
    
    Ex = E_field[:,:,0]
    Ey = E_field[:,:,1]
    E = np.sqrt(Ex**2 + Ey**2)
    
    Emax = np.std(E) * 0.05
    Ex[Ex>Emax] = Emax
    Ey[Ey>Emax] = Emax
    Ex[Ex<-Emax] = -Emax
    Ey[Ey<-Emax] = -Emax
    
    ax = plt.axes()
    
    if contour:
        E = np.log(E)
        Emin = E.min()
        Emax = E.max()
        ax.contourf(X, Y, E, levels=np.linspace(Emin, Emax, 100), cmap=cmap)
    
    ax.quiver(X, Y, Ex, Ey)
    
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    return ax
