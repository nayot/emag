# 2D Vector Field Plot Function
import matplotlib.pyplot as plt
import numpy as np
# scienceplots package ช่วยให้สามารถปรับแต่งให้กราฟสวยขึ้น แต่สำหรับบางเครื่องที่ไม่มีระบบจะ skip คำสั่งนี้โดยอัตโนมัติ
try:
    import scienceplots
    plt.style.use(['science', 'grid', 'notebook'])
except ImportError:
    pass
# ฟังก์ชันสำหรับ plot กราฟเวคเตอร์ โดยรับฟังกชัน field_func ซึ่งใช้ในการคำนวณค่าของสนาม ณ จุดต่าง ๆ
def field_plot(field_func, num_grids=20,\
               x_min=-10, x_max=10, y_min=-10, y_max=10,\
               xlabel='x', ylabel='y', title=''):
    x = np.linspace(x_min, x_max, num_grids)
    y = np.linspace(y_min, y_max, num_grids)
    
    X, Y = np.meshgrid(x, y)
    
    E_field = np.vectorize(field_func, signature='(),()->(n)')(X, Y)
    
    Ex = E_field[:,:,0]
    Ey = E_field[:,:,1]
    
    Emax = np.std(Ex) * 0.05
    Ex[Ex>Emax] = Emax
    Ey[Ey>Emax] = Emax
    Ex[Ex<-Emax] = -Emax
    Ey[Ey<-Emax] = -Emax
    
    plt.quiver(X, Y, Ex, Ey)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()