from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import time

class Plotter():
    def __init__(self):
        self.fig = plt.figure()
        self.ax = plt.axes(projection='3d')
        self.x_buff = []
        self.y_buff = []
        self.z_buff = []
        self.buffer_len = 50
    
    def update(self, x, y, z):
        
        if len(self.x_buff) > 50:
            self.x_buff.pop(0)
            self.y_buff.pop(0)
            self.z_buff.pop(0)

        self.x_buff.append(x)
        self.y_buff.append(y)
        self.z_buff.append(z)
        self.ax.plot3D(self.x_buff, self.y_buff, self.z_buff, 'gray')

        plt.pause(0.05)
    
    def update_scatter(self, x, y, z):
        self.ax.scatter3D(x, y, z, cmap='Greens');
        plt.pause(0.05)

if __name__ == "__main__":
    plotter = Plotter()
    for i in range(100):
        plotter.update(-i, -i, -i)
    time.sleep(0.01)
