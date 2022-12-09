import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import networkx as nx

plt.style.use('seaborn-pastel')

# Generate data for plotting
n=20
X, Y = np.meshgrid(np.linspace(0, 10, n), np.linspace(0, 10, n))

a = np.zeros((n,n))
b = np.zeros((n,n))
a[1, 1:n-1]=30
a[n-2, 1:n-1]=30
a[1:n-1, 1]=30
a[1:n-1, n-2]=30

for i in range(n-1):
    for j in range(n-1):
        r = (a[i+1,j] + a[i-1,j] + a[i,j-1] + a[i,j+1] - 4*a[i,j])/4
        b[i,j] = a[i,j] + 0.1*r
        b[0, 0:n-1]=0
        b[n-1, 0:n-1]=0
        b[0:n-1, 0]=0
        b[0:n-1, n-1]=0
    a = b
    
Nt = 30
def some_data(i):   # function returns a 2D data array
    return a * (i/Nt)

fig = plt.figure()
ax = plt.axes(xlim=(0, 10), ylim=(0, 10))

levels = np.linspace(0,30,10)


cont = plt.contourf(X, Y, some_data(0), levels=levels)

def animate(i):
    global cont
    z = some_data(i)
    for c in cont.collections:
        c.remove()  # removes only the contours, leaves the rest intact
    cont = plt.contourf(X, Y, a, levels)
    plt.title('t = %i:  %.2f' % (i,z[5,5]))
    return cont
   
 
anim = FuncAnimation(fig, animate, frames=Nt, repeat=False)
anim.save('C:/NIK/pp.gif', writer=PillowWriter())
