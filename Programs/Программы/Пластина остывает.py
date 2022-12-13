import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import sys

plt.style.use('seaborn-pastel')

# Generate data for plotting
n=12
p=12

X, Y = np.meshgrid(np.linspace(0,30 , n), np.linspace(0,30 , p))

Z = np.zeros((p,n))
Zn = np.zeros((p,n))

Z[1, 1:n-1]=30
Z[p-2, 1:n-1]=30
Z[1:p-1, 1]=30
Z[1:p-1, n-2]=30
print(Z)
 

k=np.linspace(0, 30, 50)


for k in range(160):
    for i in range(p-1):
        for j in range(n-1):
            r = (Z[i+1,j] + Z[i-1,j] + Z[i,j-1] + Z[i,j+1] - 4*Z[i,j])/4
            Zn[i,j] = Z[i,j] + 1.5*r
            Zn[0, 0:n-1]=0
            Zn[n-1, 0:n-1]=0
            Zn[0:n-1, 0]=0
            Zn[0:n-1, n-1]=0
    
    print(Z)        
    Z = Zn #обновление начальных условий
    
    
    
    fig = plt.figure()
    plt.xlabel("X-axis")
    plt.ylabel("Y-plot")
    plt.title("Simple x-y plot")
    
    plt.contourf(X, Y, Z, levels=k)
    

    



# anim.save('C:/NIK/pp.gif', writer=PillowWriter())
