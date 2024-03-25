# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:55:19 2024

@author: irodriguez
"""

import numpy as np
import matplotlib.pyplot as plt

# PARAMETERS___________________________________________________________________
mass = 1800; # Proton mass normalized to the electron
charge = 1; # Proton charge normalized to the electron
c=100; #Light velocity at um/ps

duration = 10000; #Number of loop iterations
dt = 1; # Time step


# ARRAYS DEFINITIONS___________________________________________________________
vi = np.array([0. , 1. , 0 ]);
xi = np.array([-1. ,0. ,0. ]);

B = np.array([0. , 0. , 20 ]); #Constant B in z direction
E = np.array([0. ,0. , 1. ]); #Constant E in z direction

xf = np.zeros((duration,3));
vf = np.zeros((duration,3));


# HIGUERA ALGORITHM____________________________________________________________
for time in range(duration):
    epsilon = (charge / mass) * E * 0.5 * dt;
    beta = (charge / mass) * B * 0.5 * dt;
    s = 2. * beta / (1. + beta**2);
    v_minus = vi + epsilon;
    gammat1 = 1 + (v_minus/c)**2 - beta**2;
    gammat2 = gammat1**2 + 4*(beta**2 + np.absolute(np.dot(beta,v_minus**2)) *c**(-2));
    gamma_new = np.sqrt(0.5 * (gammat1 + np.sqrt(gammat2)));
    v_new = v_minus - (1/gamma_new) * np.cross(beta,v_minus) + (1/gamma_new**2) * np.dot(beta**2,v_minus);
    v_plus = v_minus + np.cross(v_new,s);
    xf[time,:] += vi * dt;
    vf[time,:] = v_plus + epsilon;
    xi = xf[time,:];
    vi = vf[time,:];

# X & YREPRESENTATION FOR EVERY TEMPORAL ITERATION_____________________________
plt.plot(xf[:,0],xf[:,1],linewidth=2.0, color='red'); 
plt.xlabel('x(t)', fontsize=18, color='black');
plt.ylabel('y(t)', fontsize=18, color='black');
plt.title('Higuera Method',fontsize=20);
plt.show()

# 3D REPRESENTATION FOR EVERY TEMPORAL ITERATION_______________________________
fig = plt.figure();
ax = fig.add_subplot(111, projection='3d');
ax.plot3D(xf[:, 0],xf[:, 1], xf[:, 2]);

# MAGNETIC FIELD IN Z__________________________________________________________
z = [0, 1, 2, 3, 5];
x2 = [0] * len(z);
y2 = [0] * len(z);
ax.plot(x2, y2, z)

plt.xlabel('x');
plt.ylabel('y');
ax.set_zlabel('z');
plt.title('Temporal evolution of the particle',fontsize=10);


# OUTPUT_______________________________________________________________________
import csv
with open('data_x_higuera.csv', 'w', newline='') as csvfile:
    dataX = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_NONE);
    for time in range(duration):
        dataX.writerow(xf[time]);