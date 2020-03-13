
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import misc
from matplotlib import gridspec

## From Labrosse PEPI 2015
## little r denotes subscript rho
## little c denotes a value at the ICB


#constants
G = 6.67408e-11 #m^3/kgs^2 ---> Gravity
rho0 = 12451 # kg/m^3
Lr = 8039e3 #m -> characteristic length scale
Ar = 0.484 #unitless
P0 = 360e9 #GPa
K0 = 1403e9 #GPa
K0p = 3.567
#Tic = 5500 # K, the T at the ICB --> change this for HfB model
roc = 3480e3 #m, the radius of the core
#ric = 1221e3 #m, the radius of the IC
gamma = 1.5
k0 = 100 #W/mK, thermal conductivity at the center of the core --> change this for HfB model
Ak = 0 #radial dependence of conductivity
Cp = 750 #J/Kkg, Heat capacity
delS = 127 #J/Kkg, Entropy of crystallization
beta = 0.83 # Coefficient of compositional expansion
dTdxi = -21e3 #K
dTdP = 9e-9 #K/Pa
xi0 = 0.056 #%
delrhoxi = 580 #kg/m^3, difference in mass fraction of light elements across the ICB
#Tl0 = 5848.99392 #K, the present day melting temperature at the center
#T0 = 5698.21168 #K, the present day adiabatic temperature at the center
#Qcmb = 13e12
#Tcmb = 4100

#functions
def rho(r):
    return rho0 * (1.0 - r**2 / Lr**2 - Ar * r**4 / Lr**4)

def g(r):
    return (4.0 * np.pi / 3.0 * G * rho0 * r *
            (1 - 3.0 / 5.0 * r**2 / Lr**2 - 3.0 * Ar / 7.0 * r**4 / Lr**4))
def P(r):
    return P0 - K0*(r**2 / Lr**2 - 4.0 / 5.0 * r**4 / Lr**4)

def Ta(r, Tcmb):
    return Tcmb * (rho(r) / rho(roc))**gamma

def T0(r, Tcmb):
    Ta(0, Tcmb)

def Tic(r, Tcmb):
    Ta(ric, Tcmb)

def dTadroc(roc, Tcmb):
    return -gamma * Ta(roc, Tcmb) / rho(roc) * sp.misc.derivative(rho, roc)

def dTadric(ric, Tcmb):
    return -gamma * Ta(ric, Tcmb) / rho(ric) * sp.misc.derivative(rho, ric)

def fc(x, delta):
    return x**3 * (1.0 - 3.0 / 5.0 * (delta + 1) * x**2 - 3.0 / 14.0 *
                   (delta + 1) * (2 * Ar - delta) * x**4)

def fx(x, h):
    return x**3 * (-h**2 / (3 * Lr**2) + 1.0 / 5.0 * (1 + h**2 / Lr**2) * x**2 -
                   13.0 / 70.0 * x**4)

def fk(x):
    return x**5 / 5.0 * (1 + 5.0 / 7.0 * (2.0 - Ak + 4.0 * Ar) *
                         x**2 + 5.0 / 9.0 * (3.0 + 10.0 * Ar + 4.0 * Ar**2 -
                                             2.0 * Ak * (1 + Ar)) * x**4)

## anchoring the melting curve to the current-day adiabatic T at the ICB
Tl0 = (Ta(1221e3, 4100) + K0 * dTdP * (1221e3)**2 / Lr**2 - dTdxi * xi0 *
       (1221e3)**3 / (Lr**3 * fc((roc / Lr), 0)))

def Tl(ric):
    return (Tl0 - K0 * dTdP * ric**2 / Lr**2 + dTdxi * xi0 * ric**3 /
            (Lr**3 * fc((roc / Lr), 0)))

def dTldric(ric):
    return (-2 * K0 * dTdP * ric / Lr**2 + 3 * dTdxi * xi0 * ric**2 /
            (Lr**3 * fc(roc / Lr, 0)))

def k(r):
    return k0 * (1 - Ak * r**2 / Lr**2)

def Sk(k0):
    return 16.0 * np.pi * gamma**2 * k0 * Lr * (fk(roc / Lr) - fk(ric / Lr))

# Plotty plot
plt.figure()
r = np.arange(0,3480e3)
plt.plot(r/10**3,Ta(r,4100),label='Adiabat')
plt.plot(r/10**3,Tl(r),label='Melting curve')
plt.legend()
plt.xlabel('radius (km)')
plt.ylabel('Temperature (K)')

# Preallocate arrays
Moc = np.ones(4500) * 1.84e24
ric = np.ones(4500) * 1223e3
Pl = np.ones(4500) * 1.59e23
Sl = np.ones(4500) * 1.04e19
Px = np.ones(4500) * 1.01e23
Sx = np.ones(4500) * 2.49e19
Sk = np.ones(4500) * 5.56e8
Tphi = np.ones(4500) * 4671
Tc = np.ones(4500) * 4675
Pic = np.ones(4500) * 3.64e23
Pc = np.ones(4500) * -1.20e27
Sc = np.ones(4500) * -4.01e22
Sic = np.ones(4500) * 7.35e19
Psquig = np.ones(4500) * 6.24e23
Esquig = np.ones(4500) * 1.09e20
Ephi = np.ones(4500) * 500e6
#Qcmbtest = np.ones(4500)*12.99e12
Qcmb = np.ones(4500) * 16.6e12
Qk = np.ones(4500) * 3.19e12
dri = np.ones(4500) * -656
Tcmb = np.ones(4500) * 4100
T0 = np.ones(4500) * Ta(0, Tcmb)
time = np.zeros(4500)
dTcmb = np.ones(4500) * 0.01
dPdri = np.ones(4500) * -5.4e4

dt = -3.154e13 #s/Myr
for t in range(1,4500):
    dPdri[t] = sp.misc.derivative(P,ric[t-1])
    Moc[t] = 4.0 * np.pi / 3.0 * rho0 * Lr**3 * (fc(roc / Lr, 0) -
                                                 fc(ric[t-1] / Lr, 0))
    
    Pl[t] = 4.0 * np.pi * ric[t-1]**2 * Tl(ric[t-1]) * rho(ric[t-1]) * delS
    Sl[t] = (Pl[t-1] *
             (Ta(ric[t-1], Tcmb[t-1]) - Tcmb[t-1]) /
             (Tcmb[t-1] * Ta(ric[t-1], Tcmb[t-1])))
    Px[t] = (8.0 * np.pi**2 * xi0 * G * rho0**2 * beta *
             (ric[t-1]**2 * Lr**2 / fc(roc / Lr, 0)) *
             (fx(roc / Lr, ric[t-1]) - fx(ric[t-1] / Lr, ric[t-1])))
    Sx[t] = Px[t-1] / Tcmb[t-1]
    Tphi[t] = (Tl(ric[t-1]) /
               (1.0 - ric[t-1]**2 / Lr**2 - Ar * ric[t-1]**4 / Lr**4)**gamma *
               (fc(roc / Lr, 0) - fc(ric[t-1] / Lr, 0)) /
               (fc(roc / Lr, -gamma) - fc(ric[t-1] / Lr, -gamma)))
    Tc[t] = (Tl(ric[t-1]) *
             (1 - ric[t-1]**2 / Lr**2 - Ar * ric[t-1]**4 / Lr**4)**-gamma *
             (fc(roc / Lr, gamma) - fc(ric[t-1] / Lr, gamma)) /
             (fc(roc / Lr, 0) - fc(ric[t-1] / Lr, 0)))
    Pc[t] = -4.0 * np.pi / 3.0 * rho0 * Cp * Lr**3 * (fc(roc / Lr, gamma))
    Pic[t] = (-4.0 * np.pi / 3.0 * rho0 * Cp * Lr**3 *
              (1 - ric[t-1]**2 / Lr**2 - Ar * ric[t-1]**4 / Lr**4)**-gamma *
              (dTldric(ric[t-1]) + 2 * gamma * Tl(ric[t-1]) * ric[t-1] / Lr**2 *
               (1 + 2 * Ar * ric[t-1]**2 / Lr**2) /
               (1 - ric[t-1]**2 / Lr**2 - Ar * ric[t-1]**4 / Lr**4)) *
              (fc(roc / Lr, gamma)))
    Sic[t] = Pic[t-1] * (Tc[t-1] - Tcmb[t-1]) / (Tc[t-1] * Tcmb[t-1])
    Sc[t] = Pc[t-1] * (Tc[t-1] - Tcmb[t-1]) / (Tcmb[t-1] * Tc[t-1])
    Psquig[t] = Pl[t-1] + Px[t-1] + Pic[t-1]
    Esquig[t] = Sl[t-1] + Sx[t-1] + Sic[t-1]
    Sk[t] = 16.0 * np.pi * gamma**2 * k0 * Lr * (fk(roc / Lr) -
                                                 fk(ric[t-1] / Lr))
    # adiabatic heat flux
    Qk[t] = 4 * np.pi * roc**2 * k0 * dTadroc(roc, Tcmb[t-1])
    if ric[t-1] == 0.0:
        Ephi[t-1] = 0.
        Ephi[t] = 0.
        Qcmb[t] = Qk[t-1]
        dri[t] = 0
        dTcmb[t] = (Qcmb[t-1] / Pc[t-1]) * dt
    else:
        Ephi[t] = (Sic[t-1] * (dri[t-1] / dt) + Sl[t-1] * (dri[t-1] / dt) +
                   Sx[t-1] * (dri[t-1] / dt) - Sk[t-1])
        Qcmb[t] = 16.6e12
        dri[t]= (Qcmb[t-1] / (Pl[t-1] + Px[t-1] + Pic[t-1])) * dt
        dTcmb[t] = (dri[t-1] *
                    (dTldric(ric[t-1]) + dTadric(ric[t-1], Tcmb[t-1])) *
                    (Ta(roc, Tcmb[t-1])) /
                    (Ta(ric[t-1], Tcmb[t-1])))
    Tcmb[t] = Tcmb[t-1] + dTcmb[t-1]
    T0[t] = Ta(0, Tcmb[t-1])
    ric[t] = ric[t-1] + dri[t-1]
    # No negative core radii
    if ric[t] < 0.:
        ric[t] = 0.
    time[t] = time[t-1] + dt

# Plot em up
fig=plt.figure(figsize=(7.5,5))
gs1 = gridspec.GridSpec(2, 2, hspace=0.25, left=0.08, right=0.99,
                        top=0.99, bottom=0.15)
ax1=fig.add_subplot(gs1[0, 0])
ax1.plot(-time[3:4500] / 3.154e16, Qcmb[3:4500] / 10**12)
plt.xlabel("Age (Gy)")
plt.ylabel("Qcmb (TW)")

ax2=fig.add_subplot(gs1[0, 1])
ax2.plot(-time[3:4500] / 3.154e16, Ephi[3:4500] / 10**6)
#plt.ylim(-250,900)
plt.xlabel("Age (Gy)")
plt.ylabel("Ephi (MW/K)")

ax3=fig.add_subplot(gs1[1, 0])
ax3.plot(-time / 3.154e16, ric / 10**3)
plt.xlabel("Age (Gy)")
plt.ylabel("Inner core radius (km)")

ax4=fig.add_subplot(gs1[1, 1])
ax4.plot(-time / 3.154e16, Tcmb)
plt.xlabel("Age (Gy)")
plt.ylabel("CMB Temperature (K)")

plt.savefig('ThermalEv_constantQcmb.pdf')