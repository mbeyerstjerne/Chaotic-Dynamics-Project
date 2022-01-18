# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:53:50 2021

@author: Mark
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import gradient
from scipy.signal import savgol_filter
import pandas as pd
from scipy.signal import argrelextrema

def poincare_plot(driving_freq, time, dt, angle, angle_velocity):
    driving_period = 1/driving_freq
    
    poincare_selection_array = np.zeros_like(time)
    
    for i in range(len(time)):
        if np.isclose([time[i] % driving_period] , [0] , atol = dt) == True:
            poincare_selection_array[i] = 1
        else:
            poincare_selection_array[i] = 0
            
    poincare_angle = angle*poincare_selection_array
    poincare_angle_velocity = angle_velocity*poincare_selection_array
    
    return poincare_angle, poincare_angle_velocity
#%%

v_name = "0_3V__0_525kOhm"

file = "C:\\Users\\mbeye\\OneDrive\\3117Lab\\" + v_name + ".txt"
f = open(file)
data = f.readlines()

time = np.array([float(data[i+2].split()[0]) for i in range(len(data)-2)])
angle = np.array([float(data[i+2].split()[1]) for i in range(len(data)-2)])


d_angle = np.gradient(angle,time)

#Savitzky–Golay filter
angle_sg = savgol_filter(angle, 71, 3)
d_angle_sg = savgol_filter(d_angle, 71, 3)

#%%

l_trunc = 6000

time_t, angle_sg_t, d_angle_sg_t = time[0:l_trunc], angle_sg[0:l_trunc], d_angle_sg[0:l_trunc]

poincare_angle, poincare_angle_velocity = poincare_plot(1, time, time[0], angle_sg, d_angle_sg)
fig, axs = plt.subplots(1,3, figsize=(8,4), dpi=200)
fig.suptitle("$R_v=99.5k\Omega$,$V_0=0.3V$")
axs[0].plot(time[0:l_trunc], angle_sg[0:l_trunc])
axs[0].set_title("Voltage vs time")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Voltage (V)")
axs[0].grid()
axs[1].plot(angle_sg[0:l_trunc], d_angle_sg[0:l_trunc], 'tab:red')
axs[1].set_title("Phase plot")
axs[1].set_xlabel("Voltage (V)")
axs[1].set_ylabel("Voltage differential (V)")
axs[1].grid()
axs[2].plot(poincare_angle, poincare_angle_velocity, marker = '.', linestyle='None')
axs[2].set_title('Poincare plot')
axs[2].set_xlabel('Angle (rads)')
axs[2].set_ylabel('Angular velocity (rad/s)')
axs[2].grid()

fig.tight_layout()

#%%

import matplotlib.pyplot as plt
from scipy import integrate
import scipy.constants
from scipy.signal import peak_widths, find_peaks

plt.figure(dpi=300, figsize=(8,4))

Rv = np.linspace(47,147,41)

R_data = ["0_3V__0_000kOhm", "0_3V__0_025kOhm", "0_3V__0_050kOhm", "0_3V__0_075kOhm"]
for i in range(4,40):
    R_data.append("0_3V__0_" + str(i*25) + "kOhm")
    
R_data.append("0_3V__1_000kOhm")

for k in range(0, len(R_data)):
    file = "C:\\Users\\mbeye\\OneDrive\\3117Lab\\" + R_data[k] + ".txt"
    f = open(file)
    data = f.readlines()
        
    time = np.array([float(data[i+2].split()[0]) for i in range(len(data)-2)])
    angle = np.array([float(data[i+2].split()[1]) for i in range(len(data)-2)])
    
    angle_sg = savgol_filter(angle, 71, 3)
    
    peaks, _ = find_peaks(angle_sg, prominence=0.05)
    peak_points = angle_sg[peaks]
    plt.plot(np.ones_like(peak_points) * Rv[k], peak_points, 'k.', color='black')
    
plt.grid()
plt.title("Bifurcation diagram for chaotic circuit")
plt.xlabel("$R_v$ ($k\Omega$)")
plt.xlim(20,147)
plt.ylabel("Amplitude of maxima at bifurcation point")
plt.plot()


#%%

v_name = "0_3V__0_050kOhm"

file = "C:\\Users\\mbeye\\OneDrive\\3117Lab\\" + v_name + ".txt"
f = open(file)
data = f.readlines()
    
time = np.array([float(data[i+2].split()[0]) for i in range(len(data)-2)])
angle = np.array([float(data[i+2].split()[1]) for i in range(len(data)-2)])

angle_sg = savgol_filter(angle, 71, 3)

peaks, _ = find_peaks(angle_sg, prominence=0.1)
peak_points = angle_sg[peaks]
plt.plot(np.ones_like(peak_points) * Rv[i], peak_points, 'k.')
    
plt.grid()
plt.plot()



#%%

v_name = "0_3V__0_050kOhm"

file = "C:\\Users\\mbeye\\OneDrive\\3117Lab\\" + v_name + ".txt"
f = open(file)
data = f.readlines()
    
time = np.array([float(data[i+2].split()[0]) for i in range(len(data)-2)])
angle = np.array([float(data[i+2].split()[1]) for i in range(len(data)-2)])

angle_sg = savgol_filter(angle, 71, 3)

def poincare_plot(driving_freq, time, dt, angle, angle_velocity):
    driving_period = 1/driving_freq
    
    poincare_selection_array = np.zeros_like(time)
    
    for i in range(len(time)):
        if np.isclose([time[i] % driving_period] , [0] , atol = dt) == True:
            poincare_selection_array[i] = 1
        else:
            poincare_selection_array[i] = 0
            
    poincare_angle = angle*poincare_selection_array
    poincare_angle_velocity = angle_velocity*poincare_selection_array
    
    return poincare_angle, poincare_angle_velocity

poincare_angle, poincare_angle_velocity = poincare_plot(1, time, time[0], angle_sg, d_angle_sg)

plt.plot(poincare_angle, poincare_angle_velocity, marker = '.', linestyle='None')
plt.title('Poincare plot, $R_v=72k\Omega$,$V_0=0.3V$')
plt.xlabel('Angle (rads)')
plt.ylabel('Angular velocity (rad/s)')

#%% fourier power spectra

dt = time[0]

def fourier_spectrum(spacing,channel,kaiser=False):
#Simple function to automate the fft process. Time-spacing between data points, array of y-data to be transformed. 
#kaiser=True appends a Kaiser window to the data.
    number_samples = len(channel)
    if kaiser==True:
        channel = channel*np.kaiser(number_samples,2)
    
    freq_space = np.fft.rfftfreq(number_samples,spacing)
    fft_space = np.fft.rfft(channel)*2/number_samples
    fft_x = freq_space[:number_samples//2]
    fft_y = abs(fft_space[:number_samples//2])
    return fft_x,fft_y #Frequency x-axis data, fourier y-axis data.

fft_freq, fft_amp = fourier_spectrum(dt, angle_sg)

plt.figure(figsize=(8,3), dpi=200)
plt.plot(fft_freq, fft_amp)
plt.grid()
plt.title("Fourier power spectral density, $R_v=72k\Omega$,$V_0=0.3V$")
plt.ylabel("Power spectral density")
plt.xlabel("Frequency (Hz)")
plt.yscale('log')
plt.xlim(0, 15)


#%% recurrence plot
from scipy.spatial.distance import pdist, squareform

def rec_plot(s, eps=0.10, steps=10):
    d = pdist(s[:,None])
    d = np.floor(d/eps)
    d[d>steps] = steps
    Z = squareform(d)
    return Z

rec = []

rec = (rec_plot(angle_sg[0:6000], eps=0.02))
fig, ax = plt.subplots(1,1)
plt.title("Recurrence plot, $R_v = 57k\Omega$, $V_0 = 0.3V$")
plt.xlabel("Signal point n")
plt.ylabel("Signal point n")
plt.imshow(rec, cmap='gray')
ax.invert_yaxis()

#%%
# Generate a noisy AR(1) sample

rs = angle_sg_t
xs = [0]
for r in rs:
    xs.append(xs[-1] * 0.9 + r)
df = pd.DataFrame(xs, columns=['data'])

n = 10  # number of points to be checked before and after

# Find local peaks

df['min'] = df.iloc[argrelextrema(df.data.values, np.less_equal,
                    order=n)[0]]['data']
df['max'] = df.iloc[argrelextrema(df.data.values, np.greater_equal,
                    order=n)[0]]['data']

import array
a = np.ones_like(df['max'])*(47+97.5)

plt.scatter(a, df['max'])


#%% sg filtering example

plt.figure(figsize=(8,2.5), dpi=200)
plt.plot(time, angle, label="Voltage signal (original)")
plt.plot(time, angle_sg, label="Voltage signal (with SG filter applied)", alpha=0.5, color="black", linewidth =2)
plt.title("Example application of Savitzky-Golay filtering")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()



#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import argrelextrema

rs = angle_sg
xs = [0]
for r in rs:
    xs.append(xs[-1] * 0.9 + r)
df = pd.DataFrame(xs, columns=['data'])

n = 10

df['max'] = df.iloc[argrelextrema(df.data.values, np.greater_equal,
                    order=n)[0]]['data']

maxi0 = df['max']
maxi0 = maxi0.dropna()
maxi0 = maxi0[0:len(maxi0)-2]

maxi1 = df['max']
maxi1 = maxi1.dropna()
maxi1 = maxi1[1:len(maxi1)-1]

maxi2 = df['max']
maxi2 = maxi2.dropna()
maxi2 = maxi2[2:,]

fig, axs = plt.subplots(1,2, figsize=(8,4), dpi=200, gridspec_kw={'width_ratios': [1, 1]})
fig.suptitle("Return map of chaotic circuit, $R_v = 97k\Omega$,$V_0=0.3V$")
axs[0].plot(maxi0, maxi1, marker = '.', linestyle='None', color='black')
axs[0].scatter(np.linspace(0,2.5, 40), np.linspace(0,2.5, 40), s=0.6, color='red')
axs[0].set_title("1st return map")
axs[0].set_xlabel("x(n) (V)")
axs[0].set_ylabel("x(n+1) (V)")
axs[0].set_xlim(0,2.5)
axs[0].set_ylim(0,2.5)
axs[0].grid()
axs[1].plot(maxi0, maxi2, 'tab:red', marker = '.', linestyle='None', color='black')
axs[1].scatter(np.linspace(0,2.5, 40), np.linspace(0,2.5, 40), color='red',s=0.6)
axs[1].set_title("2nd return map")
axs[1].set_xlabel("x(n) (V)")
axs[1].set_ylabel("x(n+2) (V)")
axs[1].set_xlim(0,2.5)
axs[1].set_ylim(0,2.5)
axs[1].grid()

fig.tight_layout()


#%%
d_X = [1.869, 1.515, 1.231, 0.927, 0.617, 0.313, 0.003, 0.007, 0.007, 0.007]
X = [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15]

plt.plot(X, d_X)
plt.plot(X, d_X, marker = '.', linestyle='None')
plt.title("$D(x)$ vs Voltage signal $x$")
plt.xlabel("Voltage signal, $x$ (V)")
plt.grid()
plt.ylabel("Non-linearity function $D(x)$, (V)")


