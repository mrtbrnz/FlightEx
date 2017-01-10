#!/usr/bin/env python

# The In-Flight Trust Measurement Fusion Methodology Presented at AIAA SciTech 2017
# Feel free to use as you want as long as you acknowledge the authors and the paper

# BIBTEX FORMAT OF THE PRESENTED PAPER
# @InProceedings{SCITECH2017:DRAG_MEASUREMENT,
#   Title                    = {In-Flight Thrust Measurements using On-Board Force Sensor},
#   Author                   = {Murat Bronz and Hector Garcia de Marina and Gautier Hattenberger},
#   Booktitle                = {Atmospheric Flight Mechanics Conference},
#   Month                    = {January},
#   Year                     = {2017},
#   Organization             = {SciTech},
#   Address                  = {Gaylord Texan, Grapevine, TX},
#   Pages                    = {}
# }


from __future__ import print_function, division
import numpy as np
from numpy import sin, cos, pi, sqrt, dot
from scipy import stats, optimize
from scipy import linalg as la
#from math import sin, cos, pi
import pdb
import pprzlog as pr
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy.interpolate import griddata, interp1d


def gaussian_fusion(mu1,mu2,sigsq1,sigsq2):
    mu = (mu1*sigsq2+mu2*sigsq1)/(sigsq1+sigsq2)
    sigsq = (sigsq1*sigsq2)/(sigsq1+sigsq2)
    return [mu, sigsq]

def gaussian_add(mu1,mu2,sigsq1,sigsq2):
    mu = (mu1+mu2)
    sigsq = (sigsq1+sigsq2)
    return [mu, sigsq]

def gaussian_substraction(mu1,mu2,sigsq1,sigsq2):
    mu = (mu1-mu2)
    sigsq = (sigsq1+sigsq2)
    return [mu, sigsq]

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

print(" Reading the log for ac_id=1 from mako5_flight6.data")
ac_id="1"
filename="mako5_flight6.data"


gps = pr.read_log_gps(ac_id, filename)
esc = pr.read_log_esc(ac_id, filename)
thrust_sensor = pr.read_log_adc_generic(ac_id, filename)


gps_time=gps[:,0]
gps_course=gps[:,4]/10.0*pi/180.0
gps_speed=gps[:,6]/100.0
gps_alt = gps[:,5]/1000.0
#airspeed_sensor = payload[:,4]#-6.0
#airspeed_time   = payload[:,0]
#airspeed_func = interp1d(airspeed_time, airspeed_sensor)


esc_time     = esc[:,0]
esc_rpm      = esc[:,4]
esc_current  = esc[:,1]*1.0 # Calibration of the current measurements...
esc_volatge  = esc[:,2]


thrust_sensor_time   = thrust_sensor[:,0]

thrust_offset = -3.9 # Required in order to tare the preloading of the spring/elastic band !!! Different for each flight !!! :(

thrust_sensor_thrust  = 5.5657e-03 * thrust_sensor[:,1] -7.0239
thrust_sensor_thrust += thrust_offset


gps_speed_func     = interp1d(gps_time,gps_speed)
gps_course_func    = interp1d(gps_time,gps_course)
esc_rpm_func       = interp1d(esc_time, esc_rpm)
esc_current_func   = interp1d(esc_time, esc_current)
esc_voltage_func   = interp1d(esc_time, esc_volatge)
thrust_sensor_func = interp1d(thrust_sensor_time, thrust_sensor_thrust)
gps_alt_av         = movingaverage(gps_alt,2)


# Time Parameters
tf = 850
dt = 0.25 #
time = np.linspace(230, tf, tf/dt)  #np.linspace(0, max_time, max_sample, endpoint=False)
Thrust_propeller_wt = np.zeros(tf/dt)
Thrust = np.zeros(tf/dt)
Thrust_sensor = np.zeros(tf/dt)
Drag = np.zeros(tf/dt)
Drag_coeff = np.zeros(tf/dt)
Thrust_sigsq = np.zeros(tf/dt)
Drag_sigsq = np.zeros(tf/dt)
V_rec = np.zeros(tf/dt)
P11_rec = np.zeros(tf/dt)
P22_rec = np.zeros(tf/dt)
P33_rec = np.zeros(tf/dt)
W_rec_north = np.zeros(tf/dt)
W_rec_east = np.zeros(tf/dt)
W_rec_norm = np.zeros(tf/dt)

M_ac    = 1.0  # Mass of the Aircraft
M_motor = 0.07 # Mass of the motor block
# Windtunnel Propeller Coefficients for APC 9x6
CT0       = 1.342e-01
CT_J      = -1.975e-01
CT_RPM    = 7.048e-06
Prop_diam = 0.23 # Propeller diameter [m]
rho       = 1.225 #kg/m3
S_ref     = 0.27  #m2
# R2 = 0.983
# STD = 0.0039


# States X = [V_tas, W_north, W_east]^T
X1 = np.array([0, 0, 0])
# V_past , F_Net
X2 = np.array([0, 0])


# F prediction matrix
F1 = np.eye(3)
F2 = np.array([[0    , 0], \
               [-M_ac/dt, 0]])

# Input Matrix
G2 = np.array([1.0, M_ac/dt])

# P covariante matrix
P1 = np.array([[0.1,   0,   0], \
              [  0,  0.1,   0], \
              [  0,    0, 0.1] ])
P2 = np.array([[ 1,   0], \
              [  0,   1]])


# Q process noise matrix
qval = 0.0001
Q1 = np.eye(3) * qval
Q2 = np.outer(G2,G2.transpose())*P1[0,0]

# C (KF) sensor model matrix, (EKF) Jacobian of the sensor function
C1 = np.array([[0.0, 0.0, 0.0]]) 
C2 = np.array([[ 0,   0], \
              [  0,   0]])

# R sensor covariance, or measurement noise weight
R1 = np.array([1.0]) * 0.5
R2 = np.array([[  0.5,      0], \
              [     0,    0.5]])

# I identity matrix
I = np.eye(3)



# Assign the initial states
# V_tas   = X1[0]
# W_north = X1[1]
# W_east  = X1[2]

it = 0
for t in time:
    #print("X shape : ", X.shape ,"P shape : ", P.shape, "R shape : ", R.shape, X)
    # Propagation
    X1 = F1.dot(X1)
    P1 = F1.dot(P1).dot(F1.T) + Q1

    # Correction (measurements)
    # Extract V_north and V_east inertial velocities from the GPS ground track
    # and Ground Tracking Angle GTA
    V_g = gps_speed_func(t)
    GTA = gps_course_func(t)
    V_n = V_g * cos(GTA)
    V_e = V_g * sin(GTA)
    
    # Calculate the wind velocity errors
    N_err = V_n - X1[1]
    E_err = V_e - X1[2]

    # Calculate the airspeed magnitude
    V_a_magnitude = sqrt(N_err*N_err+E_err*E_err)
    err = V_a_magnitude - X1[0]

    # C (KF) sensor model matrix, (EKF) Jacobian of the sensor function
    C1[0,0] = -1.
    C1[0,1] = -N_err/V_a_magnitude
    C1[0,2] = -E_err/V_a_magnitude

    # Kalman gain
    S1 = C1.dot(P1).dot(C1.T) + R1
    K1 = P1.dot(C1.T).dot(la.inv(S1))


    # Update the states and covariance matrix
    X1 = X1 + (K1.dot(0-err)).reshape((3,))
    P1 = P1 - K1.dot(C1).dot(P1)


    #######################################
    #######################################
    Q2 = np.outer(G2,G2.transpose())*P1[0,0]

    X2 = F2.dot(X2) + G2*X1[0]
    P2 = F2.dot(P2).dot(F2.T) + Q2

    Force_Sensor  = thrust_sensor_func(t)

    # Update the states and covariance matrix
    # Now calculate the thrust of the propeller from
    # Windtunnel coeficients.
    rev_p_s = esc_rpm_func(t)/60.0
    Adv_ratio = X1[0]/(rev_p_s*Prop_diam)
    Thrust_propeller = rho*rev_p_s**2.0*Prop_diam**4.0*(CT0 + CT_RPM*esc_rpm_func(t) + CT_J*Adv_ratio)

    # Kalman gain
    S2 = C2.dot(P2).dot(C2.T) + R2
    K2 = P2.dot(C2.T).dot(la.inv(S2))

    # Update the states and covariance matrix
    err2 = np.array([Force_Sensor-X2[1] , Thrust_propeller-X2[1]])
    X2 = X2 + K2.dot(err2)
    P2 = P2 - K2.dot(C2).dot(P2)

    #######################################

    # This is the Force from the motor block
    X_accel = X2[1]/M_ac
    F_motor_block = M_motor * X_accel

    #######################################
    sigsq_motor_block = P2[1,1]*M_motor/M_ac
    sigsq_force_sensor = 0.2*0.2 # We looked at the line fit for the worst case point 


    #######################################
    #                  0.8
    sigsq_thrust_rpm = 1.0**2.0 * P1[0,0] # the STD of the surface fit mult by the variance of the V_tas

    F1_thrust, F1_sigsq = gaussian_substraction(Force_Sensor, F_motor_block, sigsq_force_sensor, sigsq_motor_block)

    F_Thrust, F_Thrust_sigsq = gaussian_fusion(F1_thrust, Thrust_propeller, F1_sigsq, sigsq_thrust_rpm)

    F_Drag, F_Drag_sigsq = gaussian_substraction(X2[1] , F_Thrust, P2[1,1] , F_Thrust_sigsq)


    Thrust_propeller_wt[it] = Thrust_propeller
    Thrust[it]        = F_Thrust
    Thrust_sensor[it] = F1_thrust
    Drag[it]          = F_Drag
    Drag_coeff[it]    = (-1*F_Drag)/(0.5*rho*X1[0]**2.0*S_ref) 

    Thrust_sigsq[it]  = F_Thrust_sigsq
    Drag_sigsq[it]    = F_Drag_sigsq

    V_rec[it]       = X1[0]
    W_rec_north[it] = X1[1]
    W_rec_east[it]  = X1[2]
    W_rec_norm[it]  = sqrt(X1[1]*X1[1]+X1[2]*X1[2])

    P11_rec[it] = P1[0,0]
    P22_rec[it] = P1[1,1]
    P33_rec[it] = P1[2,2]

    it += 1

#======================================
# Plotting Part
#======================================

#======================================
# Plot
#======================================
fig_thrust = plt.figure(figsize=(12, 3), dpi=90)
plt.rcParams['legend.loc'] = 'upper right'
plt.rc('font', family='Times New Roman',size=14)

plt.ylabel('Force [N]')
plt.xlabel('Time [s]')
plt.plot(thrust_sensor_time, thrust_sensor_thrust, label='Direct Sensor')
plt.grid()
plt.gca().legend()
plt.tight_layout()
plt.savefig('Flight_results_raw_thrust.pdf')

#======================================
# Plot Estimated Forces
#======================================
fig_thrust = plt.figure(figsize=(12, 4), dpi=90)
plt.rc('font', family='Times New Roman',size=14)
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams['legend.fontsize'] = 12
plt.ylabel('Force [N]')
plt.xlabel('Time [s]')
plt.plot(time, thrust_sensor_func(time), linewidth=0.5, color='black', linestyle='--',label='Direct Sensor')
plt.plot(time, Thrust, linewidth=1, color='red', label='Fusioned Thrust')
plt.plot(time, Drag, linewidth=1, color='green', label='Estimated Drag')
plt.plot(time, Thrust_propeller_wt, linewidth=1, color='blue', label='Wind Tunnel')
plt.xlim([690,750])
plt.ylim([-2,3])
plt.grid()
plt.gca().legend()
plt.tight_layout()
plt.savefig('Flight_results_estimated_forces.pdf')

#======================================
# Plot Power
#======================================
fig_thrust = plt.figure(figsize=(12, 4), dpi=90)
plt.rc('font', family='Times New Roman',size=14)
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams['legend.fontsize'] = 12
plt.ylabel('Power [W] and Efficiency [%]')
plt.xlabel('Time [s]')
plt.plot(time, V_rec*movingaverage(Thrust,20), label='Aerodynamic Power')
plt.plot(time, esc_voltage_func(time)*esc_current_func(time), label='Electrical Power')
plt.plot(time, 100.0*(V_rec*movingaverage(Thrust,25))/(esc_voltage_func(time)*esc_current_func(time)), label='Efficiency')
plt.xlim([300,800])
plt.ylim([0,60])
plt.grid()
plt.gca().legend()
plt.tight_layout()
plt.savefig('Flight_results_power.pdf')


#======================================
# Plot Drag Coeff Histogram
#======================================
# n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
# # add a 'best fit' line
# y = mlab.normpdf(bins, mu, sigma)

fig_thrust = plt.figure(figsize=(12, 4), dpi=90)
plt.rc('font', family='Times New Roman',size=14)
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams['legend.fontsize'] = 12
plt.xlabel('Drag Coefficient')
#plt.xlabel('')
n, bins, patches = plt.hist( Drag_coeff[300:500] , 150, normed=1, alpha=0.5, color='green' )
y = mlab.normpdf(bins, 0.028, 0.0035)
plt.plot(bins, y, 'r--')
# plt.hist( Drag_coeff[400:500] , 150, alpha=0.5, color='blue' )
# plt.hist( Drag_coeff[500:600] , 150, alpha=0.5, color='red' )
# plt.hist( Drag_coeff[600:700] , 150, alpha=0.5, color='black' )
plt.grid()
plt.gca().legend()
plt.tight_layout()
plt.savefig('Flight_results_drag_coeff_hist.pdf')

#======================================
# Plot
#======================================
fig1 = plt.figure(figsize=(12, 9), dpi=90)
plt.rcParams['legend.loc'] = 'upper right'
plt.rc('font', family='Times New Roman',size=14)

plt.subplot(311)
plt.xlabel('Time [s]')
plt.ylabel('Speed [m/s]')
plt.plot(time, gps_speed_func(time), label='Measured GroundSpeed (GPS)')
plt.plot(time, V_rec, linewidth=2, label='Estimated Airspeed (EKF)')
plt.xlim([0,1000])
plt.grid()
plt.gca().legend()

plt.subplot(312)
plt.xlabel('Time [s]')
plt.ylabel('Altitude ASL [m]')

#plt.plot(time, gps_speed_func(time), label='GPS')
plt.plot(gps_time, gps_alt_av, label='GPS Altitude')
plt.ylim([184,300])
plt.gca().legend()
plt.grid()

plt.subplot(313)
plt.xlabel('Time [s]')
plt.ylabel('Propeller RPM')
plt.plot(esc_time, esc_rpm , label='Propeller RPM')

plt.grid()
plt.gca().legend()
plt.tight_layout()
plt.savefig('Flight_results_alt_airspeed_rpm.pdf')





# Final Plotting Part
fig2 = plt.figure(figsize=(12, 9), dpi=90)
plt.rc('font', family='Times New Roman',size=11)

plt.subplot(311)
plt.plot(time, gps_speed_func(time), label='GPS')
plt.plot(time, V_rec, label='EKF')
#plt.plot(time, airspeed_func(time), label='Measured')
plt.grid()
plt.gca().legend()

# plt.subplot(312)
# plt.plot(time, W_rec_north, label='Wind_N')
# plt.plot(time, W_rec_east, label='Wind_E')
# plt.plot(time, W_rec_norm, label='Wind_Abs')
# plt.grid()
# plt.gca().legend()

plt.subplot(312)
plt.plot(time, Thrust_sigsq, label='Thrsut SigSqr')
plt.plot(time, Drag_sigsq, label='Drag SigSqr')
#plt.plot(time, W_rec_norm, label='Wind_Abs')
plt.grid()
plt.gca().legend()

plt.subplot(313)
plt.plot(time, P11_rec, label='P11')
plt.plot(time, P22_rec, label='P22')
plt.plot(time, P33_rec, label='P33')
plt.grid()
plt.gca().legend()
plt.tight_layout()
#plt.show()


