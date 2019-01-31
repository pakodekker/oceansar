# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 17:16:19 2019

@author: lyh
"""
import os
import numpy as np
from matplotlib import pyplot as plt

inc_s = [12, 6]
azimuth_s = [0, 45, 90, 135, 180, 225, 270, 315]
wave_scale = [60, 25, 12.5]
Amg = True
path = r'D:\research\TU Delft\Data\OceanSAR'
if Amg:
    data = np.load(os.path.join(path, 'Angular_velocity_data_unfocus.npy'))
    data_p = np.load(os.path.join(path, 'Phase_velocity_data_unfocus.npy'))
    data_d_c = np.load(os.path.join(path, 'Doppler_coherence_data_unfocus.npy'))
else:
    data = np.load(os.path.join(path, 'Angular_velocity_data.npy'))
    data_p = np.load(os.path.join(path, 'Phase_velocity_data.npy'))
    data_d_c = np.load(os.path.join(path, 'Doppler_coherence_data.npy'))

angu_v = data[0]
az_num = data[3]
pha_v = data_p[0]
Doppler_av = data_d_c[0]
Coherence = data_d_c[1]
num = data_d_c[2]

data_no = 300
# type the parameter you want to see
inc = 12
azimuth = 270
wave_number = 60

Num_b = -3#-3

value = np.zeros((np.size(angu_v[0,:,0,0,0]), np.size(angu_v[0,0,:,0,0]),
                          np.size(angu_v[0,0,0,:,0]), np.size(angu_v[0,0,0,0,:])), dtype=np.complex)
ts = np.zeros((data_no,1),dtype=np.complex)

value_p = np.zeros((np.size(pha_v[0,:,0,0,0]), np.size(pha_v[0,0,:,0,0]),
                          np.size(pha_v[0,0,0,:,0]), np.size(pha_v[0,0,0,0,:])), dtype=np.complex)

value_std = np.zeros((np.size(angu_v[0,:,0,0,0]), np.size(angu_v[0,0,:,0,0]),
                          np.size(angu_v[0,0,0,:,0]), np.size(angu_v[0,0,0,0,:])), dtype=np.complex)

value_p_std = np.zeros((np.size(pha_v[0,:,0,0,0]), np.size(pha_v[0,0,:,0,0]),
                          np.size(pha_v[0,0,0,:,0]), np.size(pha_v[0,0,0,0,:])), dtype=np.complex)

ts_p = np.zeros((data_no,1),dtype=np.complex)

#value_dop = np.zeros((data_no, np.size(Doppler_av[0,:,0,0]),
#                          np.size(Doppler_av[0,0,:,0]), np.size(Doppler_av[0,0,0,:])), dtype=np.complex)
ts_dop = np.zeros((data_no,1),dtype=np.complex)

#value_coh = np.zeros((data_no, np.size(Doppler_av[0,:,0,0]),
#                          np.size(Doppler_av[0,0,:,0]), np.size(Doppler_av[0,0,0,:])), dtype=np.complex)
ts_coh = np.zeros((data_no,1),dtype=np.complex)

if azimuth < 100:
    ref_a_v = -np.sqrt(9.8*2*np.pi/wave_number) * np.ones((np.size(az_num),),dtype=np.complex)
    ref_p_v = -np.sqrt(9.8*2*np.pi/wave_number)/(2*np.pi/wave_number)* np.ones((np.size(az_num),),dtype=np.complex)
    
    ref_a = -np.sqrt(9.8*2*np.pi/wave_number) * np.ones((data_no,1),dtype=np.complex)
    ref_p = -np.sqrt(9.8*2*np.pi/wave_number)/(2*np.pi/wave_number)* np.ones((data_no,1),dtype=np.complex)
else:
    ref_a_v = np.sqrt(9.8*2*np.pi/wave_number) * np.ones((np.size(az_num),),dtype=np.complex)
    ref_p_v = np.sqrt(9.8*2*np.pi/wave_number)/(2*np.pi/wave_number)* np.ones((np.size(az_num),),dtype=np.complex)

    ref_a = np.sqrt(9.8*2*np.pi/wave_number) * np.ones((data_no,1),dtype=np.complex)
    ref_p = np.sqrt(9.8*2*np.pi/wave_number)/(2*np.pi/wave_number)* np.ones((data_no,1),dtype=np.complex)

# for angular velocity and phase velocity
for ind_inc in range(np.size(angu_v[0,:,0,0,0])): 
    for ind_azimuth in range(np.size(angu_v[0,0,:,0,0])):  
         if (inc==inc_s[ind_inc] and azimuth==azimuth_s[ind_azimuth]):
             ts_dop = np.mean(Doppler_av[:, ind_inc, ind_azimuth, :],axis=1)                      
             ts_coh = np.mean(Coherence[:, ind_inc, ind_azimuth, :],axis=1)   
         for ind_wavelength in range(np.size(angu_v[0,0,0,:,0])):
             for az in range(np.size(angu_v[0,0,0,0,:])):
                    value [ind_inc, ind_azimuth, ind_wavelength, az] = np.mean(angu_v[:, ind_inc, ind_azimuth, ind_wavelength, az])
                    value_p [ind_inc, ind_azimuth, ind_wavelength, az] = np.mean(pha_v[:, ind_inc, ind_azimuth, ind_wavelength, az]) 
                    value_std [ind_inc, ind_azimuth, ind_wavelength, az] = np.std(angu_v[:, ind_inc, ind_azimuth, ind_wavelength, az])
                    value_p_std [ind_inc, ind_azimuth, ind_wavelength, az] = np.std(pha_v[:, ind_inc, ind_azimuth, ind_wavelength, az]) 
                    
             if (inc==inc_s[ind_inc] and azimuth==azimuth_s[ind_azimuth] and wave_number==wave_scale[ind_wavelength]):
                 ts = np.mean(angu_v[:, ind_inc, ind_azimuth, ind_wavelength, Num_b:],axis=1)                      
                 ts_p = np.mean(pha_v[:, ind_inc, ind_azimuth, ind_wavelength, Num_b:],axis=1)

av = np.mean(value[inc_s.index(inc),azimuth_s.index(azimuth),wave_scale.index(wave_number),Num_b:])
av_std = np.std(value[inc_s.index(inc),azimuth_s.index(azimuth),wave_scale.index(wave_number),Num_b:])
sv = np.mean(ts)
sv_std = np.std(ts)
av_p = np.mean(value_p[inc_s.index(inc),azimuth_s.index(azimuth),wave_scale.index(wave_number),Num_b:])
av_p_std = np.std(value_p[inc_s.index(inc),azimuth_s.index(azimuth),wave_scale.index(wave_number),Num_b:])
sv_p = np.std(ts_p)
sv_p_std = np.std(ts_p)
if Amg:
    plt.figure()
    #plt.errorbar(az_num, value[inc_s.index(inc),azimuth_s.index(azimuth),
    #                       wave_scale.index(wave_number),:],yerr=value_std[inc_s.index(inc),azimuth_s.index(azimuth),wave_scale.index(wave_number),:], fmt='-o')
    plt.errorbar(az_num, value[inc_s.index(inc),azimuth_s.index(azimuth),
                           wave_scale.index(wave_number),:],yerr=np.abs(ref_a_v-value[inc_s.index(inc),azimuth_s.index(azimuth),wave_scale.index(wave_number),:]), fmt='-o',ecolor='r',color='b')
    #plt.plot(az_num, ref_a_v,linestyle='--')
    plt.xlabel("Azimuth interval [Pixel]")
    plt.ylabel("Angular velocity (rad/s)") 
    plt.ylim(ymin=-20, ymax=10)
    plt.legend(["Estimated value","Reference value"])
    plt.text(200, -10, "Mean value = %.2f rad/s, Std = %.2f rad/s"%(av, av_std), fontsize=10)
    plt.title("Incidence angle = %d deg, Azimuth angle = %d deg, Wave length = %.1f m" % (inc, azimuth, wave_number))
    plt.show()
    
    plt.figure()
    plt.plot(ts)
    plt.plot(ref_a,linestyle='--')
    plt.xlabel("No.")
    plt.ylabel("Angular velocity (rad/s)") 
    plt.ylim(ymin=-10, ymax=5)
    plt.legend(["Estimated value","Reference value"])
    plt.text(200, -5, "Mean value = %.2f rad/s, Std = %.2f rad/s"%(sv, sv_std), fontsize=10)
    plt.title("Incidence angle = %d deg, Azimuth angle = %d deg, Wave length = %.1f m" % (inc, azimuth, wave_number))
    print(np.mean(value[inc_s.index(inc),azimuth_s.index(azimuth),wave_scale.index(wave_number),Num_b:]))
    print(np.std(ts))        
    
    plt.figure()
    #plt.errorbar(az_num, value_p[inc_s.index(inc),azimuth_s.index(azimuth),
    #                       wave_scale.index(wave_number),:],yerr=value_p_std[inc_s.index(inc),azimuth_s.index(azimuth),wave_scale.index(wave_number),:], fmt='-o')
    plt.errorbar(az_num, value_p[inc_s.index(inc),azimuth_s.index(azimuth),
                           wave_scale.index(wave_number),:],yerr=np.abs(ref_p_v-value_p[inc_s.index(inc),azimuth_s.index(azimuth),wave_scale.index(wave_number),:]), fmt='-o',ecolor='r',color='b')
    #plt.plot(az_num, ref_p_v,linestyle='--')
    plt.xlabel("Azimuth interval [Pixel]")
    plt.ylabel("Phase velocity (m/s)") 
    plt.ylim(ymin=-80, ymax=80)
    plt.text(200, -60, "Mean value = %.2f m/s, Std = %.2f m/s"%(av_p, av_p_std), fontsize=10)
    plt.legend(["Estimated value","Reference value"])
    plt.title("Incidence angle = %d deg, Azimuth angle = %d deg, Wave length = %.1f m" % (inc, azimuth, wave_number))
    
    plt.figure()
    plt.plot(ts_p)
    plt.plot(ref_p,linestyle='--')
    plt.xlabel("No.")
    plt.ylabel("Phase velocity (m/s)") 
    plt.legend(["Estimated value","Reference value"])
    plt.ylim(ymin=-20, ymax=10)
    plt.text(50, -14, "Mean value = %.2f m/s, Std = %.2f m/s"%(sv_p, sv_p_std), fontsize=10)
    plt.title("Incidence angle = %d deg, Azimuth angle = %d deg, Wave length = %.1f m" % (inc, azimuth, wave_number))
    print(np.mean(value_p[inc_s.index(inc),azimuth_s.index(azimuth),wave_scale.index(wave_number),Num_b:]))
    print(np.std(ts_p))
else:
    plt.figure()
    plt.plot(az_num, value[inc_s.index(inc),azimuth_s.index(azimuth),wave_scale.index(wave_number),:])
    plt.plot(az_num, ref_a_v,linestyle='--')
    plt.xlabel("Azimuth interval [Pixel]")
    plt.ylabel("Angular velocity (rad/s)") 
    plt.ylim(ymin=-15, ymax=5)
    plt.legend(["Estimated value","Reference value"])
    plt.text(100, -10, "Mean value = %.2f rad/s, Std = %.2f rad/s"%(av, av_std), fontsize=10)
    plt.title("Incidence angle = %d deg, Azimuth angle = %d deg, Wave length = %.1f m" % (inc, azimuth, wave_number))
    plt.show()
    
    plt.figure()
    plt.plot(ts)
    plt.plot(ref_a,linestyle='--')
    plt.xlabel("No.")
    plt.ylabel("Angular velocity (rad/s)") 
    plt.ylim(ymin=-10, ymax=5)
    plt.legend(["Estimated value","Reference value"])
    plt.text(150, -5, "Mean value = %.2f rad/s, Std = %.2f rad/s"%(sv, sv_std), fontsize=10)
    plt.title("Incidence angle = %d deg, Azimuth angle = %d deg, Wave length = %.1f m" % (inc, azimuth, wave_number))
    print(np.mean(value[inc_s.index(inc),azimuth_s.index(azimuth),wave_scale.index(wave_number),Num_b:]))
    print(np.std(ts))
    
    
    
    plt.figure()
    plt.plot(az_num, value_p[inc_s.index(inc),azimuth_s.index(azimuth),wave_scale.index(wave_number),:])
    plt.plot(az_num, ref_p_v,linestyle='--')
    plt.xlabel("Azimuth interval [Pixel]")
    plt.ylabel("Phase velocity (m/s)") 
    plt.ylim(ymin=-30, ymax=10)
    plt.text(100, -20, "Mean value = %.2f m/s, Std = %.2f m/s"%(av_p, av_p_std), fontsize=10)
    plt.legend(["Estimated value","Reference value"])
    plt.title("Incidence angle = %d deg, Azimuth angle = %d deg, Wave length = %.1f m" % (inc, azimuth, wave_number))
    
    plt.figure()
    plt.plot(ts_p)
    plt.plot(ref_p,linestyle='--')
    plt.xlabel("No.")
    plt.ylabel("Phase velocity (m/s)") 
    plt.legend(["Estimated value","Reference value"])
    plt.ylim(ymin=-20, ymax=10)
    plt.text(50, -14, "Mean value = %.2f m/s, Std = %.2f m/s"%(sv_p, sv_p_std), fontsize=10)
    plt.title("Incidence angle = %d deg, Azimuth angle = %d deg, Wave length = %.1f m" % (inc, azimuth, wave_number))
    print(np.mean(value_p[inc_s.index(inc),azimuth_s.index(azimuth),wave_scale.index(wave_number),Num_b:]))
    print(np.std(ts_p))

m_dop = np.mean(ts_dop)
m_coh = np.mean(ts_coh)
s_dop = np.std(ts_dop)
s_coh = np.std(ts_coh)

plt.figure()
plt.plot(ts_dop)
plt.xlabel("No.")
plt.ylabel("Averaged doppler centroid (Hz)") 
plt.ylim(ymin=-60, ymax=-20)
plt.text(150, -30, "Mean = %.2f Hz, Std = %.2f Hz"%(m_dop, s_dop), fontsize=10)
plt.title("Incidence angle = %d deg, Azimuth angle = %d deg" % (inc, azimuth))

plt.figure()
plt.plot(ts_coh)
plt.xlabel("No.")
plt.ylabel("Averaged coherence") 
plt.ylim(ymin=0.5, ymax=1)
plt.text(150, 0.7, "Mean = %.2f, Std = %.5f"%(m_coh, s_coh), fontsize=10)
plt.title("Incidence angle = %d deg, Azimuth angle = %d deg" % (inc, azimuth))
