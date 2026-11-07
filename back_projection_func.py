import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import stats

def filter_on_peak(size, peak_index, alpha=4):
    x = np.arange(size) - peak_index
    sinc_filter = np.sinc(x / alpha)
    return sinc_filter

def back_projection(data, prf, f0, v_ground, sr0, rg_sampling):
    c = 3e8
    az_size, rg_size = data.shape
    wavelength = c / f0
    kr = 2 * np.pi / wavelength
    epsilon = 1e-6

    image_grid = np.zeros((az_size, rg_size), dtype=complex)  
    peak_range_idx = np.bincount(np.argmax(np.abs(data), axis=1)).argmax() 
    sinc_filter_window = filter_on_peak(rg_size, peak_range_idx)
    print(peak_range_idx)
    middle_azimuth_idx = (az_size // 2)+1 
    ##### MAIN LOOP ####
    for az_idx in range(az_size):
        range_vector = data[az_idx, :] * sinc_filter_window
        range_vector *= np.max(np.abs(data[az_idx, :])) / np.max(np.abs(range_vector)) #Amp. compensation
        azimuth_time = az_idx / prf 

        d_az = v_ground * azimuth_time #Azi displace.

        for y in range(rg_size):
            range_distance = sr0 + y * rg_sampling
            slant_range = np.sqrt(d_az**2 + range_distance**2)
            # Note: Since d_az is very small that makes range and slant range almost equal. 
            # Increasing v_ground or azimuth_time will make impact, To verify i have added epsilon
            #slant_range = np.sqrt(d_az**2 + range_distance**2) +epsilon
            rg_bin_float = (slant_range - sr0) / rg_sampling
            rg_bin = int(np.floor(rg_bin_float))
            weight = rg_bin_float - rg_bin
            if 0 <= rg_bin < rg_size - 1:
                interpolated_value = (
                    (1 - weight) * range_vector[rg_bin] + weight * range_vector[rg_bin + 1]
                )
                
                phase_correction = np.exp(1j * kr * (slant_range - range_distance))
                contribution = interpolated_value * phase_correction
                """print("Contribution:", contribution)
                print("Range Vector Value:", range_vector[rg_bin])
                print("Difference (slant_range - range_distance):", slant_range - range_distance)
                print("Phase of Contribution:", np.angle(contribution))
                print("Phase of Range Vector Value:", np.angle(range_vector[rg_bin]))
                print("Phase Difference:", np.angle(contribution) - np.angle(range_vector[rg_bin]))"""                
                image_grid[middle_azimuth_idx, y] += contribution

    """plt.plot(np.abs(data[:, peak_range_idx]))
    plt.plot(np.abs(image_grid[middle_azimuth_idx, :])) 
    plt.xlabel('Range Index')
    plt.ylabel('Magnitude')
    plt.title('Accumulated Magnitude at Middle Azimuth Row')
    plt.grid(True)
    plt.show()"""

    return image_grid
