import Camera_sim as cs
import numpy as np
import matplotlib.pylab as plt
from PIL import Image


##
## Variables

X_size = 1152
Y_size = 1152
#number_of_photons = 10
offset = 100
gain = 0.24
readnoise = 0.7 / gain
start = 0
stop = 15


# The QE_gain variation if the basis of the fixed pattern noise
Quantum_efficiency = 0.7
QE_gain_variation = 0.003 / gain


fixed_pattern_noise = 0.001

list_excitations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30,
                    40, 50, 60, 70, 80, 90, 100, 200, 300,
                    400, 500, 600, 700, 800, 900, 1000, 2000,
                    3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

# Create a map of the variation in QE, this is the same for each frame
frame = np.zeros((X_size, Y_size))
fixed_pattern_map = cs.create_FPmap(np.shape(frame), Quantum_efficiency, QE_gain_variation)
line_frame = cs.create_line_noise_fusion(np.shape(frame), gain, fixed_pattern_noise)

noise_data = np.zeros(np.size(list_excitations))
mean_data = np.zeros(np.size(list_excitations))
count = 0

for number_of_photons in list_excitations:
    print(number_of_photons)
    ## Build illumination pattern
    illumination_frame = np.ones((X_size, Y_size)) * number_of_photons

    # Create the read noise map, this is different for each frame.
    read_noise_frame = cs.create_readnoise(np.shape(frame), readnoise)

    # create shot noise, this is different for each frame.
    shot_noise_frame = cs.create_shot_noise(illumination_frame)

    # The total number of electrons in the well of the camera (including the variation in QE)
    total_count_frame = shot_noise_frame + read_noise_frame + (illumination_frame * fixed_pattern_map)

    # The gain system what happens as the electrons go through the amplifiers
    # line_frame = cs.create_line_noise_fusion(np.shape(frame), gain, fixed_pattern_noise)

    # take the number of photons and multiply it by the
    frame_out_counts = total_count_frame * line_frame

    #
    frame_out_counts = frame_out_counts + offset

    #
    frame_out_counts = np.int16(frame_out_counts)

    # Calculate the PTC
    noise_data[count] = np.std(np.reshape(frame_out_counts, [1, X_size * Y_size]))
    mean_data[count] = np.mean(np.reshape(frame_out_counts, [1, X_size * Y_size]))-offset
    count = count + 1


fitted = (np.polyfit(mean_data[0:stop], noise_data[0:stop]**2, 1))

plt.loglog(mean_data, noise_data)
plt.loglog(mean_data[start:stop], noise_data[start:stop]**2)
x = range(start, stop)
y = fitted[1] + x * fitted[0]
print(fitted[0])
plt.loglog(x, y)
plt.show()
plt.imshow(frame_out_counts)
plt.show()
