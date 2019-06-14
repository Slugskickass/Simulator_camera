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
Quantum_efficiency = 0.7
QE_gain_variation = 0.3 / gain
fixed_pattern_noise = 0.1

list_excitations =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Create a map of the variation in QE, this is the same for each frame
frame = np.zeros((X_size, Y_size))
fixed_pattern_map = cs.create_FPmap(np.shape(frame), Quantum_efficiency, QE_gain_variation)
line_frame = cs.create_line_noise_fusion(np.shape(frame), 1, fixed_pattern_noise)

noise_data = np.zeros(np.size(list_excitations))
mean_data = np.zeros(np.size(list_excitations))
count = 0
for number_of_photons in list_excitations:
    ## Build illumination pattern
    illumination_frame = np.ones((X_size, Y_size)) * number_of_photons
    print(number_of_photons)
    # Create the read noise map, this is different for each frame.
    frame_out = cs.create_readnoise(np.shape(frame), readnoise)

    # create shot noise, this is different for each frame.
#    illumination_frame = illumination_frame * fixed_pattern_map
    shot_noise_frame = cs.create_shot_noise(illumination_frame)
    #near_final_frame = shot_noise_frame  + illumination_frame
    near_final_frame = shot_noise_frame + frame_out + illumination_frame



    #frame_out_counts = np.int32(near_final_frame)
    noise_data[count] = np.std(np.reshape(near_final_frame, [X_size * Y_size, 1]))
    mean_data[count] = np.mean(np.reshape(near_final_frame, [X_size * Y_size, 1]))
    count = count + 1


plt.log(mean_data, noise_data)
print(np.shape(mean_data))
plt.show()
