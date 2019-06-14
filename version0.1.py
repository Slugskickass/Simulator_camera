import Camera_sim as cs
import numpy as np
import matplotlib.pylab as plt
from PIL import Image


##
## Variables

X_size = 1152
Y_size = 1152
number_of_photons = 10
offset = 100
gain = 0.24
readnoise = 0.7 / gain
Quantum_efficiency = 0.7
QE_gain_variation = 0.3 / gain
fixed_pattern_noise = 0.1


## Real data comparison
im = Image.open('temp.tif')
np_im = np.array(im)

## Build illumination pattern
frame = np.zeros((X_size, Y_size))
illumination_frame = np.ones((X_size, Y_size)) * number_of_photons


# Create a map of the variation in QE, this is the same for each frame
fixed_pattern_map = cs.create_FPmap(np.shape(frame), Quantum_efficiency, QE_gain_variation)

# Create the read noise map, this is different for each frame.
frame_out = cs.create_readnoise(np.shape(frame), readnoise)

# create shot noise, this is different for each frame.
illumination_frame = illumination_frame * fixed_pattern_map
shot_noise_frame = cs.create_shot_noise(illumination_frame)



near_final_frame = shot_noise_frame + frame_out + illumination_frame





frame_out_counts = near_final_frame * gain

#line artifact here
line_frame = cs.create_line_noise_fusion(np.shape(frame), 1, fixed_pattern_noise)

frame_out_counts = frame_out_counts * line_frame
frame_out_counts = frame_out_counts + offset
frame_out_counts = np.int16(frame_out_counts)



plt.subplot(2, 2, 1)
plt.imshow(frame_out_counts)
plt.subplot(2, 2, 2)
plt.imshow(np_im)
plt.subplot(2, 2, 3)
plt.hist(np.reshape(frame_out_counts, [Y_size * X_size, 1]), range=[80, 125])
plt.subplot(2, 2, 4)
plt.hist(np.reshape(im, [np.size(np_im), 1]), range=[80, 125])
plt.show()