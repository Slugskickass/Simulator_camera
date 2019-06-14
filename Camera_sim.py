import numpy as np




def create_readnoise(frame, readnoise):
    read_frame = np.random.poisson(readnoise, frame)
    return read_frame

def create_FPmap(frame, gain, distrobution):
    FP_frame = np.random.normal(gain, distrobution, frame)
    return FP_frame

def create_shot_noise(illumination_frame):
    shot_noise_frame = np.zeros((np.shape(illumination_frame)))
    for X in range((illumination_frame.shape[0])):
        for Y in range((illumination_frame.shape[1])):
            shot_noise_frame[X, Y] = np.random.poisson((illumination_frame[X, Y])**0.5)
    return shot_noise_frame

def create_line_noise(frame, gain, distrobution):
    temp_frame = np.zeros(frame)
    temp_noise = np.random.normal(gain, distrobution, 2*frame[0])
    count =0
    for I in range(frame[0]):
        temp_frame[I, np.int16(0):np.int16(frame[0]/2)] = temp_noise[count]
        count = count +1
        temp_frame[I, np.int16(frame[0]/2):np.int16(frame[0])] = temp_noise[count]
        count = count + 1
    return temp_frame

def create_line_noise_fusion(frame, gain, distrobution):
    temp_frame = np.zeros(frame)
    temp_noise = np.random.normal(gain, distrobution, frame[0])
    count = 0
    for I in range(frame[0]):
        temp_frame[I, :] = temp_noise[count]
        count = count + 1
    return temp_frame
