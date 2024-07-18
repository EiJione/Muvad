import numpy as np

def mu_law_compression(frame, mu):
    # Convert frame to a numpy array and normalize it to the range [-1, 1]
    x = frame.astype(np.float64) / 32768.0
    # Apply mu-law compression
    compressed_sample = np.sign(x) * (np.log(1 + mu * np.abs(x)) / np.log(1 + mu))
    # Calculate the energy of the compressed samples
    energy = np.mean(compressed_sample ** 2)
    return energy

def mu_vad(audio_samples, sample_rate, mu):
    frame_length = (10 * sample_rate) // 1000  # 10ms analysis window

    num_frames = len(audio_samples) // frame_length
    print(frame_length,num_frames)
    energies = np.zeros(num_frames)
    # Calculate energy for the first frame and set it as the initial threshold
    first_frame = audio_samples[:frame_length]
    e_int = mu_law_compression(first_frame, mu)
    rate = np.exp(-10 * e_int)
    itl_mu = (1 + rate) * e_int
    print(f"Initial Eint: {e_int}, Initial ITLmu: {itl_mu}")
    label =[0]
    # Process from the second frame onwards
    for i in range(1, num_frames):
        start_idx = i * frame_length
        end_idx = start_idx + frame_length
        frame = audio_samples[start_idx:end_idx]
        energies[i] = mu_law_compression(frame, mu)
        print(energies[i])
        print("index:",i)
        # If window energy exceeds ITLmu, voice is detected
        if energies[i] > itl_mu:
            label.append(1)
        else:
            label.append(0)
        # Update Eint to the average energy of all processed windows
        e_int = np.mean(energies[:i + 1])
        # Calculate Rate and update ITLmu
        rate = np.exp(-10 * e_int)
        itl_mu = (1 + rate) * e_int
    return label

# # Example usage:
# import numpy as np
#
# # Simulate some audio data as int16 samples
# audio_samples = np.random.randint(-32768, 32768, 44100)  # 1 second of random samples at 44.1kHz
# sample_rate = 44100
# mu = 255
#
# # Call the voice activity detection function
# voice_present = mu_vad(audio_samples, sample_rate, mu)
# print("Voice present:", voice_present)
