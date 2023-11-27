import os
import csv
import numpy as np
import librosa
from tqdm import tqdm
import subprocess

make_wav = False
make_csv = True

def calculate_features(filename):
    y, sr = librosa.load(filename)

    # Time in sec of the file
    duration = librosa.get_duration(y=y, sr=sr)

    # Chroma feature
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft)
    chroma_stft_var = np.var(chroma_stft)

    # RMS energy
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_var = np.var(spectral_centroid)

    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_var = np.var(spectral_bandwidth)

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_var = np.var(rolloff)

    # Zero crossing rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)
    zero_crossing_rate_var = np.var(zero_crossing_rate)

    # Harmonic-to-percussive ratio
    harmonic, percussive = librosa.effects.hpss(y)
    harmony_mean = np.mean(harmonic)
    harmony_var = np.var(harmonic)
    perceptr_mean = np.mean(percussive)
    perceptr_var = np.var(percussive)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # MFCCs (13 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_var = np.var(mfccs, axis=1)

    # Combine all the features into a single list
    features = [
        duration, 
        chroma_stft_mean, chroma_stft_var, rms_mean, rms_var,
        spectral_centroid_mean, spectral_centroid_var,
        spectral_bandwidth_mean, spectral_bandwidth_var,
        rolloff_mean, rolloff_var,
        zero_crossing_rate_mean, zero_crossing_rate_var,
        harmony_mean, harmony_var, perceptr_mean, perceptr_var,
        tempo
    ]

    for i in range(20):
        features.append(mfccs_mean[i])
        features.append(mfccs_var[i])

    return np.array(features)

def count_files(directory):
    total_files = 0

    for root, dirs, files in os.walk(directory):
        total_files += len(files)

    return total_files

def main(writer=None):
    for foldername, subfolders, filenames in os.walk(in_foldername):
            new_foldername = out_foldername + foldername.replace(in_foldername, '')
            for filename in filenames:
                if filename.endswith('.mid'):
                    os.makedirs(new_foldername, exist_ok=True)
                    input_file_name = os.path.join(foldername, filename)
                    output_file_name = os.path.join(new_foldername, filename)
                    output_file_name = ".".join(output_file_name.split('.')[:-1]) + '.wav'

                    if make_wav:
                        subprocess.run(['fluidsynth', '-ni', '-g', '3', 'IK_Berlin_Grand_Piano.sf2', input_file_name, '-F', output_file_name])
                        # print('Converted', output_file_name)

                    if make_csv:
                        features = calculate_features(output_file_name)
                        writer.writerow([output_file_name, ] + list(features))

                    tqdm_object.update(1)

subfolder = '/Classical/Classical'
in_foldername = './adl-piano-midi' + subfolder
out_foldername = './adl-piano-wav' + subfolder
csv_path = './csv' + subfolder + '/data.csv'
tqdm_object = tqdm(total=count_files(in_foldername))

if make_csv:
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([
            'filename', 'duration', 'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
            'spectral_centroid_mean', 'spectral_centroid_var',
            'spectral_bandwidth_mean', 'spectral_bandwidth_var',
            'rolloff_mean', 'rolloff_var',
            'zero_crossing_rate_mean', 'zero_crossing_rate_var',
            'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var',
            'tempo',
            'mfccs_mean1', 'mfccs_var1', 'mfccs_mean2', 'mfccs_var2',
            'mfccs_mean3', 'mfccs_var3', 'mfccs_mean4', 'mfccs_var4',
            'mfccs_mean5', 'mfccs_var5', 'mfccs_mean6', 'mfccs_var6',
            'mfccs_mean7', 'mfccs_var7', 'mfccs_mean8', 'mfccs_var8',
            'mfccs_mean9', 'mfccs_var9', 'mfccs_mean10', 'mfccs_var10',
            'mfccs_mean11', 'mfccs_var11', 'mfccs_mean12', 'mfccs_var12',
            'mfccs_mean13', 'mfccs_var13', 'mfccs_mean14', 'mfccs_var14',
            'mfccs_mean15', 'mfccs_var15', 'mfccs_mean16', 'mfccs_var16',
            'mfccs_mean17', 'mfccs_var17', 'mfccs_mean18', 'mfccs_var18',
            'mfccs_mean19', 'mfccs_var19', 'mfccs_mean20', 'mfccs_var20'
        ])
        main(writer)
else:
    main()

print('Done')
