import os
from data.musan import MusanNoise
from tqdm import tqdm
import soundfile as sf
from scipy.io import wavfile
import copy

def  vox1_enroll_label():
    result = ""

    for root, _, files in os.walk('/home/rst/dataset/voxceleb1/test'):
        for file in files:
            if '.wav' in file:
                result += os.path.join(root[33:len(root)], file) + "\n"

    file = open("label/vox1_enroll.csv", "w")
    file.write(result)
    file.close()

def split_musan_noise(musan_path, save_path):
    """_summary_
    splits musan dataset into train and test.
    Args:
        musan_path (_type_): path of musan dataset
        save_path (_type_): path to save splitted data
    """
    musan_sample_rate = 16000
    win_len = musan_sample_rate * 5
    win_step = musan_sample_rate * 3
    
    for folder, _, files in os.walk(musan_path):
        for file in tqdm(files):
            if '.wav' not in file:
                continue
            
            file_id = int(file[-8: -4])
            # split data into train phase and test phase
            phase = 'train' if file_id % 2 == 0 else 'test'
            
            file = os.path.join(folder, file)
            dest = file.replace(musan_path, os.path.join(save_path, phase)).replace('.', '')

            os.makedirs(dest, exist_ok=True)
            
            sr, noise = wavfile.read(file)
            num_file = (len(noise) - win_len) // win_step
            
            # small noise file
            if num_file == 0:
                wavfile.write(f"{dest}/all.wav", sr, noise)
                continue
            # large noise file
            for i in range(num_file):
                start = i * win_step
                end = start + win_len
                wavfile.write(f"{dest}/{i*3}_{(i*3) + 5}.wav", sr, noise[start : end])
                
def make_noisy_test_set(vox1_test_path:str, musan_path:str, save_path:str):
    """_summary_
    Generates test data which simulates noisy environments. \\
           
    Args:
        vox1_test_path (str): root folder of voxceleb1 test set
        musan_path (str): root folder of noise source for test
        save_path (str): where to save generated data
    """
    
    augment = MusanNoise(root_path=musan_path)
    snrs = [0, 5, 10, 15, 20]
    #snrs = [0, 100]
    categories = ['noise', 'speech', 'music']
    
    for folder, _, files in tqdm(os.walk(vox1_test_path)):
        for file in files:
            if '.wav' not in file:
                continue
            
            # read utterance from vox1 test file
            file = os.path.join(folder, file)
            utter, sr = sf.read(file)
            
            # save by category and snr
            for category in categories:
                for snr in snrs:
                    noisy = augment(copy.deepcopy(utter), category=category, snr=snr)
                    dest = file.replace(vox1_test_path, os.path.join(save_path, f'{category}_{snr}'))
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    sf.write(dest, noisy, sr)
            
            
    
if __name__ == '__main__':
    """_summary_
    prepareing data for train and test. \\
    Set path variables below.
    """
    print("pre processing")
    path_vox1_test = '' #ex. '/home/rst/dataset/voxceleb1/test'
    path_musan = '' # ex. /home/rst/dataset/musan
    path_splitted_musan = '' # ex. '/home/rst/dataset/musan_split'
    path_noise_test = '' # ex. '/home/rst/dataset/noise_test'
    
    split_musan_noise(musan_path=path_musan, save_path=path_splitted_musan)
    
    path_splitted_musan = os.path.join(path_splitted_musan, 'test')
    make_noisy_test_set(vox1_test_path=path_vox1_test, musan_path=path_splitted_musan, save_path=path_noise_test)