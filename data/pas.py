import torch
import numpy as np
import os
import random
import soundfile as sf

class Pas:
    
    def __init__(self, root_path:str):
        """_summary_
        Args:
            root_path (str): root path of musan data
        """
        
        self.categories = ['noise', 'music', 'speech']
        
        self.files = {
            'noise' : [], 'music' : [], 'speech' : []
        }
        
        self.num_noise = {
            'noise' : (1, 1), 'music' : (1, 1), 'speech' : (3, 6)
        }
        
        self.snr_min = 0
        self.snr_max = 20
        
        
        for dir_path, _, files in os.walk(root_path):
            category = None
            for c in self.categories:
                if c in dir_path:
                    category = c
                    break
            if category is not None:
                for file in files:
                    if '.wav' in file:
                        self.files[category].append(os.path.join(dir_path, file))                
        
    def __call__(self, x, length,category=None, snr=None):
        """_summary_

        Args:
            x (any): waveform of utterance
            length : length of noise
            category (str, optional): category of noise('noise', 'music', 'speech'). Defaults to None for random.
            snr (_type_, optional): . Defaults to None for random.
            
        음성이 있는 부분의 위치도 반환한다.
        """
        
        if type(x) == torch.Tensor:
            x = x.numpy()
        
        x_size = x.shape[-1]
        x_dB = self.calculate_decibel(x)
        
        # ----------------------- None -> random ----------------------- #
        if category is None:
            category = random.choice(self.categories)
        
        if snr is None:
            snr = random.uniform(self.snr_min, self.snr_max)
        
        
        # ----------------------- select noises ----------------------- #
        
        noise_files = random.sample(# Bubble noise needs several speech noises
            self.files[category],
            random.randint(self.num_noise[category][0], self.num_noise[category][1])
        )
        
        noises = []
        for noise in noise_files:
            
            noise, _ = sf.read(noise)
            # random crop
            noise_size = noise.shape[0]
            if noise_size < length:
                shortage = length - noise_size + 1
                noise = np.pad(
                    noise, (0, shortage), 'wrap'
                )
                noise_size = noise.shape[0]
            
            index = random.randint(0, noise_size - length)
            noises.append(noise[index:index + length])


		# ----------------------- inject noise ----------------------- #
        idx = 0
        if len(noises) != 0:
            noise = np.mean(noises, axis=0)
			# calculate dB
            noise_dB = self.calculate_decibel(noise)
			# append
            p = (x_dB - noise_dB - snr)
	
            noise = np.sqrt(10 ** (p / 10)) * noise
            
            idx = random.randint(0, length - x_size)
            noise[idx : idx + x_size] = noise[idx : idx + x_size] + x
            
            x = noise

        return x, idx, idx + x_size-1
        
    
    def calculate_decibel(self, x:torch.Tensor):
        assert 0 <= np.mean(x ** 2) + 1e-4
        return 10 * np.log10(np.mean(x ** 2) + 1e-4)