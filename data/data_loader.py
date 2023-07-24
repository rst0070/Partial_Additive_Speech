import torch.utils.data as data
import torch
import config
import torchaudio
import pandas as pd
import numpy as np
import os
import random
from data.musan import MusanNoise
from data.pas import Pas

class Vox1DevSet(data.Dataset):
    """
    Voxceleb1 development data without data augmentation
    """
    def __init__(self, sys_config=config.SysConfig(), exp_config=config.ExpConfig()):
        super(Vox1DevSet, self).__init__()
        
        self.path_root_dir = sys_config.path_vox1_dev_root
        self.utter_length = exp_config.train_sample
        self.anno_table = pd.read_csv(sys_config.path_vox1_dev_label, delim_whitespace = True)
        
    def __len__(self):
        return len(self.anno_table)
    
    def __getitem__(self, idx):
        path = os.path.join(self.path_root_dir, self.anno_table.iloc[idx, 2])
        
        # -------------------- get utterance and speaker number-------------------- #
        utter, _ = torchaudio.load(path)
        utter = torch.squeeze(utter)
        spk_id = int(self.anno_table.iloc[idx, 0]) - 1
        
        # -------------------- resize utterance -------------------- #
        utter_len = len(utter)
        if utter_len < self.utter_length:
            tmp = [utter for i in range(0, (self.utter_length // utter_len))]
            
            residue = self.utter_length % utter_len
            if residue > 0: tmp.append(utter[0:residue])
            
            utter = torch.cat(tmp, dim=0)
        
        # -------------------- select random segments from utterance ------------------------ #
        start_seg = random.randint(0, utter_len - self.utter_length)
        utter = utter[start_seg : start_seg + self.utter_length]        
        
        return utter, spk_id
    
class Vox1EnrollSet(data.Dataset):
    """
    Vox1EnrollSet is used to prepare embeddings for testing.  \\
    This is used for test in clean scenario. \\ 
    
    `__getitem__(idx)` method returns `(utter, spk_id)`  \\
    
    `spk_id:str`            - path of an utterance file, this can be used as identity for test. Thus, this is used as identity for enrollment utterance.
    `utter:torch.Tensor`    - Batched utterance of a speaker identified as `spk_id`. The shape is like [batch, utterance_length]
    
    """
    
    def __init__(self, sys_config=config.SysConfig(), exp_config=config.ExpConfig()) -> None:
        super(Vox1EnrollSet, self).__init__()
        self.path_root_dir = sys_config.path_vox1_test_root
        self.utter_length = exp_config.test_sample
        self.anno_table = pd.read_csv(sys_config.path_vox1_enroll_label, delim_whitespace = True)
        
    def __len__(self):
        return len(self.anno_table)
    
    def __getitem__(self, idx):
        
        # ------------------- speaker's id is a path of the utterance audio file ----------------- #
        spk_id = self.anno_table.iloc[idx, 0]
        
        # -------------------- get utterance-------------------- #
        path = os.path.join(self.path_root_dir, spk_id)
        utter, _ = torchaudio.load(path)
        utter = torch.squeeze(utter)
        
        # -------------------- resize utterance -------------------- #
        utter_len = len(utter)
        if utter_len < self.utter_length:
            tmp = [utter for i in range(0, (self.utter_length // utter_len))]
            
            residue = self.utter_length % utter_len
            if residue > 0: tmp.append(utter[0:residue])
            
            utter = torch.cat(tmp, dim=0)
        
        # -------------------- make batch -------------------------- #
        #seg_idx = np.arange(start=0, stop=len(utter) - self.utter_length, step=self.utter_length, dtype=int)
        seg_idx = torch.linspace(start=0, end=utter_len-self.utter_length, steps=30, dtype=int).tolist()
        tmp = []
        for start in seg_idx:
            tmp.append(utter[start : start + self.utter_length])
        #tmp.append(utter[-self.utter_length:])
        utter = torch.stack(tmp, dim = 0)        
        # shape of utter == [num_seg, self.utter_length]
        return utter, spk_id


def vox1_trial_list(sys_config=config.SysConfig()):
    """_summary_
    Trial list for test.
    
    Returns:
        list: list of pairs. pair: (utterance1 path, utterance2 path, is same).
        The path means relative path of the utterance file.
        This system uses the path as enrollment id.
    """
    anno_table = pd.read_csv(sys_config.path_vox1_test_label, delim_whitespace = True)
    result = []
    for idx in range(0, len(anno_table)):
        is_same = anno_table.iloc[idx, 0]
        spk_id1 = anno_table.iloc[idx, 1]
        spk_id2 = anno_table.iloc[idx, 2]
        result.append([spk_id1, spk_id2, is_same])
    return result


class PasTrainSet(data.Dataset):
    """_summary_
    Training dataset with PAS data augmentation method.
    For 1/4 chance, do not augmentation.
    """
    
    def __init__(self, sys_config=config.SysConfig(), exp_config=config.ExpConfig()):
        super(PasTrainSet, self).__init__()
        self.exp_config = exp_config
        self.vox1_dev = Vox1DevSet()
        self.pas = Pas(root_path=sys_config.path_musan_train)
        
    def __len__(self):
        return len(self.vox1_dev)
    
    def __getitem__(self, idx):
        utter, spk_id = self.vox1_dev.__getitem__(idx)
        
        if random.randint(0, 3) == 0:
            return utter, spk_id
        
        length = utter.shape[-1]
        
        utter_len = random.randint(self.exp_config.pas_min_utter, length)
        utter_s = random.randint(0, length - utter_len)
        utter = utter[utter_s : utter_s + utter_len]
        
        utter, _, _ = self.pas(x = utter, length = length)
        utter = torch.Tensor(utter)
        
        return utter, spk_id

class TanTrainSet(data.Dataset):
    """_summary_
    TAN( Traditional Additive Noise ) train set. \\
    This randomly adds Musan noise to Vox1DevSet for entire duration of each speech. \\
    This gives clean(1/4 chance) or noisy(3/4 chance) utterance. \\
    
    The probability is based on the number of categories of MUSAN.(categories are splitted in speech, noise, music)
    """
    
    def __init__(self, sys_config=config.SysConfig(), exp_config=config.ExpConfig()):
        super(TanTrainSet, self).__init__()
        self.vox1_dev = Vox1DevSet()
        self.musan = MusanNoise(root_path=sys_config.path_musan_train)
        
    def __len__(self):
        return len(self.vox1_dev)
    
    def __getitem__(self, idx):
        utter, spk_id = self.vox1_dev.__getitem__(idx)
        
        add_noise = random.randint(0, 3)
        
        if add_noise != 0: # noisy train data : 3/4 chance
            utter = torch.Tensor(self.musan(utter)) # MusanNoise randomly select category of noise
        
        return utter, spk_id


class NoisyEnrollSet(data.Dataset):
    """_summary_
    Dataset used for test in noisy environment. \\
    This reads audio files from the root_path parameter of initiator. \\
    """
    
    def __init__(self, root_path, sys_config=config.SysConfig(), exp_config=config.ExpConfig()) -> None:
        super(NoisyEnrollSet, self).__init__()
        self.path_root_dir = root_path
        self.utter_length = exp_config.test_sample
        self.anno_table = pd.read_csv(sys_config.path_vox1_enroll_label, delim_whitespace = True)
        
    def __len__(self):
        return len(self.anno_table)
    
    def __getitem__(self, idx):
        
        # ------------------- speaker's id is a path of the utterance audio file ----------------- #
        spk_id = self.anno_table.iloc[idx, 0]
        
        # -------------------- get utterance-------------------- #
        path = os.path.join(self.path_root_dir, spk_id)
        utter, _ = torchaudio.load(path)
        utter = torch.squeeze(utter)
        
        # -------------------- resize utterance -------------------- #
        utter_len = len(utter)
        if utter_len < self.utter_length:
            tmp = [utter for i in range(0, (self.utter_length // utter_len))]
            
            residue = self.utter_length % utter_len
            if residue > 0: tmp.append(utter[0:residue])
            
            utter = torch.cat(tmp, dim=0)
        
        # -------------------- make batch -------------------------- #
        seg_idx = torch.linspace(start=0, end=utter_len-self.utter_length, steps=30, dtype=int).tolist()
        tmp = []
        for start in seg_idx:
            tmp.append(utter[start : start + self.utter_length])
        
        utter = torch.stack(tmp, dim = 0)        
        # shape of utter == [num_seg, self.utter_length]
        return utter, spk_id