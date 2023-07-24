import config
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from data.feature import FeatureExtractor
import torch.utils.data as data
from data.data_loader import PasTrainSet, TanTrainSet, Vox1EnrollSet, vox1_trial_list, NoisyEnrollSet
from trainer.metric import calculate_EER
from loss.aam_softmax import AAMSoftmaxLoss
import wandb

class Trainer:
    """
    Training : uses NoisyTrainSet
    Testing  : uses whole dataset
    """
    def __init__(self, model:torch.nn.Module, optimizer, sys_config=config.SysConfig(), exp_config=config.ExpConfig()):
                
        self.sys_config = sys_config
        self.exp_config = exp_config
        
        
        ###
        ###    Model and optimizer
        ###
        self.preprocessing = FeatureExtractor().to(sys_config.device)
        self.model = model.to(sys_config.device)
        
        self.loss_fn = AAMSoftmaxLoss(
            embedding_size=exp_config.embedding_size,
            num_class=exp_config.output_size,
            scale_constant=exp_config.loss_scale,
            margin=exp_config.loss_margin
        ).to(sys_config.device)
        
        self.optimizer = optimizer
        self.optimizer.add_param_group({
            'params' : self.loss_fn.parameters()
        })
        
        ###
        ###    Select Training dataset between TAN and PAS
        ###
        train_dataset = None
        if exp_config.use_pas:
            train_dataset = PasTrainSet()
        else:
            train_dataset = TanTrainSet()
        
        self.train_loader = data.DataLoader(
            dataset=train_dataset,
            batch_size=exp_config.batch_size,
            shuffle=True,
            num_workers=sys_config.num_workers,
            pin_memory=True
        )
        '''train data loader.'''
        
        
        ###
        ###    Setting datasets for enrollment and record table for performance
        ###
        self.best_eer = {}
        self.enroll_datasets = {}
        '''clean and noisy enrollment datasets. key = `noise-type_snr` or `clean`'''
        
        for key in sys_config.path_noisy_tests.keys():
            root_path = sys_config.path_noisy_tests[key]
            self.enroll_datasets[key] = NoisyEnrollSet(root_path=root_path)
            
            self.best_eer[key] = 100.
            
        self.enroll_datasets['clean'] = Vox1EnrollSet()
        self.best_eer['clean']    = 100.
        
        
        self.trial_list = vox1_trial_list()
        '''trial pairs'''
        self.enrolled = {}
        '''enrolled pairs of speaker id and embedding'''
        
        
    
    def train(self):
        
        self.model.train()
        self.loss_fn.train()
        
        itered = 0
        loss_sum = 0
        
        pbar = tqdm(self.train_loader)
        for x, y in pbar:
            
            itered += 1
            
            self.optimizer.zero_grad()
            
            x = x.to(self.sys_config.device)
            y = y.to(self.sys_config.device)
            
            x = self.preprocessing(x)
            x = self.model(x)
            loss = self.loss_fn(x, y)
            
            pbar.set_description(f"train: {loss}")
            loss_sum += loss.detach()
            
            loss.backward()
            self.optimizer.step()
            
            if itered == 50:
                wandb.log({'Loss':loss_sum / float(itered)})
                itered = 0
                loss_sum = 0
        wandb.log({'Loss':loss_sum / float(itered)})
                
        
    def test(self, epoch:int):
        
        self.eer = {}
        self.model.eval()
        
        for enroll_type in self.enroll_datasets.keys():
        # ------------------ enroll test utterances ------------------ #
        
            enrolled = self.enrollment(enroll_dataset=self.enroll_datasets[enroll_type])
        
        # ------------------ getting similarity scores between trial embeddings ------------------ #
            similarity = []
            is_same = []
        
            for spk_id1, spk_id2, same in tqdm(self.trial_list, desc=f"eval_{enroll_type}"):
            
                spk1_embs =enrolled[spk_id1] # shape: (N, embedding_size), size: 1
                spk2_embs =enrolled[spk_id2] # shape: (N, embedding_size), size: 1
                _score = torch.matmul(spk1_embs, spk2_embs.T).clamp(min=-1, max=1)                        
                score = torch.mean(_score).clamp(min=-1, max=1)
                    
                similarity.append(score)
                is_same.append(same)
            
        # ------------------ calculate EER of Trials ------------------ #
            self.eer[enroll_type] = calculate_EER(similarity, is_same) * 100.
            
        # ------------------ log ------------------ #
            wandb.log({
                f"{enroll_type}_eer": self.eer[enroll_type],
                "epoch" : epoch
            })
            
            if self.eer[enroll_type] < self.best_eer[enroll_type]:
                self.best_eer[enroll_type] = self.eer[enroll_type]
                wandb.log({f"{enroll_type}_best_eer" : self.eer[enroll_type], "epoch" : epoch})
        
            
        
        return self.eer
    
    def enrollment(self, enroll_dataset):
        """_summary_

        Args:
            enroll_dataset (Dataset): Dataset to use for enrollment

        Returns:
            dict: embeddings identified by path of utterance file(vox1 test - eg.'id10270/5r0dWxy17C8/00001.wav')
        """
        
        enroll_loader = data.DataLoader(
            dataset=enroll_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.sys_config.num_workers
        )
        
        enrolled = {}
        
        with torch.no_grad():
            for uts, spk_id in tqdm(enroll_loader, desc="enrollment"):
                uts = uts.squeeze(0)
                spk_id = spk_id[0]
                uts = uts.to(self.sys_config.device)
                uts = self.preprocessing(uts)
                embs = self.model(uts).cpu()
                enrolled[spk_id] = F.normalize(input=embs, p=2, dim=1)
                
        return enrolled
                
                
                