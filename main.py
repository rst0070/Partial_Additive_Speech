import wandb
import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np
import config
import os
from models.se_resnet import SEResNet34
from models.ecapa_tdnn import ECAPA_TDNN
from trainer.exp_trainer import Trainer

class Main:
    
    def __init__(self):
        
        sys_config = config.SysConfig()
        exp_config = config.ExpConfig()
        
        ###
        ###    seed setting
        ###
        seed = sys_config.random_seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        #  cudnn setting to enhance speed of dilated conv
        cudnn.deterministic = False
        cudnn.benchmark = True
        
        # cudnn.deterministic = True
        # cudnn.benchmark = False
        
        
        ###
        ###    wandb setting
        ###
        if sys_config.wandb_disabled:
            os.system("wandb disabled")
            
        os.system(f"wandb login {sys_config.wandb_key}")
        wandb.init(
            project = sys_config.wandb_project,
            entity  = sys_config.wandb_entity,
            name    = sys_config.wandb_name
        )
        
        
        ###
        ###    training environment setting
        ###
        self.max_epoch = exp_config.max_epoch
        #self.model = SEResNet34().to(sys_config.device)
        self.model = ECAPA_TDNN().to(sys_config.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr = exp_config.lr,
            weight_decay = exp_config.weight_decay,
            amsgrad=exp_config.amsgrad
        )
        
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=exp_config.lr_sch_step_size,
            gamma=exp_config.lr_sch_gamma
        )
        
        self.trainer = Trainer(model=self.model, optimizer=self.optimizer)
        
        
    def start(self):
        
        min_eer = None
        model_state = {}
        
        for epoch in range(1, self.max_epoch + 1):
            
            # --------------- train --------------- #
            self.trainer.train()
            
            self.lr_scheduler.step()  
            
            if epoch < 10 or epoch % 5 != 0:
                continue
            
            # --------------- test --------------- #
            eer = self.trainer.test(epoch)
            
            # --------------- eer check and save --------------- #
            if min_eer is None:
                min_eer = eer
                for test_type in min_eer.keys():
                    model_state[test_type] = self.model.state_dict()
            else:
                for test_type in min_eer.keys():
                    # save process by test type
                    if min_eer[test_type] < eer[test_type]:
                        continue
                    
                    min_eer[test_type] = eer[test_type]
                    model_state[test_type] = self.model.state_dict()
                    
                    file_name = f"{epoch}_{test_type}_{min_eer[test_type]}.pth"
                    torch.save(model_state[test_type], file_name)
                    wandb.save(file_name)
                    
            # --------------- log and schedule learning rate --------------- #
            print(f"epoch: {epoch} \neer: {eer} \nmin_eer:{min_eer}")        
            

if __name__ == '__main__':
    program = Main()
    program.start()
    