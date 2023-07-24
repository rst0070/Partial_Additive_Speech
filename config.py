import torch

class ExpConfig:
    """
    
    """
    def __init__(self):
        
        self.max_epoch              =  100
        self.batch_size             =  100
        # ------------------ mel spectrogram config ------------------ #
        self.n_mels                 =   64
        self.sample_rate            =   16000
        self.n_fft                  =   1024
        self.win_length             =   400
        self.hop_length             =   160
        self.window_fn              =   torch.hamming_window
        # ------------------ length of utterances ------------------ #
        self.windows                =   320
        self.train_sample           =   int(self.sample_rate * 3.2) - 1
        self.test_sample            =   int(self.sample_rate * 3.2) - 1
        # ------------------ pre emphasis coefficient ------------- #
        self.pre_emphasis           =   0.97
        # ------------------ model output config ------------------ #
        self.embedding_size         =   128
        self.output_size            =   1211
        # ------------------ loss config ------------------ #
        self.loss_scale             =   15.
        self.loss_margin            =   0.3
        # ------------------ optimizer setting ------------------ #
        self.lr                     =   1e-3
        self.lr_min                 =   1e-7
        self.weight_decay           =   1e-4
        self.amsgrad                =   True
        # ------------------ learning rate scheduler ------------------ #
        self.lr_sch_step_size       =   1       
        self.lr_sch_gamma           =   0.94
        # ------------------ data augmentation setting ------------------ #
        self.use_pas                =   True # if False: use TAN
        self.pas_min_utter          =   1 * self.sample_rate # duration * sample rate
        
class SysConfig:
    
    def __init__(self):
        # ------------------ path of voxceleb1 ------------------ #
        self.path_vox1_dev_root     = '/data/voxceleb1/train'
        #self.path_vox1_dev_label    = 'label/check_vox1_dev.csv'
        self.path_vox1_dev_label    = 'label/vox1_dev.csv'
        self.path_vox1_test_root    = '/data/voxceleb1/test'
        #self.path_vox1_enroll_label = 'label/check_vox1_enroll.csv'
        self.path_vox1_enroll_label = 'label/vox1_enroll.csv'
        #self.path_vox1_test_label   = 'label/check_vox1_test.csv'
        self.path_vox1_test_label   = 'label/vox1_test.csv'
        # ------------------ path of musan ------------------ #
        self.path_musan_train       = '/data/vox1_musan/musan_split/train'
        self.path_musan_test        = '/data/vox1_musan/musan_split/test'
        # ------------------ paths of (vox1 test + musan) ------------------ #
        # key = 'category_snr', value = path of root folder of test
        self.path_noisy_tests       = {
            'music_0' : '/data/vox1_musan/test/music_0', 'music_5' : '/data/vox1_musan/test/music_5',
            'music_10' : '/data/vox1_musan/test/music_10', 'music_15' : '/data/vox1_musan/test/music_15', 'music_20' : '/data/vox1_musan/test/music_20',
            
            'noise_0' : '/data/vox1_musan/test/noise_0', 'noise_5' : '/data/vox1_musan/test/noise_5',
            'noise_10' : '/data/vox1_musan/test/noise_10', 'noise_15' : '/data/vox1_musan/test/noise_15', 'noise_20' : '/data/vox1_musan/test/noise_20',
            
            'speech_0' : '/data/vox1_musan/test/speech_0', 'speech_5' : '/data/vox1_musan/test/speech_5',
            'speech_10' : '/data/vox1_musan/test/speech_10', 'speech_15' : '/data/vox1_musan/test/speech_15', 'speech_20' : '/data/vox1_musan/test/speech_20'  
        }
        # ------------------ wandb setting ------------------ #
        self.wandb_disabled         = True
        self.wandb_key              = '8c8d77ae7f92de2b007ad093af722aaae5f31003'
        self.wandb_project          = 'data_aug'
        self.wandb_entity           = 'rst0070'
        self.wandb_name             = ''
        # ------------------ device setting ------------------ #
        self.num_workers            = 4
        self.device                 =   'cuda:0'
        """device to use for training and testing"""
        
        self.random_seed            = 1234

if __name__ == "__main__":
    exp = ExpConfig()
    print(exp.n_mels)