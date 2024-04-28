from matplotlib.pyplot import step
from task.config.config_node import ConfigNode

config = ConfigNode()



config.project_name = "lightweight"


#train

config.train = ConfigNode()
config.train.output_dir = "logs/RAFDB/joint_learning/"



config.train.checkpoint_path = 'checkpoints/RAFDB/joint_learning/'
config.train.best_checkpoint_path = 'checkpoints/RAFDB/joint_learning/'
config.train.joint_frequency = 5
config.train.print_freq = 20
config.train.trans_mode = 2
config.train.seed = 114514
# Loss
config.train.loss1 = 'CrossEntropy'
config.train.loss2 = 'L1'
config.train.loss3 = 'L2'
config.train.remarks = ''
# optim
config.train.optimizer = 'sgd'
config.train.no_weight_decay_on_bn = False
config.train.weight_decay = 1e-4
config.train.base_lr = 0.01
config.train.momentum = 0.9
config.train.nesterov = False
#scheduler
config.scheduler = ConfigNode()
config.scheduler.type = 'step'
config.scheduler.step_size = 15
config.scheduler.lr_decay = 0.1
config.scheduler.start_epoch = 0
config.scheduler.epochs = 60
#dataset
config.dataset = ConfigNode()
config.dataset.traindir1 = "/home/frank/subFolder/CC-FER202203/RAFDBfromAll.txt"
config.dataset.traindir2 = "/home/frank/dataset/listpath/aff_train7_listpath.txt"
config.dataset.valdir = "/home/frank/dataset/listpath/RAFDB_aligned_test.txt"
#dataloader
config.dataloader = ConfigNode()
config.dataloader.batch_size = 128
config.dataloader.workers = 4


#model
config.model = ConfigNode()
config.model.name = 'ARM_joint'
config.model.beta = 0.6   #for MANET
config.model.reg_weight = 1.0






def get_default_config():
    return config.clone()
