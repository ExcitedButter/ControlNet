# from share import *

# import pytorch_lightning as pl
# from torch.utils.data import DataLoader
# from tutorial_dataset import MyDataset
# from cldm.logger import ImageLogger
# from cldm.model import create_model, load_state_dict


# # Configs
# resume_path = './models/control_sd15_ini.ckpt'
# batch_size = 4
# logger_freq = 300
# learning_rate = 1e-5
# sd_locked = True
# only_mid_control = False


# # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
# model = create_model('./models/cldm_v15.yaml').cpu()
# model.load_state_dict(load_state_dict(resume_path, location='cpu'))
# model.learning_rate = learning_rate
# model.sd_locked = sd_locked
# model.only_mid_control = only_mid_control


# # Misc
# dataset = MyDataset()
# dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
# logger = ImageLogger(batch_frequency=logger_freq)
# trainer = pl.Trainer(auto_select_gpus=True, accelerator="gpu", precision=32, callbacks=[logger], accumulate_grad_batches=4)


# # Train!
# trainer.fit(model, dataloader)

from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# First use CPU to load models. PyTorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)

# Use all available GPUs and set the distributed training strategy to 'ddp'
trainer = pl.Trainer(
    accelerator="gpu",
    devices=-1,  # Use all available GPUs
    strategy="ddp",  # Distributed Data Parallel
    precision=32,
    callbacks=[logger],
    accumulate_grad_batches=4
)

# Train!
trainer.fit(model, dataloader)
