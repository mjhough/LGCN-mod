import world
import utils
from world import cprint
import torch
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

'''
change to checkpoint save name when loading pretrained model
of the form 'checkpoint_name.pkl'
'''
CHECKPOINT_NAME = None # change to checkpoint save name when loading pretrained model

# For saving epoch seconds
#  epoch_secs = []
#  epoch_mbs = []

if CHECKPOINT_NAME is not None and world.model_name == "lgcn2":
    checkpoint = torch.load(CHECKPOINT_NAME)
    Recmodel = register.MODELS[world.model_name](world.config, dataset, checkpoint['state_dict'])
    Recmodel = Recmodel.to(world.device)
    bpr = utils.BPRLoss(Recmodel, world.config)
    # if saved checkpoint
    #  bpr.getOpt().load_state_dict(checkpoint['optimizer'])
else:
    Recmodel = register.MODELS[world.model_name](world.config, dataset)
    Recmodel = Recmodel.to(world.device)
    bpr = utils.BPRLoss(Recmodel, world.config)

for key, val in checkpoint['state_dict'].items():
    Recmodel.state_dict()[key] = val

# FREEZE HERE
#  Recmodel.embedding_user.weight.requires_grad = False
#  Recmodel.embedding_item.weight.requires_grad = False

Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")


# Store the model size in a variable:
#mb_params = 1e-6*sum([param.nelement()*param.element_size() for param in Recmodel.parameters()])


try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch %60 == 0:
            cprint("[TEST]")

            #torch.cuda.reset_max_memory_allocated() # reset max memory stats for next iter
            test_t0 = time.time()
            Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            test_t1 = time.time()
            #  test_mb = 1e-6*torch.cuda.max_memory_allocated()
            #  torch.cuda.reset_max_memory_allocated() # reset max memory stats for next iter
            #  print("Test time: ", test_t1-test_t0)
            #  print("Test MB: ", test_mb)

            state = {
                'state_dict': Recmodel.state_dict(),
                'optimizer': bpr.getOpt().state_dict()
            }
            torch.save(state, 'NAME_OF_CHECKPOINT.pkl')
            print("SAVED")

        epoch_start = time.time()
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        epoch_end = time.time()
        #  print("Time for epoch:", epoch_end - epoch_start)

        #epoch_secs.append(epoch_end - epoch_start) # record training time of current epoch
        #epoch_mb = 1e-6*torch.cuda.max_memory_allocated()
        #epoch_mbs.append(epoch_mb) # record max mem allocated each iter
        #torch.cuda.reset_max_memory_allocated() # reset max memory stats for next iter
        #print("Max mem this iter:", epoch_mb)

        print(f'(FULL) EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
finally:
    if world.tensorboard:
        w.close()

#df = pd.DataFrame({'secs': epoch_secs, 'mbs': epoch_mbs})
#df.to_csv("stats.csv",sep=',')
#print("Param memory usage: ", mb_params)
