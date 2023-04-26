OUT_DIR = 'mnist/'
NUM_WORKERS = 16
BATCH_SIZE = 128

from robustness1.robustness import model_utils, datasets, train, defaults
from robustness1.robustness.datasets import MNIST
import torch as ch

# We use cox (http://github.com/MadryLab/cox) to log, store and analyze
# results. Read more at https//cox.readthedocs.io.
from cox.utils import Parameters
import cox.store

# Hard-coded dataset, architecture, batch size, workers
ds = MNIST('mnist/')
m, _ = model_utils.make_and_restore_model(arch='Net', dataset=ds)
train_loader, val_loader = ds.make_loaders(batch_size=BATCH_SIZE, workers=NUM_WORKERS)

# Create a cox store for logging
out_store = cox.store.Store(OUT_DIR)
adv_crit = ch.nn.CrossEntropyLoss(reduction='none').cuda()
def custom_adv_loss(model, inp, targ):
    logits = model(inp)
    return -adv_crit(logits, targ), logits

# Hard-coded base parameters
train_kwargs = {
    'out_dir': "train_out",
    'adv_train': 1,
    'custom_loss': custom_adv_loss,
    'constraint': 'FW',# '2'
    'eps': 5.0, #5.0
    'attack_lr': 1.0,#0.1, #1
    'attack_steps': 20, #40, 40, 50
    'should_normalize': 0,
    'epochs': 100,
    'weight_decay':1e-4
}
train_args = Parameters(train_kwargs)

# Fill whatever parameters are missing from the defaults
train_args = defaults.check_and_fill_args(train_args,
                        defaults.TRAINING_ARGS, MNIST)
train_args = defaults.check_and_fill_args(train_args,
                        defaults.PGD_ARGS, MNIST)

print(m)
#print(train_args)

# Train a model
train.train_model(train_args, m, (train_loader, val_loader), store=out_store)
