from polarmae.datasets import PILArNetDataModule
import torch
from polarmae.models.finetune import SemanticSegmentation
from polarmae.utils.checkpoint import load_finetune_checkpoint
from polarmae.utils import transforms
from math import sqrt

# Turn off gradient tracking so we don't run out of memory
torch.set_grad_enabled(False);

dataset = PILArNetDataModule(
    # data_path=f'/home/sc5303/sbnd_data/inss2025/data/schung_xyze_1e4.h5',
    data_path=f'/home/sc5303/sbnd_data/inss2025/data/challenge_xyze_1e4.h5',
    batch_size=32,
    num_workers=0,
    dataset_kwargs={
        'emin': 1.0e-3,                      # min energy for log transform
        'emax': 20.0,                        # max energy for log transform
        'energy_threshold': 0.13,            # remove points with energy < 0.13
        'remove_low_energy_scatters': False,  # remove low energy scatters (PID=4)
        'maxlen': -1,                        # max number of events to load
        'min_points': 0,                  # min number of points/event to load
    }
)
dataset.setup()

model = load_finetune_checkpoint(SemanticSegmentation,
                                 './polarmae_fft_segsem.ckpt',
                                 data_path=dataset.hparams.data_path,
                                 pretrained_ckpt_path='./polarmae_pretrain.ckpt').cuda()
model.eval();

normalize = transforms.PointcloudCenterAndNormalize(
                    center=[384, 384, 384],
                    scale_factor=1 / (768 * sqrt(3) / 2)
                )

val_loader = dataset.val_dataloader()  # or DataLoader(dataset.val_dataset, batch_size=..., shuffle=False)


batch_idx = 0

for batch in val_loader:
    points = batch['points'].cuda()
    if len(model.val_transformations.transforms) > 0:
        points = model.val_transformations(points)
    else:
        points = normalize(points) # scale and normalize
    lengths = batch['lengths'].cuda()

    output = model(points, lengths)
    pred = output['id_pred']
    point_mask = output['point_mask']
    pred = pred[point_mask]
    
    # Save all predictions from this batch
    import numpy as np
    np.save(f'predictions/predictions_batch_{batch_idx}.npy', pred.cpu().numpy())
    batch_idx+=1