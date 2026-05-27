"""Stage 1 / Stage 2 configuration for the nuclei counter branch.

Placeholder strings (``<FILL ON CLUSTER>``) make it visually obvious that the
cluster assembly step still has paths to fill in.
"""

STAGE1 = {
    'adios_checkpoint':     '/data1/vanderbc/nandas1/ADIOS-CellViT/logs/checkpoint_iter_00094000.pth',
    'pannuke_path':         '/data1/vanderbc/test_dinov2_swaraj/ADIOS/data/pannuke',
    'output_dir':           './logs/stage1',
    'magnification':        '40x',
    'batch_size':           32,
    'num_workers':          4,
    'max_epochs':           60,
    'lr':                   1e-4,
    'weight_decay':         1e-5,
    'warmup_epochs':        2,
    'val_split':            0.2,
    'early_stop_patience':  15,
    'normalize_mean':       (0.6816, 0.5640, 0.7232),
    'normalize_std':        (0.1617, 0.1714, 0.1389),
    'seed':                 42,
}

STAGE2 = {
    'adios_checkpoint':     '/data1/vanderbc/nandas1/ADIOS-CellViT/logs/checkpoint_iter_00094000.pth',
    'stage1_selector':      './logs/stage1/stage1_selector.pth',
    'pannuke_path':         '/data1/vanderbc/test_dinov2_swaraj/ADIOS/data/pannuke',
    'output_dir':           './logs/stage2',
    'magnification':        '40x',
    'batch_size':           32,
    'num_workers':          4,
    'max_epochs':           50,
    'lr_heads':             1e-4,
    'lr_adios_encoder':     1e-5,
    'lr_adios_decoder':     1e-4,
    'weight_decay':         1e-5,
    'warmup_epochs':        5,
    'val_split':            0.2,
    'early_stop_patience':  10,
    'num_classes':          6,  # 0=background + 5 PanNuke foreground classes
    'loss_weights': {
        'w_xentropy':  1.0,
        'w_dice':      1.0,
        'w_mse':       2.5,
        'w_msge':      8.0,
        'w_ftversky':  0.0,
        'w_nc':        1.0,
    },
    'normalize_mean':       (0.6816, 0.5640, 0.7232),
    'normalize_std':        (0.1617, 0.1714, 0.1389),
    'seed':                 42,
}
