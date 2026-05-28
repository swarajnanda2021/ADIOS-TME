"""Stage 1 / Stage 2 / VitB configuration for the nuclei counter branch.

Placeholder strings (``<FILL ON CLUSTER>`` for STAGE1/2, ``<FILL ON CLUSTER (VITB)>``
for STAGE_VITB) make it visually obvious that the cluster assembly step still
has paths to fill in.

STAGE_VITB intentionally uses a *different* placeholder string than STAGE1/2.
``assemble_cluster.sh``'s PHASE E does four ordered ``str.replace`` calls
against the literal ``<FILL ON CLUSTER>`` and then errors out if any of that
exact substring remain. Tagging STAGE_VITB's placeholders with ``(VITB)``
keeps them invisible to PHASE E, so existing assembly still works without
modification — the user fills the VITB placeholders manually (or via sed)
after assemble completes. See HANDOFF_pathB2.md §"Cluster assembly".
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

# Path B-2: ViT-B encoder + three-branch CellViT + optional ADIOS prior
# consistency.  Trained by ``train_vitb.py``; evaluated by
# ``eval_full_v2_vitb.py``.  num_classes=6 follows STAGE2 (0=background +
# 5 PanNuke foreground classes); CellViT's NC head is 6-channel and
# ``F.cross_entropy(pred[6ch], class_mask in {0..5}, weight=nc_class_weights)``
# matches the Stage 2 convention.
STAGE_VITB = {
    'vitb_checkpoint':       '<FILL ON CLUSTER (VITB)>',  # e.g. .../FMC_ViT-B_recipe_canonteach/logs/checkpoint_iter_00150000.pth
    'adios_checkpoint':      '<FILL ON CLUSTER (VITB)>',  # same value as STAGE2['adios_checkpoint']
    'stage1_selector':       '<FILL ON CLUSTER (VITB)>',  # path to the trained stage-1 ChannelSelector checkpoint
    'pannuke_path':          '<FILL ON CLUSTER (VITB)>',  # same value as STAGE2['pannuke_path']
    'output_dir':            '<FILL ON CLUSTER (VITB)>',  # e.g. ./logs/vitb_pathb2
    'magnification':         '40x',
    'num_classes':           6,  # 0=background + 5 PanNuke foreground classes
    'normalize_mean':        (0.6816, 0.5640, 0.7232),
    'normalize_std':         (0.1617, 0.1714, 0.1389),
    'val_split':             0.1,
    'seed':                  42,
    'num_workers':           4,
    'epochs':                100,
    'batch_size':            16,
    'encoder_lr':            1e-5,
    'heads_lr':              1e-4,
    'weight_decay':          1e-5,
    'warmup_epochs':         2,
    'early_stop_patience':   15,   # break loop if val_total hasn't improved in N epochs
    'use_adios_consistency': True,
    'lambda_adios':          0.1,
    # HV-weight tuning: mirror STAGE2 (Path Z) and STAGE_CELLVITPP's
    # known-working values. The original 1.0/1.0 defaults left val_mse
    # pegged at ~0.0145 and produced under-detection (count bias -6 to -7
    # cells/patch). MSGE in particular drives sharp HV-map boundaries
    # that watershed needs for instance separation.
    'loss_weights': {
        'w_xentropy': 1.0,
        'w_dice':     1.0,
        'w_mse':      2.5,
        'w_msge':     8.0,
        'w_nc':       1.0,
    },
}

# Path B-3: ViT-B encoder + NP/HV CellViT branches + per-cell MLP classifier
# (CellViT++ style). Same training regime as STAGE_VITB so the only
# experimental delta is the NC head architecture: per-pixel decoder ->
# per-cell pooled-token MLP. Trained by ``train_vitb_cellvitpp.py``;
# evaluated by ``eval_full_v3_cellvitpp.py``.
#
# num_cell_classes = 5: the MLP is foreground-only (neoplastic / inflammatory
# / connective / dead / epithelial). Background is not a cell-level class.
# CE target shifts modal-class {1..5} -> {0..4} inside the training loop.
#
# Placeholder tag (CELLVITPP) is distinct from <FILL ON CLUSTER> so
# assemble_cluster.sh PHASE E's substring match doesn't try to substitute
# these or crash on their presence. See HANDOFF_cellvitpp.md.
STAGE_CELLVITPP = {
    'vitb_checkpoint':       '<FILL ON CLUSTER (CELLVITPP)>',  # e.g. .../FMC_ViT-B_baseline/logs/checkpoint_iter_00150000.pth
    'adios_checkpoint':      '<FILL ON CLUSTER (CELLVITPP)>',  # same value as STAGE2['adios_checkpoint']; only needed when use_adios_consistency=True
    'stage1_selector':       '<FILL ON CLUSTER (CELLVITPP)>',  # trained stage-1 selector checkpoint; only needed when use_adios_consistency=True
    'pannuke_path':          '<FILL ON CLUSTER (CELLVITPP)>',  # same value as STAGE2['pannuke_path']
    'output_dir':            '<FILL ON CLUSTER (CELLVITPP)>',  # e.g. ./logs/cellvitpp
    'magnification':         '40x',
    'num_cell_classes':      5,  # foreground-only: 5 PanNuke classes
    'classifier_hidden_dim': 384,
    'classifier_dropout':    0.1,
    'normalize_mean':        (0.6816, 0.5640, 0.7232),
    'normalize_std':         (0.1617, 0.1714, 0.1389),
    'val_split':             0.1,
    'seed':                  42,
    'num_workers':           4,
    'epochs':                100,
    'batch_size':            16,
    'encoder_lr':            1e-5,
    'heads_lr':              1e-4,
    'weight_decay':          1e-5,
    'warmup_epochs':         2,
    'early_stop_patience':   15,   # break loop if val_total hasn't improved in N epochs
    'use_adios_consistency': True,
    'lambda_adios':          0.1,
    # HV-weight tuning: mirror STAGE2's known-working values
    # (w_mse=2.5, w_msge=8.0). The defaults 1.0/1.0 in earlier B-2 / B-3
    # runs left val_mse pegged at ~0.0145 and produced under-detection
    # (count bias -6 to -7).  MSGE in particular drives sharp HV-map
    # boundaries, which watershed needs for instance separation.
    'loss_weights': {
        'w_xentropy': 1.0,
        'w_dice':     1.0,
        'w_mse':      2.5,
        'w_msge':     8.0,
        'w_nc':       1.0,
    },
}

# Stream A: ViT-B encoder + NP/HV CellViT branches + cell-context-attention
# classifier. Identical to STAGE_CELLVITPP except the per-cell NC head: B-3's
# pooled-token MLP -> a small transformer that lets cells in an image attend to
# each other (2D positional encoding from centroids). This is the single
# experimental delta vs B-3. Trained by ``train_vitb_cellvitpp_context.py``;
# the model class is ``ViTBCellViTPPContext``.
#
# context_* knobs size the transformer (the experimental head).
# classifier_hidden_dim / classifier_dropout now size the MLP head that sits
# *after* attention. num_cell_classes stays 5 (foreground-only).
#
# loss_weights stay w_mse=2.5 / w_msge=8.0: Stream A is still single-dataset
# PanNuke, so the known-good detection weights apply (the 1.0/1.0 start is only
# for the 10-class multi-dataset Stream B, where MSGE x8 collapses rare classes).
#
# Placeholder tag (CONTEXT) is distinct from <FILL ON CLUSTER>, (VITB) and
# (CELLVITPP) so assemble_cluster.sh PHASE E's substring match leaves it alone.
STAGE_CELLVITPP_CONTEXT = {
    'vitb_checkpoint':         '<FILL ON CLUSTER (CONTEXT)>',  # e.g. .../FMC_ViT-B_baseline/logs/checkpoint_iter_00150000.pth
    'adios_checkpoint':        '<FILL ON CLUSTER (CONTEXT)>',  # same value as STAGE2['adios_checkpoint']; only needed when use_adios_consistency=True
    'stage1_selector':         '<FILL ON CLUSTER (CONTEXT)>',  # trained stage-1 selector checkpoint; only needed when use_adios_consistency=True
    'pannuke_path':            '<FILL ON CLUSTER (CONTEXT)>',  # same value as STAGE2['pannuke_path']
    'output_dir':              '<FILL ON CLUSTER (CONTEXT)>',  # e.g. ./logs/cellvitpp_context
    'magnification':           '40x',
    'num_cell_classes':        5,    # foreground-only: 5 PanNuke classes
    'classifier_hidden_dim':   384,  # width of the MLP head after attention
    'classifier_dropout':      0.1,
    # Cell-context attention knobs (the experimental delta vs B-3's MLP head).
    'context_num_layers':      2,
    'context_num_heads':       8,
    'context_dim_feedforward': 2048,
    'normalize_mean':          (0.6816, 0.5640, 0.7232),
    'normalize_std':           (0.1617, 0.1714, 0.1389),
    'val_split':               0.1,
    'seed':                    42,
    'num_workers':             4,
    'epochs':                  100,
    'batch_size':              16,
    'encoder_lr':              1e-5,
    'heads_lr':                1e-4,
    'weight_decay':            1e-5,
    'warmup_epochs':           2,
    'early_stop_patience':     15,
    'use_adios_consistency':   True,
    'lambda_adios':            0.1,
    'loss_weights': {
        'w_xentropy': 1.0,
        'w_dice':     1.0,
        'w_mse':      2.5,
        'w_msge':     8.0,
        'w_nc':       1.0,
    },
}
