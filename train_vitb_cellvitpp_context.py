"""Stream A training: ViT-B + NP/HV + cell-context-attention classifier.

Single experimental delta vs Path B-3 (``train_vitb_cellvitpp.py``): the
per-cell NC head is ``CellContextClassifier`` (transformer over cells +
2D centroid pos-enc) instead of the pooled-token MLP. Everything else —
NP/HV losses, per-cell CE, class weights, dataloaders, atomic save, early
stop — is reused verbatim by importing it from the B-3 script, so the
detection objective and training regime are byte-identical.

For the clean comparison against the B-3 no-cons baseline, run with the
ADIOS prior off:

    python train_vitb_cellvitpp_context.py --config configs/nuclei_counter.py \\
        --output_dir ./logs/cellvitpp_context --use_adios_consistency false

The verdict driver (HANDOFF_pathB3_results.md): does letting cells attend
to each other recover the connective recall that the pooled-token MLP lost?
"""

import copy
import importlib
import os

import torch

from cellvit.utils import WarmupDecayScheduler, set_seed

from adios_cellvit.adios_backbone import load_adios_mask_model
from adios_cellvit.channel_selector import ChannelSelector
from adios_cellvit.vitb_backbone import load_vitb_encoder
from adios_cellvit.vitb_cellvitpp_context_model import ViTBCellViTPPContext

# Reuse the B-3 loss + helpers unchanged (the NC term is still per-cell CE
# on the flat [N_cells, K] logits, which the context classifier preserves).
from train_vitb_cellvitpp import (
    CombinedLossCellViTPP,
    build_loaders,
    compute_cell_class_weights,
    parse_args,
    run_epoch,
)


def _load_config(config_path: str) -> dict:
    spec = importlib.util.spec_from_file_location('nuclei_counter_cfg', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.STAGE_CELLVITPP_CONTEXT


def main():
    args = parse_args()
    config = _load_config(args.config)
    for key in ('vitb_checkpoint', 'adios_checkpoint', 'stage1_selector',
                'pannuke_path', 'output_dir'):
        cli_val = getattr(args, key)
        if cli_val is not None:
            config[key] = cli_val
    if args.use_adios_consistency is not None:
        config['use_adios_consistency'] = args.use_adios_consistency
    # Record the architecture in the saved config so any .pth traces back to
    # the exact model class that produced it.
    config['model_class'] = 'ViTBCellViTPPContext'

    set_seed(config['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(config['output_dir'], exist_ok=True)

    # ---- model -------------------------------------------------------------
    encoder, patch_size, num_registers, embed_dim = load_vitb_encoder(
        config['vitb_checkpoint'], device,
    )
    model = ViTBCellViTPPContext(
        encoder=encoder,
        patch_size=patch_size,
        num_registers=num_registers,
        encoder_dim=embed_dim,
        num_cell_classes=config['num_cell_classes'],
        classifier_hidden_dim=config['classifier_hidden_dim'],
        classifier_dropout=config['classifier_dropout'],
        context_num_layers=config['context_num_layers'],
        context_num_heads=config['context_num_heads'],
        context_dim_feedforward=config['context_dim_feedforward'],
        drop_rate=0.1,
    ).to(device)

    if config.get('use_adios_consistency', False):
        prior_mask_model = load_adios_mask_model(config['adios_checkpoint'], device)
        prior_selector = ChannelSelector(num_masks=3).to(device)
        stage1_ckpt = torch.load(
            config['stage1_selector'], map_location=device, weights_only=False,
        )
        prior_selector.load_state_dict(stage1_ckpt['selector_state_dict'])
        model.set_adios_prior(prior_mask_model, prior_selector)
        print(f"[train_context] ADIOS prior attached "
              f"(lambda_adios={config['lambda_adios']})")
    else:
        print("[train_context] no ADIOS prior (no-consistency ablation)")

    # ---- loss --------------------------------------------------------------
    cell_weights = compute_cell_class_weights(
        config['pannuke_path'],
        config['output_dir'],
        split='Training',
        magnification=config['magnification'],
        num_cell_classes=config['num_cell_classes'],
    )
    criterion = CombinedLossCellViTPP(
        weights=config['loss_weights'],
        cell_class_weights=cell_weights,
        lambda_adios=config['lambda_adios'] if config.get('use_adios_consistency', False) else 0.0,
    ).to(device)

    # ---- optimizer ---------------------------------------------------------
    encoder_params = list(model.cellvit.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    # Heads = NP + HV branches + the cell-context classifier (the frozen NC
    # decoder is excluded by its requires_grad=False).
    heads_params = [
        p for p in model.parameters()
        if id(p) not in encoder_ids and p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {'params': heads_params,   'lr': config['heads_lr']},
            {'params': encoder_params, 'lr': config['encoder_lr']},
        ],
        weight_decay=config['weight_decay'],
    )
    scheduler = WarmupDecayScheduler(
        optimizer,
        warmup_epochs=config['warmup_epochs'],
        total_epochs=config['epochs'],
        base_lr=config['heads_lr'],
        final_lr=config['heads_lr'] / 10,
        warmup_start_lr=config['heads_lr'] / 100,
    )
    optimizer.param_groups[1]['lr'] = config['encoder_lr']

    train_loader, val_loader = build_loaders(config)

    # ---- loop --------------------------------------------------------------
    best_val_total = float('inf')
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0
    patience = config.get('early_stop_patience', 15)
    ckpt_path = os.path.join(config['output_dir'], 'cellvitpp_context.pth')

    for epoch in range(config['epochs']):
        train_log = run_epoch(model, criterion, train_loader, optimizer, device, train=True)
        val_log = run_epoch(model, criterion, val_loader, optimizer, device, train=False)
        scheduler.step()
        optimizer.param_groups[1]['lr'] = config['encoder_lr']

        lr_hds = optimizer.param_groups[0]['lr']
        lr_enc = optimizer.param_groups[1]['lr']
        print(
            f"[context] epoch {epoch+1}/{config['epochs']} "
            f"train_total={train_log['total']:.4f} val_total={val_log['total']:.4f} "
            f"val_nc={val_log['nc']:.4f} val_mse={val_log['mse']:.4f} "
            f"train_cons={train_log['cons']:.4f} val_cons={val_log['cons']:.4f} "
            f"lr_enc={lr_enc:.2e} lr_heads={lr_hds:.2e}"
        )

        if val_log['total'] < best_val_total:
            best_val_total = val_log['total']
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_since_improve = 0
            # Atomic incremental save: write .tmp then os.replace — Ctrl+C at
            # any later epoch leaves the best-so-far checkpoint intact.
            tmp_path = ckpt_path + '.tmp'
            torch.save(
                {
                    'model_state_dict': best_state,
                    'epoch': best_epoch,
                    'val_total': best_val_total,
                    'config': config,
                },
                tmp_path,
            )
            os.replace(tmp_path, ckpt_path)
            print(f"  [new best @ ep {best_epoch}] saved {ckpt_path}")
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"[context] early stop at epoch {epoch+1} "
                      f"(no val improvement in {patience} epochs)")
                break

    print(f"[context] training complete.  "
          f"Best val_total={best_val_total:.4f} @ epoch {best_epoch}.  "
          f"Saved to {ckpt_path}")


if __name__ == '__main__':
    main()
