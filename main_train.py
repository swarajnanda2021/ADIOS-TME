import argparse
import multiprocessing as mp
from pathlib import Path
from configs.config import get_args_parser
from training.trainer import train_adios_tme

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(
        'ADIOS-TME Training', 
        parents=[get_args_parser()]
    )
    args = parser.parse_args()
    
    # Set ADIOS-specific defaults
    args.num_masks = 3
    args.mask_update_freq = 5
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Use ADIOS trainer
    train_adios_tme(args)