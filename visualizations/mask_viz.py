"""
Mask visualization utilities for ADIOS-TME training.
Based on the reference ADIOS implementation with comprehensive visualization.
"""

import os
import signal
import contextlib
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc


class TimeoutError(Exception):
    pass


@contextlib.contextmanager
def timeout_context(seconds):
    """Context manager for timeout handling"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def save_iteration_masks_efficient(
    images, 
    masks, 
    iteration, 
    save_dir, 
    reconstructed_images=None, 
    num_samples=4, 
    timeout_seconds=30
):
    """
    Efficient, robust mask visualization that prevents hanging.
    Creates comprehensive visualizations showing original, masks, combined, and reconstructed images.
    
    Args:
        images: Input images tensor [B, C, H, W]
        masks: Generated masks tensor [B, num_masks, H, W] 
        iteration: Current training iteration
        save_dir: Directory to save visualizations
        reconstructed_images: Pre-computed reconstructed images tensor [B, C, H, W] (optional)
        num_samples: Number of samples to visualize (default: 4)
        timeout_seconds: Maximum time allowed for visualization (default: 30)
    """
    try:
        with timeout_context(timeout_seconds):
            os.makedirs(save_dir, exist_ok=True)
            
            # Move everything to CPU immediately and convert to float32
            images_cpu = images.detach().cpu().float()
            masks_cpu = masks.detach().cpu().float()
            reconstructed_cpu = None
            if reconstructed_images is not None:
                reconstructed_cpu = reconstructed_images.detach().cpu().float()
            
            # Clear CUDA cache to prevent memory issues
            torch.cuda.empty_cache()
            
            # Sample selection - reduce to manageable size
            batch_size = images_cpu.size(0)
            num_samples = min(num_samples, batch_size, 4)  # Cap at 4 samples max
            
            torch.manual_seed(42)
            if batch_size > num_samples:
                indices = torch.randperm(batch_size)[:num_samples]
                images_cpu = images_cpu[indices]
                masks_cpu = masks_cpu[indices]
                if reconstructed_cpu is not None:
                    reconstructed_cpu = reconstructed_cpu[indices]
            
            # Denormalization on CPU using pathology-specific values
            mean_cpu = torch.tensor([0.6816, 0.5640, 0.7232]).view(1, 3, 1, 1)
            std_cpu = torch.tensor([0.1617, 0.1714, 0.1389]).view(1, 3, 1, 1)
            
            # Unnormalize images
            images_norm = images_cpu * std_cpu + mean_cpu
            images_norm = torch.clamp(images_norm, 0, 1)
            
            # Unnormalize reconstructed images if available
            if reconstructed_cpu is not None:
                reconstructed_norm = reconstructed_cpu * std_cpu + mean_cpu
                reconstructed_norm = torch.clamp(reconstructed_norm, 0, 1)
            
            num_masks = masks_cpu.size(1)
            
            # Create comprehensive visualization
            # Columns: Original + Individual Masks + RGB Combined + (optional) Reconstructed
            cols = min(num_masks + 2, 6)  # Original + masks + combined, max 6 cols
            if reconstructed_cpu is not None:
                cols += 1  # Add column for reconstruction
            
            fig, axes = plt.subplots(num_samples, cols, figsize=(3*cols, 3*num_samples))
            
            # Handle single row case
            if num_samples == 1:
                axes = axes.reshape(1, -1)
            
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            
            for i in range(num_samples):
                col_idx = 0
                
                # Column 1: Original image
                img_np = images_norm[i].permute(1, 2, 0).numpy()
                axes[i, col_idx].imshow(img_np)
                axes[i, col_idx].axis('off')
                if i == 0:
                    axes[i, col_idx].set_title('Original', fontsize=10)
                col_idx += 1
                
                # Columns 2-N: Individual masks (limit to available space)
                masks_to_show = min(num_masks, cols - 2 - (1 if reconstructed_cpu is not None else 0))
                for j in range(masks_to_show):
                    mask_np = masks_cpu[i, j].numpy()
                    axes[i, col_idx].imshow(mask_np, cmap='viridis', vmin=0, vmax=1)
                    axes[i, col_idx].axis('off')
                    if i == 0:
                        axes[i, col_idx].set_title(f'Mask {j+1}', fontsize=10)
                    col_idx += 1
                
                # Combined visualization (RGB if 3+ masks, grayscale otherwise)
                if num_masks >= 3:
                    # RGB combination using first 3 masks
                    rgb_masks = torch.stack([
                        masks_cpu[i, 0], 
                        masks_cpu[i, 1], 
                        masks_cpu[i, 2]
                    ], dim=0).permute(1, 2, 0).numpy()
                    axes[i, col_idx].imshow(rgb_masks, vmin=0, vmax=1)
                    title = 'RGB Combined'
                else:
                    # Average of all masks
                    avg_mask = masks_cpu[i].mean(dim=0).numpy()
                    axes[i, col_idx].imshow(avg_mask, cmap='viridis', vmin=0, vmax=1)
                    title = 'Avg Masks'
                
                axes[i, col_idx].axis('off')
                if i == 0:
                    axes[i, col_idx].set_title(title, fontsize=10)
                col_idx += 1
                
                # Reconstructed image (if available)
                if reconstructed_cpu is not None:
                    recon_np = reconstructed_norm[i].permute(1, 2, 0).numpy()
                    axes[i, col_idx].imshow(recon_np)
                    axes[i, col_idx].axis('off')
                    if i == 0:
                        axes[i, col_idx].set_title('Reconstructed', fontsize=10)
            
            # Save with error handling
            save_path = os.path.join(save_dir, f'iter_{iteration:06d}_masks.png')
            plt.savefig(save_path, bbox_inches='tight', dpi=100, facecolor='white')
            
            # Explicit cleanup
            plt.close(fig)
            plt.clf()
            
            # Force garbage collection
            del images_cpu, masks_cpu, images_norm
            if reconstructed_cpu is not None:
                del reconstructed_cpu, reconstructed_norm
            gc.collect()
            
            print(f"Mask visualization saved to {save_path}")
            
    except TimeoutError:
        print(f"Visualization timed out after {timeout_seconds}s, skipping")
    except Exception as e:
        print(f"Visualization failed: {str(e)}")
        # Clean up any partial matplotlib state
        plt.close('all')
        plt.clf()
    finally:
        # Ensure matplotlib cleanup
        try:
            plt.close('all')
        except:
            pass


def safe_visualization_wrapper(images, masks, iteration, save_dir, reconstructed_images=None):
    """
    Wrapper function that ensures visualization never crashes training.
    Only runs on main process to avoid conflicts.
    
    Args:
        images: Input images tensor [B, C, H, W]
        masks: Generated masks tensor [B, num_masks, H, W]
        iteration: Current training iteration
        save_dir: Directory to save visualizations
        reconstructed_images: Optional reconstructed images tensor [B, C, H, W]
    """
    try:
        # Only visualize on main process to avoid conflicts
        import utils
        if not utils.is_main_process():
            return
        
        save_iteration_masks_efficient(images, masks, iteration, save_dir, reconstructed_images)
        
    except Exception as e:
        print(f"Visualization wrapper failed: {str(e)}")
        print("Training will continue without visualization")