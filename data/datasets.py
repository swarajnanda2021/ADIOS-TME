"""
Dataset for ADIOS-TME training with efficient sharding and corruption handling.
"""

import os
import torch
from torch.utils.data import Dataset, IterableDataset
from PIL import Image, PngImagePlugin, ImageFile
import zipfile
import io
import random
import numpy as np
import json
import glob
import time
import pickle
import fcntl
from datetime import datetime
from typing import Set


class ADIOSPathologyDataset(IterableDataset):
    """
    Memory-efficient dataset with proper sharding for distributed training.
    Returns single normalized images (no multi-crop augmentation).
    
    Args:
        data_path: Root directory containing zip files
        index_file: Path to dataset index pickle file
        img_size: Image size
        mean: Normalization mean
        std: Normalization std
        max_samples: Maximum samples for debugging
        corruptions_dir: Directory containing corruption logs
    """
    def __init__(
        self,
        data_path: str,
        index_file: str = "dataset_index.pkl",
        img_size: int = 224,
        mean: tuple = (0.6816, 0.5640, 0.7232),
        std: tuple = (0.1617, 0.1714, 0.1389),
        max_samples: int = None,
        corruptions_dir: str = "corruption_results"
    ):
        super().__init__()
        self.data_path = data_path
        self.index_file = index_file
        self.img_size = img_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.max_samples = max_samples
        self.corruptions_dir = corruptions_dir
        
        # Worker info (set dynamically)
        self.worker_id = 0
        self.num_workers = 1
        self.seed = 42
        
        # Setup corruption logging
        self.corruption_log_file = "runtime_corrupted_files.json"
        self.corruption_lock_file = f"{self.corruption_log_file}.lock"
        if not os.path.exists(self.corruption_lock_file):
            with open(self.corruption_lock_file, 'w') as f:
                pass
        
        # Load pre-known corrupted files
        self.corrupted_zip_files = self._load_known_corrupted_files()
        print(f"Loaded {len(self.corrupted_zip_files)} known corrupted zip files to exclude")
        
        # Load metadata
        self.index_metadata = self._load_index_metadata()
        self.total_images = self.index_metadata['total_images']
        
        # Filter corrupted files
        self._filter_corrupted_zip_files()
        
        # Defer shard calculation
        self.shard_calculated = False
        self.worker_files = []
        self.worker_image_ranges = []
        
        # Resume capability
        self.samples_to_skip = 0
        
        # Rate limiting
        self.error_count = 0
        self.last_error_time = time.time()
        self.max_errors_per_minute = 10
        
        print(f"Dataset initialized with {self.total_images} total images")
    
    def set_worker_info(self, worker_id: int, num_workers: int):
        """Set actual worker ID and num_workers from PyTorch DataLoader."""
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.shard_calculated = False
        print(f"Worker info set: worker_id={worker_id}, num_workers={num_workers}")
    
    def _load_known_corrupted_files(self) -> Set[str]:
        """Load pre-scanned corrupt file information from JSON files."""
        corrupted_zip_files = set()
        
        # Check both current directory and data path for corruptions directory
        possible_corruption_dirs = [
            self.corruptions_dir,
            os.path.join(os.getcwd(), self.corruptions_dir),
            os.path.join(self.data_path, self.corruptions_dir),
        ]
        
        corruption_dir = None
        for cdir in possible_corruption_dirs:
            if os.path.exists(cdir):
                corruption_dir = cdir
                break
        
        if not corruption_dir:
            print(f"Warning: Corruptions directory not found, checked: {possible_corruption_dirs}")
            return corrupted_zip_files
        
        json_files = glob.glob(os.path.join(corruption_dir, "*.json"))
        if not json_files:
            print(f"Warning: No corruption JSON files found in {corruption_dir}")
            return corrupted_zip_files
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    corruptions = json.load(f)
                
                for item in corruptions:
                    if 'zip_path' in item and item.get('error_type') == 'BadZipFile':
                        corrupted_zip_files.add(item['zip_path'])
                        
                print(f"Processed {json_file}: found {len(corruptions)} corruptions")
            except Exception as e:
                print(f"Error loading corruption file {json_file}: {e}")
        
        return corrupted_zip_files
    
    def _filter_corrupted_zip_files(self):
        """Filter out known corrupted zip files from the dataset."""
        if not self.corrupted_zip_files:
            self.zip_files = self.index_metadata['zip_files']
            self.images_per_zip = self.index_metadata['images_per_zip']
            return
        
        filtered_zip_files = []
        filtered_images_per_zip = []
        filtered_count = 0
        filtered_zips = 0
        
        for i, zip_path in enumerate(self.index_metadata['zip_files']):
            if zip_path in self.corrupted_zip_files:
                filtered_count += self.index_metadata['images_per_zip'][i]
                filtered_zips += 1
                continue
            
            filtered_zip_files.append(zip_path)
            filtered_images_per_zip.append(self.index_metadata['images_per_zip'][i])
        
        self.zip_files = filtered_zip_files
        self.images_per_zip = filtered_images_per_zip
        
        if filtered_zips > 0:
            print(f"Filtered out {filtered_count} images from {filtered_zips} corrupted zip files")
    
    def _load_index_metadata(self):
        """Load only metadata about the index."""
        # Check multiple locations for index file
        possible_paths = [
            self.index_file,
            os.path.join(os.getcwd(), self.index_file),
            os.path.join(self.data_path, self.index_file),
        ]
        
        index_path = None
        for path in possible_paths:
            if os.path.exists(path):
                index_path = path
                print(f"Found index file at: {index_path}")
                break
        
        if not index_path:
            raise FileNotFoundError(
                f"Could not find index file. Searched:\n" +
                "\n".join(f"  - {p}" for p in possible_paths)
            )
        
        # Check for metadata file
        index_metadata_path = index_path.replace('.pkl', '_metadata.pkl')
        
        if not os.path.exists(index_metadata_path):
            print(f"Creating index metadata from {index_path}")
            self._create_index_metadata(index_path, index_metadata_path)
        
        with open(index_metadata_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def _create_index_metadata(index_path, metadata_path):
        """Create lightweight metadata file from full index."""
        print(f"Loading full index from {index_path}...")
        with open(index_path, 'rb') as f:
            all_index = pickle.load(f)
        
        metadata = {
            'total_images': 0,
            'zip_files': [],
            'images_per_zip': []
        }
        
        for zip_path, image_names in all_index:
            # Filter for 448px images only
            filtered_images = [
                img for img in image_names 
                if '_448_' in img and '_224_' not in img
            ]
            num_images = len(filtered_images)
            
            if num_images > 0:
                metadata['zip_files'].append(zip_path)
                metadata['images_per_zip'].append(num_images)
                metadata['total_images'] += num_images
        
        print(f"Saving metadata with {metadata['total_images']} images to {metadata_path}")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def _calculate_worker_shard(self):
        """Calculate which files and image indices belong to this worker."""
        # For single-GPU training, world_size=1, rank=0
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        global_worker_id = rank * self.num_workers + self.worker_id
        total_workers = world_size * self.num_workers
        
        self.worker_files = []
        self.worker_image_ranges = []
        
        worker_indices = []
        
        for zip_idx, (zip_path, num_images) in enumerate(zip(self.zip_files, self.images_per_zip)):
            if zip_path in self.corrupted_zip_files:
                continue
                
            zip_seed = self.seed + hash(zip_path) % 10000
            rng = random.Random(zip_seed)
            
            for img_idx in range(num_images):
                worker = rng.randint(0, total_workers - 1)
                if worker == global_worker_id:
                    worker_indices.append((zip_idx, img_idx))
        
        if worker_indices:
            worker_indices.sort()
            
            current_zip = worker_indices[0][0]
            start_img = worker_indices[0][1]
            
            for i, (zip_idx, img_idx) in enumerate(worker_indices[1:] + [(None, None)]):
                if zip_idx != current_zip or zip_idx is None:
                    self.worker_files.append(self.zip_files[current_zip])
                    self.worker_image_ranges.append((start_img, worker_indices[i][1] + 1))
                    
                    if zip_idx is not None:
                        current_zip = zip_idx
                        start_img = img_idx
        
        self.shard_calculated = True
        print(f"Worker {global_worker_id}/{total_workers} will process {len(worker_indices)} images from {len(self.worker_files)} zip files")
    
    def _get_image_names(self, zip_path, start_idx, end_idx):
        """Get specific image names for a range within a zip file."""
        if zip_path in self.corrupted_zip_files:
            return []
            
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                all_images = [f for f in zf.namelist() 
                            if f.endswith('.png') 
                            and ('_448_' in f) 
                            and ('_224_' not in f)]
                all_images.sort()
                
                image_names = all_images[start_idx:end_idx]
            
            return image_names
        except Exception as e:
            print(f"Error reading zip file {zip_path}: {e}")
            self._log_corrupt_file(zip_path, "", e)
            self.corrupted_zip_files.add(zip_path)
            return []
    
    def _log_corrupt_file(self, zip_path, image_name, exception):
        """Log a corrupt file to JSON log with proper locking."""
        current_time = time.time()
        if current_time - self.last_error_time >= 60:
            self.error_count = 0
            self.last_error_time = current_time
            
        if self.error_count >= self.max_errors_per_minute:
            return
            
        self.error_count += 1
        
        corrupt_entry = {
            'zip_path': zip_path,
            'image_name': image_name,
            'error_type': type(exception).__name__,
            'error_msg': str(exception),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(self.corruption_lock_file, 'r+') as lockf:
                try:
                    fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    try:
                        try:
                            with open(self.corruption_log_file, 'r') as f:
                                existing_data = json.load(f)
                        except (FileNotFoundError, json.JSONDecodeError):
                            existing_data = []
                        
                        existing_data.append(corrupt_entry)
                        
                        with open(self.corruption_log_file, 'w') as f:
                            json.dump(existing_data, f, indent=2)
                    finally:
                        fcntl.flock(lockf, fcntl.LOCK_UN)
                except BlockingIOError:
                    pass
        except Exception as e:
            print(f"Error logging corrupt file: {e}")
    
    def _load_image(self, zip_path, image_name):
        """Load a single image from a zip file with error handling."""
        if zip_path in self.corrupted_zip_files:
            raise IOError(f"Skipping image from known corrupted zip file: {zip_path}")
            
        PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 * 1024)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        max_retries = 3
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    data = zf.read(image_name)
                    try:
                        img = Image.open(io.BytesIO(data)).convert('RGB')
                        return img
                    except ValueError as e:
                        if "Decompressed data too large" in str(e):
                            buffer = io.BytesIO(data)
                            img = Image.open(buffer)
                            img.load()
                            return img.convert('RGB')
                        raise
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
        
        current_time = time.time()
        if current_time - self.last_error_time >= 60:
            self.error_count = 0
            self.last_error_time = current_time
            
        if self.error_count < self.max_errors_per_minute:
            print(f"Failed to load image {image_name} from {zip_path} after {max_retries} attempts: {str(last_exception)}")
            self.error_count += 1
        
        self._log_corrupt_file(zip_path, image_name, last_exception)
        
        if isinstance(last_exception, (zipfile.BadZipFile, zipfile.LargeZipFile)):
            self.corrupted_zip_files.add(zip_path)
        
        raise IOError(f"Failed to load image after {max_retries} attempts: {str(last_exception)}")

    def __iter__(self):
        """Iterator with memory-efficient shuffling and resume support."""
        if not self.shard_calculated:
            self._calculate_worker_shard()
        
        rng = random.Random(self.seed + self.worker_id)
        
        shuffled_files = list(enumerate(self.worker_files))
        rng.shuffle(shuffled_files)
        
        samples_yielded = 0
        
        for file_idx, zip_path in shuffled_files:
            if zip_path in self.corrupted_zip_files:
                continue
                
            start_idx, end_idx = self.worker_image_ranges[file_idx]
            
            try:
                image_names = self._get_image_names(zip_path, start_idx, end_idx)
                
                if not image_names:
                    continue
                    
                rng.shuffle(image_names)
                
                for img_name in image_names:
                    if samples_yielded < self.samples_to_skip:
                        samples_yielded += 1
                        continue
                    
                    # Apply max_samples limit if set
                    if self.max_samples and samples_yielded >= self.max_samples:
                        return
                    
                    try:
                        img = self._load_image(zip_path, img_name)
                        
                        # Resize and normalize
                        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
                        img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
                        img_tensor = (img_tensor - self.mean) / self.std
                        
                        samples_yielded += 1
                        yield img_tensor
                        
                    except Exception as e:
                        if isinstance(e, IOError) and "BadZipFile" in str(e):
                            self.corrupted_zip_files.add(zip_path)
                            break
                            
                        if self.error_count < self.max_errors_per_minute:
                            print(f"Skipping corrupted image {img_name} from {zip_path}: {e}")
                            self.error_count += 1
                        continue
                        
            except Exception as e:
                if self.error_count < self.max_errors_per_minute:
                    print(f"Error processing zip file {zip_path}: {e}")
                    self.error_count += 1
                self.corrupted_zip_files.add(zip_path)
    
    def __len__(self):
        """Return number of samples this worker will process."""
        if not self.shard_calculated:
            # Return approximate total (will be divided among workers)
            return self.total_images
        
        total = 0
        for i, (start_idx, end_idx) in enumerate(self.worker_image_ranges):
            if i < len(self.worker_files) and self.worker_files[i] in self.corrupted_zip_files:
                continue
            total += (end_idx - start_idx)
        
        if self.max_samples:
            total = min(total, self.max_samples)
        
        return total