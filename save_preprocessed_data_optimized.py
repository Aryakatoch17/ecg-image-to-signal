import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
import random
import multiprocessing

# Import custom modules
from ecg_image_to_signal_trainer import ECGImageSignalDataset

class ECGDatasetWithID(ECGImageSignalDataset):
    """
    Wrapper to return file ID along with data.
    """
    def __getitem__(self, idx):
        # reuse parent method
        image_path, signal_path = self.samples[idx]
        
        # We need to manually load to get the ID, or just get it from path
        # reuse parent getitem to get processed data
        image, signal = super().__getitem__(idx)
        
        # Extract ID
        file_id = os.path.splitext(os.path.basename(image_path))[0]
        
        return image, signal, file_id

def save_batch(batch, save_dir_images, save_dir_signals):
    images, signals, file_ids = batch
    # This runs in the main process, but the batch preparation runs in workers
    for i in range(len(file_ids)):
        torch.save(images[i], os.path.join(save_dir_images, f"{file_ids[i]}.pt"))
        torch.save(signals[i], os.path.join(save_dir_signals, f"{file_ids[i]}.pt"))

def process_and_save(dataset, output_dir, split_name, num_workers=4, batch_size=32):
    """
    Process dataset using DataLoader and save to disk.
    """
    save_dir_images = os.path.join(output_dir, split_name, 'images')
    save_dir_signals = os.path.join(output_dir, split_name, 'signals')
    
    os.makedirs(save_dir_images, exist_ok=True)
    os.makedirs(save_dir_signals, exist_ok=True)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False
    )
    
    print(f"Processing {split_name} with {num_workers} workers...")
    
    for batch in tqdm(loader, desc=f"Saving {split_name}"):
        save_batch(batch, save_dir_images, save_dir_signals)

def main():
    # Configuration
    dataset_root = 'output'
    output_dir = 'preprocessed_data'
    image_size = (224, 224)
    signal_length = 5000
    
    # Use roughly 75% of available cores
    num_workers = max(1, int(os.cpu_count() * 0.75))
    batch_size = 32
    
    print(f"Preprocessing data from '{dataset_root}' to '{output_dir}'")
    
    # Clean output directory if it exists
    # user logic: if it exists, maybe we want to resume? 
    # But for safety/consistency, let's clean it or maybe just overwrite.
    # The previous script did clean it. Let's clean it to be sure.
    if os.path.exists(output_dir):
        # shutil.rmtree(output_dir)
        # print(f"Cleaned existing output directory: {output_dir}")
        print("Output directory exists. Overwriting files...")
    
    # 1. Dataset Discovery & Split
    image_dir = os.path.join(dataset_root, 'images')
    signal_dir = os.path.join(dataset_root, 'signals')
    
    all_samples = ECGImageSignalDataset._find_samples(image_dir, signal_dir)
    
    # Reproducible split
    random.seed(42)
    random.shuffle(all_samples)
    
    train_size = int(0.8 * len(all_samples))
    train_samples = all_samples[:train_size]
    val_samples = all_samples[train_size:]
    
    print(f"Found {len(all_samples)} total samples")
    
    # 2. Define Transform
    base_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Create Datasets (Wrapped)
    train_dataset = ECGDatasetWithID(
        image_dir=image_dir,
        signal_dir=signal_dir,
        image_size=image_size,
        signal_length=signal_length,
        transform=base_transform,
        augment=False, 
        samples=train_samples
    )
    
    val_dataset = ECGDatasetWithID(
        image_dir=image_dir,
        signal_dir=signal_dir,
        image_size=image_size,
        signal_length=signal_length,
        transform=base_transform,
        augment=False,
        samples=val_samples
    )
    
    # 4. Save Data
    # Optimization: Process Validation first (it's smaller) to verify it works quickly
    process_and_save(val_dataset, output_dir, 'val', num_workers, batch_size)
    process_and_save(train_dataset, output_dir, 'train', num_workers, batch_size)
    
    print("\nPreprocessing complete!")

if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    main()
