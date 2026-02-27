"""
ECG Image-to-Signal Reconstruction Training Pipeline
=====================================================
Deep learning model to digitize ECG images into 1D waveforms.

Author: Expert Deep Learning Researcher
Purpose: Robust ECG digitization for downstream sleep apnea analysis
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.stats import pearsonr
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')


# ==================== Dataset Implementation ====================

class ECGImageSignalDataset(Dataset):
    """
    Custom dataset for paired ECG images and signals.
    
    Architecture rationale:
    - Images are normalized to [0,1] and resized to fixed dimensions
    - Signals are interpolated to fixed length for batch processing
    - Supports data augmentation for robustness to noise
    """
    
    def __init__(self, image_dir, signal_dir, image_size=(224, 224), 
                 signal_length=5000, transform=None, augment=False, samples=None, cache_data=False):
        """
        Args:
            image_dir: Directory containing ECG images (.png)
            signal_dir: Directory containing ECG signals (.npy)
            image_size: Target image size (H, W)
            signal_length: Target signal length (for interpolation)
            transform: Optional image transforms
            augment: Whether to apply data augmentation
        """
        self.image_dir = image_dir
        self.signal_dir = signal_dir
        self.image_size = image_size
        self.signal_length = signal_length
        self.augment = augment
        self.cache_data = cache_data
        self.cache = {}
        
        if samples is not None:
            self.samples = samples
        else:
            self.samples = self._find_samples(image_dir, signal_dir)
            
        print(f"Dataset initialized with {len(self.samples)} samples")

        # Define image transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),  # Converts to [0,1] and (C, H, W)
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Optional augmentation for training robustness
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=2, translate=(0.02, 0.02)),
            ])
        else:
            self.augment_transform = None
            
        if self.cache_data:
            self._cache_dataset()


    @staticmethod
    def _find_samples(image_dir, signal_dir):
        # Find all signal files and match with images
        signal_files = sorted(glob.glob(os.path.join(signal_dir, "*_lr.npy")))
        samples = []
        
        for signal_path in signal_files:
            # Extract ID from signal filename (e.g., "00001_lr.npy" -> "00001")
            signal_id = os.path.basename(signal_path).split('_')[0]
            
            # Look for corresponding image
            image_path = os.path.join(image_dir, f"{signal_id}.png")
            if not os.path.exists(image_path):
                # Try alternative naming conventions
                image_path = os.path.join(image_dir, f"{signal_id}_lr.png")
            
            if os.path.exists(image_path):
                samples.append((image_path, signal_path))
        return samples
        

    
    def __len__(self):
        return len(self.samples)
    
    def _load_single_sample(self, idx):
        image_path, signal_path = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
        
        # Load signal
        try:
            signal = np.load(signal_path).astype(np.float32)
        except Exception as e:
            print(f"Error loading signal {signal_path}: {e}")
            return None
            
        return idx, image, signal

    def _cache_dataset(self):
        print(f"Caching {len(self.samples)} samples into RAM...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(self._load_single_sample, range(len(self.samples))), 
                              total=len(self.samples), desc="Caching"))
            
        # Store in dictionary
        for res in results:
            if res is not None:
                idx, image, signal = res
                self.cache[idx] = (image, signal)
                
        end_time = time.time()
        print(f"Cached {len(self.cache)} samples in {end_time - start_time:.2f} seconds")

    def __getitem__(self, idx):
        if self.cache_data and idx in self.cache:
            image, signal = self.cache[idx]
        else:
            # Fallback to slow loading if not cached
            image_path, signal_path = self.samples[idx]
            image = Image.open(image_path).convert('RGB')
            signal = np.load(signal_path).astype(np.float32)

        # Apply transforms
        if self.augment_transform:
            image = self.augment_transform(image)
        image = self.transform(image)
        
        # Handle variable-length signals via interpolation
        if len(signal) != self.signal_length:
            signal = self._interpolate_signal(signal, self.signal_length)
        
        # Normalize signal
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        signal = torch.from_numpy(signal)
        
        return image, signal
    
    @staticmethod
    def _interpolate_signal(signal, target_length):
        """Interpolate signal to target length using linear interpolation."""
        original_length = len(signal)
        original_indices = np.linspace(0, original_length - 1, original_length)
        target_indices = np.linspace(0, original_length - 1, target_length)
        interpolated = np.interp(target_indices, original_indices, signal)
        return interpolated.astype(np.float32)


# ==================== Model Architecture ====================

class ResidualConvBlock(nn.Module):
    """
    Residual block for 1D CNN decoder with skip connections.
    """
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ELU(inplace=True)
        
        # Skip connection with 1x1 conv if dimensions change
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        # Upsample input for skip connection
        if isinstance(self.skip, nn.Conv1d):
            # Upsample to match conv output size
            skip_out = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
            skip_out = self.skip(skip_out)
        else:
            skip_out = self.skip(x)
        
        out = self.conv(x)
        out = self.bn(out)
        out = out + skip_out  # Residual connection
        out = self.activation(out)
        return out


class LightweightTransformerBlock(nn.Module):
    """
    Lightweight transformer block optimized for small datasets.
    """
    def __init__(self, embed_dim, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class ECGImageToSignalModel(nn.Module):
    """
    Spatially-Aware Hybrid CNN + Transformer Architecture.
    
    Preserves spatial information using a transformer on image patches 
    before decoding to the temporal domain.
    """
    def __init__(self, signal_length=5000, pretrained=True):
        super().__init__()
        self.signal_length = signal_length
        self.embed_dim = 256
        self.hidden_dim = 256
        self.initial_length = 64
        
        # Stage 1: ResNet18 backbone
        # Remove avgpool and fc to keep spatial grid (7x7)
        resnet = models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Freeze early layers
        for name, param in self.backbone.named_parameters():
            if 'layer1' in name or 'layer2' in name or '0.' in name:
                param.requires_grad = False
                
        cnn_output_dim = 512
        self.num_spatial_tokens = 49  # 7x7
        
        # Stage 2: Projection & Positional Encoding
        self.input_projection = nn.Linear(cnn_output_dim, self.embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_spatial_tokens, self.embed_dim) * 0.02)
        
        # Stage 3: Spatial Transformer
        self.transformer_blocks = nn.ModuleList([
            LightweightTransformerBlock(self.embed_dim, num_heads=4, mlp_ratio=2.0)
            for _ in range(2)
        ])
        self.norm = nn.LayerNorm(self.embed_dim)
        
        # Stage 4: Temporal Projection
        # Flatten (B, 49, 256) -> (B, 49*256) -> (B, 256*64) -> (B, 256, 64)
        self.temporal_projection = nn.Sequential(
            nn.Linear(self.num_spatial_tokens * self.embed_dim, self.hidden_dim * self.initial_length),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Stage 5: 1D CNN Decoder
        # Upsampling: 64 -> 128 -> 256 -> 512 -> 1024 -> 2048 -> Signal Length
        self.decoder = nn.Sequential(
            # 64 -> 128
            ResidualConvBlock(self.hidden_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.Dropout(0.2),
            # 128 -> 256
            ResidualConvBlock(256, 128, kernel_size=4, stride=2, padding=1),
            nn.Dropout(0.2),
            # 256 -> 512
            ResidualConvBlock(128, 64, kernel_size=4, stride=2, padding=1),
            # 512 -> 1024
            ResidualConvBlock(64, 32, kernel_size=4, stride=2, padding=1),
            # 1024 -> 2048
            ResidualConvBlock(32, 16, kernel_size=4, stride=2, padding=1),
            # Final projection
            nn.Conv1d(16, 1, kernel_size=7, padding=3)
        )
        
        # Learnable output scaling
        self.output_scale = nn.Parameter(torch.tensor([3.0]))
        self.output_bias = nn.Parameter(torch.zeros(1))
        
        self._init_weights()

    def _init_weights(self):
        for m in self.decoder.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.decoder[-1].weight, gain=1.0)

    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. Feature Extraction (B, 512, 7, 7)
        features = self.backbone(x)
        
        # 2. Spatial Tokens (B, 49, 512)
        features = features.flatten(2).permute(0, 2, 1)
        
        # 3. Transformer Bridge
        features = self.input_projection(features)  # (B, 49, 256)
        features = features + self.pos_embedding
        for block in self.transformer_blocks:
            features = block(features)
        features = self.norm(features)
        
        # 4. Temporal Projection
        features_flat = features.reshape(batch_size, -1)
        temporal = self.temporal_projection(features_flat)
        temporal = temporal.view(batch_size, self.hidden_dim, self.initial_length)
        
        # 5. Decode
        signal = self.decoder(temporal)
        signal = signal.squeeze(1)
        
        # Interpolate to target length
        if signal.size(1) != self.signal_length:
            signal = F.interpolate(
                signal.unsqueeze(1), 
                size=self.signal_length, 
                mode='linear', 
                align_corners=False
            ).squeeze(1)
            
        return signal * self.output_scale + self.output_bias


# ==================== Loss Functions ====================

class CompositeLoss(nn.Module):
    """
    Composite loss combining MSE, Pearson Correlation, and Derivative Loss.
    """
    def __init__(self, mse_weight=1.0, pearson_weight=0.5, derivative_weight=0.1, dtw_weight=0.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.pearson_weight = pearson_weight
        self.derivative_weight = derivative_weight
        self.dtw_weight = dtw_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target):
        # MSE
        mse = self.mse_loss(pred, target)
        
        # Pearson
        pearson = self._batch_pearson_loss(pred, target)
        
        # Derivative (for R-peak sharpness)
        derivative = self._derivative_loss(pred, target)
        
        total_loss = (self.mse_weight * mse + 
                     self.pearson_weight * pearson + 
                     self.derivative_weight * derivative)
                     
        return total_loss, {
            'mse': mse.item(), 
            'pearson': pearson.item(),
            'derivative': derivative.item()
        }
    
    def _batch_pearson_loss(self, pred, target):
        pred_centered = pred - pred.mean(dim=1, keepdim=True)
        target_centered = target - target.mean(dim=1, keepdim=True)
        numerator = (pred_centered * target_centered).sum(dim=1)
        pred_std = torch.sqrt((pred_centered ** 2).sum(dim=1) + 1e-8)
        target_std = torch.sqrt((target_centered ** 2).sum(dim=1) + 1e-8)
        correlation = numerator / (pred_std * target_std)
        return 1.0 - correlation.mean()  # Minimize (1 - r)
    
    def _derivative_loss(self, pred, target):
        # Simple finite difference approximation
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        return F.mse_loss(pred_diff, target_diff)
    
    def _batch_dtw_loss(self, pred, target):
        return torch.abs(pred - target).mean()  # Placeholder


# ==================== Evaluation Metrics ====================

def compute_pearson_correlation(pred, target):
    """
    Compute Pearson correlation between predicted and target signals.
    
    Args:
        pred: [B, T] tensor
        target: [B, T] tensor
    Returns:
        mean_corr: Average correlation across batch
        corr_list: List of correlations for each sample
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    corr_list = []
    for i in range(pred_np.shape[0]):
        corr, _ = pearsonr(pred_np[i], target_np[i])
        corr_list.append(corr)
    
    return np.mean(corr_list), corr_list


def detect_r_peaks(signal, fs=500, distance=250):
    """
    Simple R-peak detector using scipy's find_peaks.
    
    Args:
        signal: 1D numpy array
        fs: Sampling frequency
        distance: Minimum distance between peaks (in samples)
    Returns:
        peaks: Indices of detected R-peaks
    """
    peaks, _ = scipy_signal.find_peaks(signal, distance=distance, prominence=0.5)
    return peaks


def compute_r_peak_error(pred, target, fs=500):
    """
    Compute R-peak detection error between predicted and target signals.
    
    Metric: Average time error (in ms) between matched peaks.
    
    Args:
        pred: [B, T] tensor
        target: [B, T] tensor
        fs: Sampling frequency (Hz)
    Returns:
        mean_error: Average R-peak error in milliseconds
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    errors = []
    for i in range(pred_np.shape[0]):
        pred_peaks = detect_r_peaks(pred_np[i], fs=fs)
        target_peaks = detect_r_peaks(target_np[i], fs=fs)
        
        if len(pred_peaks) == 0 or len(target_peaks) == 0:
            continue
        
        # Match peaks using nearest neighbor
        for tp in target_peaks:
            if len(pred_peaks) > 0:
                nearest_pred = pred_peaks[np.argmin(np.abs(pred_peaks - tp))]
                error_samples = abs(nearest_pred - tp)
                error_ms = (error_samples / fs) * 1000  # Convert to milliseconds
                errors.append(error_ms)
    
    return np.mean(errors) if errors else float('inf')


# ==================== Visualization ====================

def plot_comparison(pred_signals, target_signals, num_samples=4, save_path=None):
    """
    Plot original vs reconstructed ECG waveforms.
    
    Args:
        pred_signals: [B, T] tensor
        target_signals: [B, T] tensor
        num_samples: Number of random samples to plot
        save_path: Path to save the figure
    """
    pred_np = pred_signals.detach().cpu().numpy()
    target_np = target_signals.detach().cpu().numpy()
    
    num_samples = min(num_samples, pred_np.shape[0])
    indices = np.random.choice(pred_np.shape[0], num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        axes[i].plot(target_np[idx], label='Ground Truth', linewidth=1.5, alpha=0.7)
        axes[i].plot(pred_np[idx], label='Predicted', linewidth=1.5, alpha=0.7)
        
        # Compute metrics for this sample
        corr, _ = pearsonr(pred_np[idx], target_np[idx])
        mse = np.mean((pred_np[idx] - target_np[idx]) ** 2)
        
        axes[i].set_title(f'Sample {idx} | Correlation: {corr:.4f} | MSE: {mse:.4f}')
        axes[i].set_xlabel('Time (samples)')
        axes[i].set_ylabel('Amplitude')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    plt.show()


# ==================== Training Pipeline ====================

class ECGTrainer:
    """
    Complete training pipeline with checkpointing, early stopping, and evaluation.
    """
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 device, checkpoint_dir='checkpoints', patience=10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.patience = patience
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_pearson': [],
            'val_r_peak_error': []
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_mse = 0
        total_pearson = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for images, signals in pbar:
            images = images.to(self.device)
            signals = signals.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                pred_signals = self.model(images)
                loss, loss_dict = self.criterion(pred_signals, signals)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            total_mse += loss_dict['mse']
            total_pearson += loss_dict['pearson']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mse': f"{loss_dict['mse']:.4f}",
                'pearson': f"{loss_dict['pearson']:.4f}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_mse = total_mse / len(self.train_loader)
        avg_pearson = total_pearson / len(self.train_loader)
        
        return avg_loss, avg_mse, avg_pearson
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        all_pred = []
        all_target = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for images, signals in pbar:
                images = images.to(self.device)
                signals = signals.to(self.device)
                
                with autocast():
                    pred_signals = self.model(images)
                    loss, loss_dict = self.criterion(pred_signals, signals)
                
                total_loss += loss.item()
                total_mse += loss_dict['mse']
                
                all_pred.append(pred_signals)
                all_target.append(signals)
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Concatenate all predictions and targets
        all_pred = torch.cat(all_pred, dim=0)
        all_target = torch.cat(all_target, dim=0)
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        avg_mse = total_mse / len(self.val_loader)
        pearson_corr, _ = compute_pearson_correlation(all_pred, all_target)
        r_peak_error = compute_r_peak_error(all_pred, all_target)
        
        print(f"\n[Validation] Loss: {avg_loss:.4f} | MSE: {avg_mse:.4f} | "
              f"Pearson: {pearson_corr:.4f} | R-peak Error: {r_peak_error:.2f} ms")
        
        return avg_loss, pearson_corr, r_peak_error, all_pred, all_target
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model with val_loss: {val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def train(self, num_epochs, resume_from=None):
        """
        Full training loop with early stopping.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
        """
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from)
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(start_epoch, num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_mse, train_pearson = self.train_epoch(epoch + 1)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, val_pearson, val_r_peak_error, pred, target = self.validate(epoch + 1)
            self.history['val_loss'].append(val_loss)
            self.history['val_pearson'].append(val_pearson)
            self.history['val_r_peak_error'].append(val_r_peak_error)
            
            # Checkpointing
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            self.save_checkpoint(epoch + 1, val_loss, is_best)
            
            # Plot every 5 epochs
            if (epoch + 1) % 5 == 0:
                plot_path = os.path.join(self.checkpoint_dir, f'epoch_{epoch+1}_comparison.png')
                plot_comparison(pred, target, num_samples=4, save_path=plot_path)
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\n⚠ Early stopping triggered after {self.patience} epochs without improvement")
                break
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot training and validation curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Pearson correlation
        axes[0, 1].plot(self.history['val_pearson'], label='Val Pearson', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Pearson Correlation')
        axes[0, 1].set_title('Validation Pearson Correlation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # R-peak error
        axes[1, 0].plot(self.history['val_r_peak_error'], label='R-peak Error', color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Error (ms)')
        axes[1, 0].set_title('Validation R-peak Error')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Remove empty subplot
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.checkpoint_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved training curves to {save_path}")
        plt.show()


# ==================== Main Function ====================

def main(dataset_root='output', image_size=(224, 224), signal_length=5000,
         batch_size=16, num_epochs=100, learning_rate=1e-4, num_workers=4,
         checkpoint_dir='checkpoints', resume_from=None):
    """
    Main training function.
    
    Args:
        dataset_root: Root directory containing 'images/' and 'signals/' subdirs
        image_size: Target image size (H, W)
        signal_length: Target signal length
        batch_size: Batch size for training
        num_epochs: Number of epochs to train
        learning_rate: Initial learning rate
        num_workers: Number of data loading workers
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
    """
    
    print("="*60)
    print("ECG Image-to-Signal Reconstruction Training")
    print("="*60)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Dataset paths
    image_dir = os.path.join(dataset_root, 'images')
    signal_dir = os.path.join(dataset_root, 'signals')
    
    print(f"\nDataset configuration:")
    print(f"Image directory: {image_dir}")
    print(f"Signal directory: {signal_dir}")
    print(f"Image size: {image_size}")
    print(f"Signal length: {signal_length}")
    
    # Create datasets with explicit separation
    print("\nPreparing datasets...")
    # Find all samples first
    all_samples = ECGImageSignalDataset._find_samples(image_dir, signal_dir)
    
    # Shuffle and split
    import random
    random.seed(42)
    random.shuffle(all_samples)
    
    train_size = int(0.8 * len(all_samples))
    train_samples = all_samples[:train_size]
    val_samples = all_samples[train_size:]
    
    # Define transforms
    # 1. Base transform (Tensor + Normalize)
    base_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Train transform (Include RandomErasing after normalize)
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3))
    ])
    
    # Train Dataset
    train_dataset = ECGImageSignalDataset(
        image_dir=image_dir,
        signal_dir=signal_dir,
        image_size=image_size,
        signal_length=signal_length,
        transform=train_transform,
        augment=True,
        samples=train_samples
    )
    
    # Robust Augmentation (PIL level)
    train_dataset.augment_transform = transforms.Compose([
        # Geometric transformations
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        
        # Photometric transformations
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    ])
    
    # Validation Dataset (Clean)
    val_dataset = ECGImageSignalDataset(
        image_dir=image_dir,
        signal_dir=signal_dir,
        image_size=image_size,
        signal_length=signal_length,
        transform=base_transform,
        augment=False,
        samples=val_samples
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = ECGImageToSignalModel(signal_length=signal_length, pretrained=True)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = CompositeLoss(mse_weight=1.0, pearson_weight=0.5, dtw_weight=0.0)
    
    # Optimizer with higher weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize trainer
    trainer = ECGTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        patience=15
    )
    
    # Train
    trainer.train(num_epochs=num_epochs, resume_from=resume_from)
    
    print("\n" + "="*60)
    print("Training pipeline completed successfully!")
    print("="*60)


if __name__ == '__main__':
    # Example usage with your dataset structure
    main(
        dataset_root='output',  # Root directory containing images/ and signals/
        image_size=(224, 224),
        signal_length=5000,
        batch_size=16,
        num_epochs=100,
        learning_rate=1e-4,
        num_workers=4,
        checkpoint_dir='checkpoints',
        resume_from=None  # Set to checkpoint path to resume training
    )
