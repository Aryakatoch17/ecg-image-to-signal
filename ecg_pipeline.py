import os
import glob
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for faster rendering
import matplotlib.pyplot as plt
import cv2
import wfdb
import ast
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

class ECGPreprocessor:
    """
    Handles loading, filtering, and plotting of ECG signals.
    """
    def __init__(self, target_sr=100):
        self.target_sr = target_sr
        # Pre-calculate filter coefficients for reuse
        self._filter_cache = {}
        
    def load_signal(self, file_path, lead='II'):
        """
        Loads ECG signal from WFDB file.
        Returns the signal array for the specified lead and the sampling rate.
        """
        try:
            # record_name = file_path.replace('.hea', '').replace('.dat', '')
            # wfdb.rdsamp expects the path without extension
            record_path = os.path.splitext(file_path)[0]
            signals, fields = wfdb.rdsamp(record_path)
            
            # Find lead index
            if lead in fields['sig_name']:
                lead_idx = fields['sig_name'].index(lead)
            else:
                # Fallback to index 1 (often Lead II in standard 12-lead) if available, else 0
                print(f"Warning: Lead {lead} not found in {fields['sig_name']}. Using index 1.")
                lead_idx = 1 if len(fields['sig_name']) > 1 else 0
            
            ecg_signal = signals[:, lead_idx]
            fs = fields['fs']
            
            return ecg_signal, fs
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None

    def process_signal(self, ecg_signal, fs):
        """
        Applies bandpass filtering and normalization.
        Filter: 0.5-40 Hz Butterworth bandpass.
        Normalization: Z-score or MinMax. Here we map to physical units if possible, 
        but for image generation, we often want to center it. 
        However, keeping amplitude meaningful (mV) is important for the grid.
        """
        # Remove baseline wander and high freq noise
        # 0.5 Hz highpass, 40 Hz lowpass
        lowcut = 0.5
        highcut = 40.0
        
        # Cache filter coefficients based on sampling rate
        if fs not in self._filter_cache:
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            # 4th order Butterworth filter
            b, a = signal.butter(4, [low, high], btype='band')
            self._filter_cache[fs] = (b, a)
        
        b, a = self._filter_cache[fs]
        filtered_ecg = signal.filtfilt(b, a, ecg_signal)
        
        # Note: We do NOT normalize to [0,1] here because we want to preserve 
        # the mV amplitude relationship for the plot grid (10mm/mV).
        # However, we might want to center the baseline to 0.
        filtered_ecg -= np.mean(filtered_ecg)  # In-place operation
        
        return filtered_ecg

    def create_ecg_image(self, ecg_signal, fs, output_file):
        """
        Plots the ECG signal with a standard ECG grid.
        
        Standards:
        - Time: 25 mm/s
        - Voltage: 10 mm/mV
        
        Grid:
        - Minor grid: 1 mm squares (0.04s x 0.1mV)
        - Major grid: 5 mm squares (0.2s x 0.5mV)
        """
        # Config for output image resolution
        dpi = 200
        
        # Calculate duration and required width in inches
        duration_sec = len(ecg_signal) / fs
        
        # Width calculation:
        # We need 25 mm per second. 
        # 25 mm = 0.984 inches.
        # Width_in_inches = duration_sec * (25mm/s) * (1 inch / 25.4 mm)
        width_inch = duration_sec * (25 / 25.4)
        
        # Height calculation:
        # Standard ECG paper height is often fixed, but let's base it on signal range.
        # Let's accommodate +/- 2 mV range typical. 4 mV total.
        # 4 mV * 10 mm/mV = 40 mm height.
        # Let's add padding. Say 60 mm total height.
        height_mm = 60
        height_inch = height_mm / 25.4
        
        # Create figure with exact size
        fig = plt.figure(figsize=(width_inch, height_inch), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])  # Full span
        
        # Pre-calculate time array for plotting
        t = np.arange(len(ecg_signal), dtype=np.float32) / fs
        
        # Grid settings
        # Minor ticks: 1 mm. Major ticks: 5 mm.
        # X axis: 1 sec = 25 mm. 
        # 1 mm = 1/25 = 0.04 sec.
        # 5 mm = 0.2 sec.
        
        # Y axis: 1 mV = 10 mm.
        # 1 mm = 0.1 mV.
        # 5 mm = 0.5 mV.
        
        # Set limits
        # Center Y around 0
        y_center = 0.0
        y_range_mv = (height_mm / 10.0) # total mV height
        y_min = y_center - y_range_mv / 2.0
        y_max = y_center + y_range_mv / 2.0
        
        ax.set_xlim(0, duration_sec)
        ax.set_ylim(y_min, y_max)
        
        # Setup Grid
        # Major ticks
        major_ticks_x = np.arange(0, duration_sec + 0.1, 0.2)
        major_ticks_y = np.arange(np.floor(y_min), np.ceil(y_max), 0.5)
        
        ax.set_xticks(major_ticks_x)
        ax.set_yticks(major_ticks_y)
        
        # Minor ticks
        minor_ticks_x = np.arange(0, duration_sec + 0.02, 0.04)
        minor_ticks_y = np.arange(np.floor(y_min), np.ceil(y_max), 0.1)
        
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(minor_ticks_y, minor=True)
        
        # Grid Styling
        # Pink/Red grid on white background
        ax.grid(which='major', color='red',  linewidth=0.8, alpha=0.5)
        ax.grid(which='minor', color='red',  linewidth=0.3, alpha=0.3)
        
        # Plot Signal
        # Standard ECG trace is black or blue.
        ax.plot(t, ecg_signal, color='black', linewidth=1.2, antialiased=False)
        
        # Remove axes borders/numbers for pure image (we want just grid + signal)
        # But keeping the grid lines is key.
        ax.tick_params(axis='both', which='both', length=0, labelsize=0)
        # Hide standard spines
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        # Draw explicit box if needed, but grid usually suffices.
        
        # Save to buffer/file with optimization
        plt.savefig(output_file, dpi=dpi, pad_inches=0, format='png', 
                   pil_kwargs={'optimize': True, 'compress_level': 6})
        plt.close(fig)

class ImageAugmenter:
    """
    Applies realistic camera distortions to ECG images.
    """
    @staticmethod
    def augment(image_path):
        """
        Reads image, applies transform, returns augmented image.
        """
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return None
            
        rows, cols = img.shape[:2]  # More efficient unpacking
        
        # 1. Random Rotation (+/- 5 degrees)
        angle = random.uniform(-5, 5)
        M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        # Verify valid border handling (white padding)
        img = cv2.warpAffine(img, M_rot, (cols, rows), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        
        # 2. Perspective Warp (Simulate angled photo)
        # Source points: corners
        pts1 = np.float32([[0,0], [cols,0], [0,rows], [cols,rows]])
        
        # Destination points: Shift corners slightly
        # max shift 5% of dimension
        shift_x = cols * 0.05
        shift_y = rows * 0.05
        
        pts2 = np.float32([
            [random.uniform(0, shift_x), random.uniform(0, shift_y)],
            [cols - random.uniform(0, shift_x), random.uniform(0, shift_y)],
            [random.uniform(0, shift_x), rows - random.uniform(0, shift_y)],
            [cols - random.uniform(0, shift_x), rows - random.uniform(0, shift_y)]
        ])
        
        M_persp = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M_persp, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        
        # 3. Gaussian Blur (Simulate out of focus)
        ksize = random.choice([1, 3]) # Keep it subtle
        if ksize > 1:
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
            
        # 4. Brightness / Contrast Jitter
        # alpha = contrast [0.8, 1.2], beta = brightness [-30, 30]
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-20, 20)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # 5. JPEG Compression Artifacts
        # Encode with quality factor 50-95 and decode back
        quality = random.randint(60, 95)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality,
                       int(cv2.IMWRITE_JPEG_OPTIMIZE), 1]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        if result:
            decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
            return decimg
        return img

def process_single_record(args):
    """
    Process a single ECG record. Designed for multiprocessing.
    """
    record_path, OUTPUT_DIR_IMG, OUTPUT_DIR_SIG = args
    
    try:
        preprocessor = ECGPreprocessor(target_sr=100)
        augmenter = ImageAugmenter()
        
        filename = os.path.basename(record_path).replace('.hea', '')
        
        # 1. Load Signal
        signal_raw, fs = preprocessor.load_signal(record_path, lead='II')
        
        if signal_raw is None:
            return False
            
        # 2. Preprocess
        signal_proc = preprocessor.process_signal(signal_raw, fs)
        
        # 3. Generate Image
        # Save as .png (lossless) or .jpg (with high quality)
        img_path = os.path.join(OUTPUT_DIR_IMG, f"{filename}.png") 
        signal_out_path = os.path.join(OUTPUT_DIR_SIG, f"{filename}.npy")
        
        preprocessor.create_ecg_image(signal_proc, fs, img_path)
        
        # 4. Optional: Save Augmented Version (for visualization or separate test set)
        # For now, we ONLY want clean images for the training set to let the DataLoader handle augmentation.
        # If we wanted to keep the old behavior, we would rename this.
        # passed
        
        # 5. Save Signal with compression
        np.save(signal_out_path, signal_proc.astype(np.float32))
        
        return True
        
    except Exception as e:
        return False

def main():
    # Configuration
    DATASET_ROOT = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
    INPUT_DIR = os.path.join(DATASET_ROOT, 'records100')
    OUTPUT_DIR_IMG = 'output/images'
    OUTPUT_DIR_SIG = 'output/signals'
    
    # Set this to a number (e.g., 10) to test on a small subset, or None to process all
    # With multiprocessing on 8 cores: ~22,000 records takes approximately 1-2 hours
    MAX_RECORDS = None  # Process all records
    
    # Number of parallel workers (use all available CPU cores)
    NUM_WORKERS = cpu_count()
    
    os.makedirs(OUTPUT_DIR_IMG, exist_ok=True)
    os.makedirs(OUTPUT_DIR_SIG, exist_ok=True)
    
    # Load database info just to get filenames if we want to follow the CSV, 
    # but recursively searching files is more robust if CSV is missing or partial.
    # Searching for header files (.hea)
    print(f"Scanning for ECG records in {INPUT_DIR}...")
    record_files = []
    # Using glob to find .hea files recursively
    # ptb-xl structure: records100/00000/00001_lr.hea
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith('.hea'):
                record_files.append(os.path.join(root, file))
    
    # Process limit
    if MAX_RECORDS is not None:
        record_files = record_files[:MAX_RECORDS]
    
    print(f"Found {len(record_files)} records. Starting processing with {NUM_WORKERS} parallel workers...")
    
    # Prepare arguments for multiprocessing
    process_args = [(record_path, OUTPUT_DIR_IMG, OUTPUT_DIR_SIG) for record_path in record_files]
    
    # Use multiprocessing Pool to process records in parallel
    with Pool(NUM_WORKERS) as pool:
        # Use imap_unordered for better performance with progress bar
        results = list(tqdm(pool.imap_unordered(process_single_record, process_args), 
                           total=len(record_files),
                           desc="Processing ECG records"))
    
    # Summary
    successful = sum(results)
    failed = len(results) - successful
    print(f"\nProcessing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()
