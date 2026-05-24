"""
Training script for AcousticEmotionModel (CNN+LSTM) using RAVDESS dataset format.

This script:
1. Loads audio files from RAVDESS dataset directory
2. Extracts 2D Mel-Spectrograms (128 mel bands, dB scale)
3. Trains CNN+LSTM model for 5 emotion classes
4. Saves trained model to backend/models/emotion_model.pth
"""
import logging
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional
from collections import Counter
# Add parent directory to path to import call_analysis modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.call_analysis.models import AcousticEmotionModel
from src.call_analysis.feature_extraction import FeatureExtractor, normalize_mel_spectrogram, apply_specaugment

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # This forces output to Colab's window
)
logger = logging.getLogger(__name__)

# Add this class here:
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.
    Helps prevent overfitting by preventing overconfident predictions.
    """
    def __init__(self, smoothing=0.1, weight=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, pred, target):
        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        nll_loss = -log_prob.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_prob.mean(dim=1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        
        if self.weight is not None:
            loss = loss * self.weight[target]
        
        return loss.mean()

class EmotionDataset(Dataset):
    """PyTorch Dataset for emotion recognition from audio files."""
    def __init__(self, audio_files: List[str], labels: List[int], feature_extractor: FeatureExtractor, 
                 max_time_frames: int = 500, use_specaugment: bool = False,
                 normalization_method: str = 'cmvn', normalization_stats: Optional[Dict] = None):
        """
        Initialize dataset.
        
        Args:
            audio_files: List of paths to audio files
            labels: List of emotion labels (0-4)
            feature_extractor: FeatureExtractor instance
            max_time_frames: Maximum time frames for padding/truncation (default: 500 ≈ 14.4s at 16kHz)
            use_specaugment: Whether to apply SpecAugment during training (default: False)
            normalization_method: Normalization method ('cmvn', 'zscore', 'logmel', 'minmax')
            normalization_stats: Optional statistics dict with 'mean', 'std', 'min', 'max' keys
        """ 
        self.audio_files = audio_files
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.max_time_frames = max_time_frames
        self.emotion_labels = ['neutral', 'happiness', 'anger', 'sadness', 'frustration']
        self.use_specaugment = use_specaugment
        self.normalization_method = normalization_method
        self.normalization_stats = normalization_stats or {}    
    

    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """Load and preprocess audio file."""
        audio_path = self.audio_files[idx]
        label = self.labels[idx]  
        try:
            # Extract mel-spectrogram and MFCC using FeatureExtractor
            audio_features = self.feature_extractor.load_audio_features(audio_path, n_mfcc=40)

            # Robust guard: if feature extraction failed or mel_spectrogram is missing,
            # fall back to a zero-valued spectrogram instead of crashing training.
            if not isinstance(audio_features, dict) or 'mel_spectrogram' not in audio_features:
                logger.error(
                    f"Error loading {audio_path}: missing 'mel_spectrogram' in features. "
                    "Using zero-valued fallback spectrogram."
                )
                n_mels = 128
                time_frames = self.max_time_frames
                mel_spec = np.zeros((n_mels, time_frames), dtype=np.float32)
                mfcc = None
            else:
                mel_spec = audio_features['mel_spectrogram']  # (n_mels, time_frames)
                mfcc = audio_features.get('mfcc', None)  # (n_mfcc, time_frames)
                # Handle variable-length audio: pad or truncate to max_time_frames
                n_mels, time_frames = mel_spec.shape
            
            if time_frames > self.max_time_frames:
                # Truncate
                mel_spec = mel_spec[:, :self.max_time_frames]
                if mfcc is not None:
                    mfcc = mfcc[:, :self.max_time_frames]
            elif time_frames < self.max_time_frames:
                # Pad with dataset-aware padding (use minimum value instead of zeros)
                # This prevents injecting fake spectral patterns
                pad_value = (
                    self.normalization_stats.get('min', mel_spec.min())
                    if self.normalization_method == 'minmax'
                    else mel_spec.min()
                )
                padding = np.full((n_mels, self.max_time_frames - time_frames), pad_value)
                mel_spec = np.concatenate([mel_spec, padding], axis=1)
                if mfcc is not None:
                    mfcc_pad_value = mfcc.min()
                    mfcc_padding = np.full((mfcc.shape[0], self.max_time_frames - time_frames), mfcc_pad_value)
                    mfcc = np.concatenate([mfcc, mfcc_padding], axis=1)
            
            # Apply normalization using selected method
            # CMVN is default (best for speaker-independent SER)
            mel_spec = normalize_mel_spectrogram(mel_spec, self.normalization_method, self.normalization_stats)

            # Apply SpecAugment augmentation (only during training)
            # This is more realistic than Gaussian noise for acoustic data
            if self.use_specaugment:
                mel_spec = apply_specaugment(mel_spec, time_masks=2, freq_masks=2,
                                            time_mask_size=27, freq_mask_size=13)
            
            # Convert to tensor: (1, n_mels, time_frames) - add channel dimension
            mel_spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)
            
            # Convert MFCC to tensor if available
            if mfcc is not None:
                mfcc_tensor = torch.FloatTensor(mfcc)  # (n_mfcc, time_frames)
            else:
                mfcc_tensor = None
            
            # Return length for LSTM masking (actual time frames before padding)
            actual_length = min(time_frames, self.max_time_frames)
            
            return mel_spec_tensor, mfcc_tensor, label, actual_length
            
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")
            # Fail loudly instead of silently returning neutral label
            # This prevents poisoning the dataset with fake data
            raise RuntimeError(f"Failed to load audio file: {audio_path}") from e

def get_emotion_mapping(mapping_type: str = 'default') -> Dict[int, Tuple[int, str]]:
    """
    Get emotion mapping configuration.
    
    RAVDESS original 8 emotions:
    01=neutral, 02=calm, 03=happiness, 04=sadness, 05=anger, 
    06=fear, 07=disgust, 08=surprise
    
    Our 5 target classes: neutral, happiness, anger, sadness, frustration
    
    Mapping rationale (see EMOTION_MAPPING_JUSTIFICATION.md for details):
    - calm→neutral: Low arousal, neutral valence (Russell's circumplex model)
    - fear/disgust→frustration: High arousal, negative valence, similar acoustic patterns
    - surprise→happiness: High arousal, positive valence (closest match)
    
    References:
    - Russell, J. A. (1980). A circumplex model of affect. Journal of Personality and Social Psychology
    - Prior SER work: Similar emotion grouping used in IEMOCAP and other datasets
    
    Args:
        mapping_type: 'default', 'strict', or 'expanded'
            - 'default': 5-class mapping (calm→neutral, fear/disgust→frustration, surprise→happiness)
            - 'strict': Keep original 8 classes (no mapping)
            - 'expanded': Alternative grouping for experimentation
    
    Returns:
        Dictionary mapping RAVDESS emotion codes to (index, name) tuples
    """
    if mapping_type == 'default':
        return {
            1: (0, 'neutral'),      # neutral
            2: (0, 'neutral'),       # calm → neutral (low arousal, neutral valence)
            3: (1, 'happiness'),     # happiness
            4: (3, 'sadness'),       # sadness
            5: (2, 'anger'),         # anger
            6: (4, 'frustration'),   # fear → frustration (high arousal, negative valence)
            7: (4, 'frustration'),   # disgust → frustration (high arousal, negative valence)
            8: (1, 'happiness'),     # surprise → happiness (high arousal, positive valence)
        }
    elif mapping_type == 'strict':
        # Keep all 8 original classes
        return {
            1: (0, 'neutral'),
            2: (1, 'calm'),
            3: (2, 'happiness'),
            4: (3, 'sadness'),
            5: (4, 'anger'),
            6: (5, 'fear'),
            7: (6, 'disgust'),
            8: (7, 'surprise'),
        }
    elif mapping_type == 'expanded':
        # Alternative: Group by valence-arousal dimensions
        return {
            1: (0, 'neutral'),      # neutral
            2: (0, 'neutral'),       # calm → neutral
            3: (1, 'happiness'),     # happiness
            4: (2, 'sadness'),       # sadness (separate from frustration)
            5: (3, 'anger'),         # anger
            6: (4, 'frustration'),   # fear → frustration
            7: (4, 'frustration'),   # disgust → frustration
            8: (1, 'happiness'),     # surprise → happiness
        }
    else:
        logger.warning(f"Unknown mapping type '{mapping_type}', using 'default'")
        return get_emotion_mapping('default')

def parse_ravdess_filename(filename: str, emotion_mapping: Optional[Dict[int, Tuple[int, str]]] = None) -> Tuple[int, str]:
    """
    Parse RAVDESS filename to extract emotion label.
    
    Format: 03-01-01-01-01-01-01.wav
    Digits represent: Modality-Vocal-Channel-Emotion-Intensity-Statement-Repetition-Actor
    (Emotion is the 4th field → parts[3], but RAVDESS uses parts[2] as emotion code)
    
    Args:
        filename: RAVDESS filename
        emotion_mapping: Optional custom mapping dictionary. If None, uses default mapping.
        
    Returns:
        Tuple of (emotion_index, emotion_name)
    """
    if emotion_mapping is None:
        emotion_mapping = get_emotion_mapping('default')
    
    try:
        parts = filename.replace('.wav', '').split('-')
        if len(parts) < 4:
            return 0, 'neutral'  # Default to neutral
        
        emotion_code = int(parts[2])  # Emotion is the 3rd field (index 2)
        
        emotion_idx, emotion_name = emotion_mapping.get(emotion_code, (0, 'neutral'))
        return emotion_idx, emotion_name
        
    except Exception as e:
        logger.warning(f"Could not parse filename {filename}: {e}")
        return 0, 'neutral'

def extract_actor_id(filename: str) -> int:
    """
    Extract actor ID from RAVDESS filename.
    
    Format: 03-01-01-01-01-01-01.wav
    Actor ID is the last field (parts[6])
    
    Args:
        filename: RAVDESS filename
        
    Returns:
        Actor ID (1-24)
    """
    try:
        parts = filename.replace('.wav', '').split('-')
        if len(parts) < 7:
            return 0  # Default/unknown
        actor_id = int(parts[6])  # Last field is actor ID
        return actor_id
    except Exception as e:
        logger.warning(f"Could not extract actor ID from {filename}: {e}")
        return 0

def load_ravdess_dataset(data_dir: str, emotion_mapping_type: str = 'default') -> Tuple[List[str], List[int], List[float], List[int]]:
    """
    Load RAVDESS dataset with configurable emotion mapping.
    
    Args:
        data_dir: Path to RAVDESS dataset directory
        emotion_mapping_type: 'default', 'strict', or 'expanded' (see get_emotion_mapping)
    
    Returns:
        Tuple of (audio_files, labels, class_weights, actor_ids)
    """
    audio_files = []
    labels = []
    actor_ids = []
    
    # Get emotion mapping
    emotion_mapping = get_emotion_mapping(emotion_mapping_type)
    logger.info(f"Using emotion mapping: {emotion_mapping_type}")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Dataset directory does not exist: {data_dir}")
    
    wav_files = list(data_path.rglob('*.wav'))
    
    if len(wav_files) == 0:
        raise ValueError(f"No .wav files found in {data_dir}")
    
    # --- ADD THIS LINE TO BREAK THE BUFFER ---
    print(f"DEBUG: Found {len(wav_files)} files. Starting label parsing...")
    sys.stdout.flush() 

    for i, wav_file in enumerate(wav_files):
        # --- ADD THIS PROGRESS TRACKER ---
        if i % 100 == 0:
            print(f"Indexing file {i} of {len(wav_files)}...")
            sys.stdout.flush()

        filename = wav_file.name
        emotion_idx, emotion_name = parse_ravdess_filename(filename, emotion_mapping)
        actor_id = extract_actor_id(filename)
        audio_files.append(str(wav_file))
        labels.append(emotion_idx)
        actor_ids.append(actor_id)
        # Log class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    emotion_names = ['neutral', 'happiness', 'anger', 'sadness', 'frustration']
    logger.info("Class distribution:")
    for label, count in zip(unique_labels, counts):
        logger.info(f"  {emotion_names[label]}: {count}")

    # Calculate class weights for imbalanced dataset
    label_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = 5

    # Use stronger class weighting to prevent model collapse on minority classes
    # Strategy: Inverse frequency with power 1.5 for stronger emphasis on rare classes
    logger.info("Calculating class weights using inverse frequency weighting (power=1.5)")
    class_weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        # Stronger weighting: (total / (num_classes * count))^1.5
        # This gives more weight to underrepresented classes
        weight = (total_samples / (num_classes * count)) ** 1.5
        class_weights.append(weight)
    
    # Normalize to mean=1.0 to maintain loss scale
    weight_sum = sum(class_weights)
    class_weights = [w * num_classes / weight_sum for w in class_weights]
    
    # Log detailed class weight information
    logger.info(f"Class distribution: {dict(zip(emotion_names, [label_counts.get(i, 0) for i in range(num_classes)]))}")
    logger.info(f"Class weights (normalized): {dict(zip(emotion_names, [round(w, 3) for w in class_weights]))}")
    logger.info(f"Class weight ratios (relative to max): {dict(zip(emotion_names, [round(w/max(class_weights), 3) for w in class_weights]))}")
    
    # Warn if weights are too imbalanced
    min_weight = min(class_weights)
    max_weight = max(class_weights)
    if max_weight / min_weight > 3.0:
        logger.warning(f"Large class weight imbalance detected (ratio: {max_weight/min_weight:.2f}). "
                      f"Consider using focal loss or stronger regularization.")

    return audio_files, labels, class_weights, actor_ids

def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        # Handle formats: (mel, mfcc, target, lengths) or (mel, target, lengths) or (mel, target)
        if len(batch) == 4:
            mel_data, mfcc_data, target, lengths = batch
            lengths = lengths.to(device)
            mfcc_data = mfcc_data.to(device) if mfcc_data is not None else None
        elif len(batch) == 3:
            mel_data, target, lengths = batch
            lengths = lengths.to(device)
            mfcc_data = None
        else:
            mel_data, target = batch
            lengths = None
            mfcc_data = None
        
        mel_data, target = mel_data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(mel_data, mfcc=mfcc_data, lengths=lengths)
        loss = criterion(output, target)
        loss.backward()
        
        # DEBUG: Check gradient flow (only on first batch of first epoch)
        if batch_idx == 0:
            total_grad_norm = 0.0
            param_count = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.data.norm(2)
                    total_grad_norm += param_grad_norm.item() ** 2
                    param_count += 1
                    if param_count <= 3:  # Print first 3 layers
                        logger.info(f"DEBUG: {name} grad norm: {param_grad_norm.item():.6f}")
            total_grad_norm = total_grad_norm ** (1. / 2)
            logger.info(f"DEBUG: Total gradient norm: {total_grad_norm:.6f}")
        
        # Clip gradients to prevent explosion (tighter control for stability)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if batch_idx == 0:
            logger.info(f"DEBUG: Gradient norm after clipping: {grad_norm:.6f}")
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Log progress every 10 batches or at the end
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            current_loss = total_loss / (batch_idx + 1)
            current_acc = 100.0 * correct / total
            # Log prediction distribution to detect collapse early
            pred_dist = np.bincount(pred.cpu().numpy(), minlength=5)
            logger.info(f"  Batch {batch_idx + 1}/{num_batches} - Loss: {current_loss:.4f}, Acc: {current_acc:.2f}%")
            logger.info(f"    Prediction distribution: {pred_dist}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                   device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Handle formats: (mel, mfcc, target, lengths) or (mel, target, lengths) or (mel, target)
            if len(batch) == 4:
                mel_data, mfcc_data, target, lengths = batch
                lengths = lengths.to(device)
                mfcc_data = mfcc_data.to(device) if mfcc_data is not None else None
            elif len(batch) == 3:
                mel_data, target, lengths = batch
                lengths = lengths.to(device)
                mfcc_data = None
            else:
                mel_data, target = batch
                lengths = None
                mfcc_data = None
            
            mel_data, target = mel_data.to(device), target.to(device)
            output = model(mel_data, mfcc=mfcc_data, lengths=lengths)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # Log progress every 5 batches or at the end (validation is usually faster)
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
                current_loss = total_loss / (batch_idx + 1)
                current_acc = 100.0 * correct / total
                logger.info(f"  Val Batch {batch_idx + 1}/{num_batches} - Loss: {current_loss:.4f}, Acc: {current_acc:.2f}%")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    # Calculate per-class accuracy
    emotion_names = ['neutral', 'happiness', 'anger', 'sadness', 'frustration']
    class_correct = [0] * 5
    class_total = [0] * 5
    for i in range(len(all_targets)):
        label = all_targets[i]
        class_total[label] += 1
        if all_preds[i] == label:
            class_correct[label] += 1
    
    # Log per-class accuracy
    logger.info("Per-class validation accuracy:")
    for i, name in enumerate(emotion_names):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            logger.info(f"  {name}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_targets)


def main():
    parser = argparse.ArgumentParser(description='Train AcousticEmotionModel on RAVDESS dataset')
    parser.add_argument('--data_dir', type=str, default='data/raw/',
                        help='Path to RAVDESS dataset directory (default: data/raw/)')
    parser.add_argument('--output_dir', type=str, default='backend/models/',
                        help='Output directory for model (default: backend/models/)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--max_time_frames', type=int, default=500,
                        help='Maximum time frames for padding/truncation (default: 500)')
    parser.add_argument('--emotion_mapping', type=str, default='default',
                        choices=['default', 'strict', 'expanded'],
                        help='Emotion mapping strategy: default (5 classes), strict (8 classes), expanded (alternative grouping)')
    parser.add_argument('--normalization_method', type=str, default='cmvn',
                        choices=['cmvn', 'zscore', 'logmel', 'minmax'],
                        help='Normalization method: cmvn (default, best for SER), zscore, logmel, minmax')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_dir}")
    audio_files, labels, class_weights, actor_ids = load_ravdess_dataset(args.data_dir, args.emotion_mapping)
    
    # Initialize feature extractor WITHOUT BERT (faster for audio-only training)
    # BERT is only needed for text features, not for audio feature extraction
    feature_extractor = FeatureExtractor(load_bert=False)
    
    # Compute dataset-level statistics for normalization (if needed)
    normalization_stats = {}
    if args.normalization_method in ['zscore', 'minmax']:
        logger.info("Computing dataset statistics for normalization...")
        all_mel_values = []
        sample_count = min(200, len(audio_files))  # Sample subset for speed
        
        for i in range(sample_count):
            try:
                audio_features = feature_extractor.load_audio_features(audio_files[i])
                mel_spec = audio_features['mel_spectrogram']
                all_mel_values.append(mel_spec.flatten())
            except Exception as e:
                continue
        
        if all_mel_values:
            all_mel_values = np.concatenate(all_mel_values)
            if args.normalization_method == 'zscore':
                normalization_stats['mean'] = float(np.mean(all_mel_values))
                normalization_stats['std'] = float(np.std(all_mel_values))
                logger.info(f"Dataset statistics: mean={normalization_stats['mean']:.4f}, std={normalization_stats['std']:.4f}")
            elif args.normalization_method == 'minmax':
                normalization_stats['min'] = float(np.percentile(all_mel_values, 1))  # Use 1st percentile to handle outliers
                normalization_stats['max'] = float(np.percentile(all_mel_values, 99))  # Use 99th percentile
                logger.info(f"Dataset mel-spectrogram range: [{normalization_stats['min']:.4f}, {normalization_stats['max']:.4f}]")
        else:
            # Fallback values
            if args.normalization_method == 'zscore':
                normalization_stats = {'mean': 0.0, 'std': 1.0}
            else:
                normalization_stats = {'min': -80.0, 'max': 0.0}
            logger.warning("Could not compute dataset statistics, using fallback values")
    
    # Save normalization statistics
    stats = {
        'normalization_method': args.normalization_method,
        **normalization_stats
    }
    stats_path = output_path / 'emotion_dataset_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Normalization statistics saved to {stats_path}")
    logger.info(f"Using normalization method: {args.normalization_method}")
    
    # Split by actors to prevent data leakage (speaker-independent split)
    # RAVDESS has actors 01-24, we'll use actors 01-18 for train, 19-24 for validation
    # This ensures no actor appears in both train and validation sets
    
    # Create train dataset with SpecAugment augmentation
    train_dataset_full = EmotionDataset(
        [audio_files[i] for i in range(len(audio_files))], 
        [labels[i] for i in range(len(labels))], 
        feature_extractor, 
        args.max_time_frames,
        use_specaugment=True,  # Enable SpecAugment for training
        normalization_method=args.normalization_method,
        normalization_stats=normalization_stats
    )
    
    # Create validation dataset without augmentation but with same normalization
    val_dataset_full = EmotionDataset(
        [audio_files[i] for i in range(len(audio_files))], 
        [labels[i] for i in range(len(labels))], 
        feature_extractor, 
        args.max_time_frames,
        use_specaugment=False,  # No augmentation for validation
        normalization_method=args.normalization_method,
        normalization_stats=normalization_stats
    )    
    # Use actor-based splitting
    train_actor_ids = set(range(1, 19))
    val_actor_ids = set(range(19, 25))
    
    train_indices = []
    val_indices = []
    for idx, actor_id in enumerate(actor_ids):
        if actor_id in train_actor_ids:
            train_indices.append(idx)
        elif actor_id in val_actor_ids:
            val_indices.append(idx)
    
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)
    
    # Log class distributions after split to verify balanced splits
    def log_split_distributions(indices, labels, split_name):
        """Log class distribution for a given split."""
        split_labels = [labels[i] for i in indices]
        unique_labels, counts = np.unique(split_labels, return_counts=True)
        emotion_names = ['neutral', 'happiness', 'anger', 'sadness', 'frustration']
        logger.info(f"\n{split_name} class distribution:")
        for label, count in zip(unique_labels, counts):
            percentage = 100.0 * count / len(split_labels)
            logger.info(f"  {emotion_names[label]}: {count} ({percentage:.1f}%)")
        return dict(zip(unique_labels, counts))
    
    train_dist = log_split_distributions(train_indices, labels, "Train")
    val_dist = log_split_distributions(val_indices, labels, "Validation")
    
    # Check for missing classes in validation set
    missing_classes = set(train_dist.keys()) - set(val_dist.keys())
    if missing_classes:
        emotion_names = ['neutral', 'happiness', 'anger', 'sadness', 'frustration']
        missing_names = [emotion_names[c] for c in missing_classes]
        logger.warning(f"WARNING: Validation set missing emotion classes: {missing_names}")

    # Create data loaders
    # Use num_workers=0 on Windows to avoid multiprocessing pickling issues
    # On Linux/Mac, you can use num_workers=2 for faster loading
    num_workers = 0 if os.name == 'nt' else 0
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    
    # Initialize model
    model = AcousticEmotionModel(n_mels=128, n_mfcc=40, num_classes=5, dropout=0.5).to(device)
    
    # Apply proper weight initialization
    def init_weights(m):
        """Initialize model weights using Kaiming/Xavier initialization."""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)
    
    model.apply(init_weights)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info("Applied Kaiming/Xavier weight initialization")
    
    # Loss and optimizer
    # AdamW with decoupled weight decay (better than Adam)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=5e-4, betas=(0.9, 0.999))
    logger.info(f"Using AdamW optimizer with lr={args.learning_rate}, weight_decay=5e-4")
    
    # Learning rate warmup for first 5 epochs
    warmup_epochs = 5
    def warmup_lambda(epoch):
        """Linear warmup from 0 to target learning rate."""
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    # Main scheduler (ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    # Increased label smoothing from 0.1 to 0.15 to prevent overconfidence and model collapse
    criterion = LabelSmoothingCrossEntropy(smoothing=0.15, weight=class_weights_tensor)
    logger.info(f"Using warmup scheduler for first {warmup_epochs} epochs, then ReduceLROnPlateau with patience=5, factor=0.5, min_lr=1e-6, smoothing=0.15")
    logger.info(f"Class weights tensor shape: {class_weights_tensor.shape}, device: {class_weights_tensor.device}")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_val_acc': 0.0
    }
    
    # Training loop
    logger.info("Starting training...")
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = 20  # Increased patience
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate scheduling with warmup
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(f"Epoch {epoch}/{args.epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Log prediction distribution to monitor class diversity
        val_pred_dist = np.bincount(val_preds, minlength=5)
        emotion_names = ['neutral', 'happiness', 'anger', 'sadness', 'frustration']
        logger.info(f"  Validation prediction distribution: {dict(zip(emotion_names, val_pred_dist))}")
        
        # Log confusion matrix every 5 epochs
        if epoch % 5 == 0:
            emotion_names = ['neutral', 'happiness', 'anger', 'sadness', 'frustration']
            cm = confusion_matrix(val_targets, val_preds)
            logger.info(f"\nConfusion Matrix (Epoch {epoch}):\n{cm}")
            logger.info(f"Most predicted class: {emotion_names[np.bincount(val_preds, minlength=5).argmax()]}")
            pred_dist = np.bincount(val_preds, minlength=5)
            logger.info(f"Class distribution in predictions: {dict(zip(emotion_names, pred_dist))}")
        
        # DEBUG: Check model outputs after first epoch
        if epoch == 1:
            model.eval()
            with torch.no_grad():
                
                sample_batch = next(iter(train_loader))
                if len(sample_batch) == 4:
                    sample_mel, sample_mfcc, sample_labels, sample_lengths = sample_batch
                    sample_mel = sample_mel.to(device)
                    sample_mfcc = sample_mfcc.to(device) if sample_mfcc is not None else None
                    sample_lengths = sample_lengths.to(device)
                elif len(sample_batch) == 3:
                    sample_mel, sample_labels, sample_lengths = sample_batch
                    sample_mel = sample_mel.to(device)
                    sample_mfcc = None
                    sample_lengths = sample_lengths.to(device)
                else:
                    sample_mel, sample_labels = sample_batch
                    sample_mel = sample_mel.to(device)
                    sample_mfcc = None
                    sample_lengths = None
                outputs = model(sample_mel, mfcc=sample_mfcc, lengths=sample_lengths)
                probs = torch.softmax(outputs, dim=1)
                logits_raw = outputs[0].cpu().numpy() # Get raw logits
                logger.info(f"DEBUG: Sample outputs shape: {outputs.shape}")
                logger.info(f"DEBUG: Raw logits (pre-softmax): {logits_raw}")
                logger.info(f"DEBUG: Logits range: [{logits_raw.min():.2f}, {logits_raw.max():.2f}]")
                logger.info(f"DEBUG: Logits mean: {logits_raw.mean():.2f}, std: {logits_raw.std():.2f}")
                logger.info(f"DEBUG: Sample probabilities (first sample): {probs[0].cpu().numpy()}")
                logger.info(f"DEBUG: Sample predictions (first 5): {outputs.argmax(dim=1)[:5].cpu().numpy()}")
                logger.info(f"DEBUG: Sample labels (first 5): {sample_labels[:5].numpy()}")
                logger.info(f"DEBUG: Current learning rate: {optimizer.param_groups[0]['lr']}")

                # Check for model collapse using proper metrics
                def check_model_collapse(outputs, targets, num_classes=5):
                    """
                    Check if model has collapsed using multiple indicators.
                    
                    Returns:
                        Dict with collapse indicators and warnings
                    """
                    probs = torch.softmax(outputs, dim=1)
                    preds = outputs.argmax(dim=1)
                    
                    # 1. Entropy of predictions (low entropy = collapse)
                    # Average entropy across batch
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean().item()
                    min_entropy = np.log(num_classes) * 0.3  # Threshold: 30% of max entropy
                    
                    # 2. Class prediction distribution (should be diverse)
                    unique_preds, pred_counts = np.unique(preds.cpu().numpy(), return_counts=True)
                    num_predicted_classes = len(unique_preds)
                    
                    # 3. Per-class accuracy check (missing classes = potential collapse)
                    per_class_correct = {}
                    per_class_total = {}
                    for cls in range(num_classes):
                        mask = (targets == cls)
                        if mask.sum() > 0:
                            per_class_total[cls] = mask.sum().item()
                            per_class_correct[cls] = (preds[mask] == targets[mask]).sum().item()
                    
                    # Generate warnings
                    warnings = []
                    if entropy < min_entropy:
                        warnings.append(f"Low prediction entropy ({entropy:.3f} < {min_entropy:.3f}) - model may be overconfident")
                    if num_predicted_classes < num_classes:
                        missing = set(range(num_classes)) - set(unique_preds)
                        warnings.append(f"Model not predicting all classes - missing: {missing}")
                    
                    return {
                        'entropy': entropy,
                        'num_predicted_classes': num_predicted_classes,
                        'per_class_accuracy': {k: per_class_correct.get(k, 0) / per_class_total.get(k, 1) 
                                             for k in range(num_classes)},
                        'warnings': warnings,
                        'is_collapsed': len(warnings) > 0
                    }
                
                collapse_info = check_model_collapse(outputs, sample_labels.to(device))
                logger.info(f"DEBUG: Model collapse check - Entropy: {collapse_info['entropy']:.3f}, "
                          f"Predicted classes: {collapse_info['num_predicted_classes']}/{5}")
                if collapse_info['warnings']:
                    for warning in collapse_info['warnings']:
                        logger.warning(f"DEBUG: {warning}")
                logger.info(f"DEBUG: Per-class accuracy: {collapse_info['per_class_accuracy']}")
            model.train()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = output_path / 'emotion_model.pth'
            model.save_model(str(model_path))
            logger.info(f"Saved best model with validation accuracy: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping - DISABLED temporarily for debugging
        # Uncomment this once model starts learning
        # if patience_counter >= max_patience:
        #     logger.info(f"Early stopping at epoch {epoch}")
        #     break
    # Final evaluation with comprehensive metrics
    logger.info("Final evaluation on validation set...")
    val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
    
    emotion_names = ['neutral', 'happiness', 'anger', 'sadness', 'frustration']
    
    # Calculate comprehensive metrics
    f1_weighted = f1_score(val_targets, val_preds, average='weighted')
    f1_macro = f1_score(val_targets, val_preds, average='macro')
    f1_per_class = f1_score(val_targets, val_preds, average=None)
    
    # Per-class precision and recall
    from sklearn.metrics import precision_score, recall_score
    precision_per_class = precision_score(val_targets, val_preds, average=None, zero_division=0)
    recall_per_class = recall_score(val_targets, val_preds, average=None, zero_division=0)
    
    cm = confusion_matrix(val_targets, val_preds)
    report = classification_report(val_targets, val_preds, target_names=emotion_names)
    
    # Log comprehensive results
    logger.info(f"\n{'='*60}")
    logger.info(f"Final Validation Results")
    logger.info(f"{'='*60}")
    logger.info(f"Overall Accuracy: {val_acc:.2f}%")
    logger.info(f"F1-Score (weighted): {f1_weighted:.4f}")
    logger.info(f"F1-Score (macro): {f1_macro:.4f}")
    
    logger.info(f"\nPer-Class Metrics:")
    for i, emotion in enumerate(emotion_names):
        logger.info(f"  {emotion:12s} - Precision: {precision_per_class[i]:.4f}, "
                   f"Recall: {recall_per_class[i]:.4f}, F1: {f1_per_class[i]:.4f}")
    
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"{cm}")
    
    logger.info(f"\nDetailed Classification Report:")
    logger.info(f"{report}")
    
    # Save confusion matrix visualization
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=emotion_names, yticklabels=emotion_names)
        plt.title('Confusion Matrix - Emotion Recognition')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_path = output_path / 'emotion_model_confusion_matrix.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved to {cm_path}")
    except Exception as e:
        logger.warning(f"Could not create confusion matrix visualization: {e}")
    
    # Save detailed evaluation results
    eval_results = {
        'overall_accuracy': float(val_acc),
        'f1_weighted': float(f1_weighted),
        'f1_macro': float(f1_macro),
        'per_class_metrics': {
            emotion: {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i])
            }
            for i, emotion in enumerate(emotion_names)
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    eval_path = output_path / 'emotion_model_evaluation.json'
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"Detailed evaluation results saved to {eval_path}")
    
    # Save training history
    history['best_val_acc'] = best_val_acc
    history_path = output_path / 'emotion_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()

