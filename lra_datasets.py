"""
LRA Benchmark Dataset Loaders for FRACTAL model.
Implements ListOps, Text (IMDB), and Image (sCIFAR-10) tasks.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ============================================================
# ListOps Task (seq_len=2048, 10 classes)
# ============================================================

def _generate_listops_example(max_depth=5, max_args=5, max_length=2000):
    """Generate a single ListOps example.
    
    Operations: [MIN, MAX, MED, SM (sum_mod)]
    Operands: single digits 0-9
    Result: integer 0-9
    """
    OPS = {'[MIN': min, '[MAX': max, '[MED': lambda x: sorted(x)[len(x)//2], 
           '[SM': lambda x: sum(x) % 10}
    OP_NAMES = list(OPS.keys())
    
    def _generate(depth=0):
        if depth >= max_depth or (depth > 0 and random.random() < 0.3):
            # Return a single digit
            val = random.randint(0, 9)
            return [str(val)], val
        
        op_name = random.choice(OP_NAMES)
        n_args = random.randint(2, max_args)
        
        tokens = [op_name]
        values = []
        
        for _ in range(n_args):
            sub_tokens, sub_val = _generate(depth + 1)
            tokens.extend(sub_tokens)
            values.append(sub_val)
        
        tokens.append(']')
        result = OPS[op_name](values)
        return tokens, result
    
    # Keep generating until we get something reasonable length
    for _ in range(100):
        tokens, result = _generate()
        if 10 <= len(tokens) <= max_length:
            return tokens, result
    
    return tokens, result


# Vocabulary for ListOps
LISTOPS_VOCAB = {
    '<pad>': 0, '[MIN': 1, '[MAX': 2, '[MED': 3, '[SM': 4, ']': 5,
    '0': 6, '1': 7, '2': 8, '3': 9, '4': 10, '5': 11, 
    '6': 12, '7': 13, '8': 14, '9': 15
}


class ListOpsDataset(Dataset):
    """ListOps classification task. 10 classes (digits 0-9)."""
    
    def __init__(self, n_samples=96000, seq_len=2048, split='train', seed=42):
        super().__init__()
        self.seq_len = seq_len
        self.n_samples = n_samples
        
        # Adjust samples per split
        if split == 'train':
            actual_seed = seed
            self.n_samples = n_samples
        elif split == 'val':
            actual_seed = seed + 1
            self.n_samples = n_samples // 10
        else:  # test
            actual_seed = seed + 2
            self.n_samples = n_samples // 10
        
        # Generate all examples
        rng = random.Random(actual_seed)
        old_state = random.getstate()
        random.setstate(rng.getstate())
        
        self.data = []
        self.labels = []
        
        for _ in range(self.n_samples):
            tokens, result = _generate_listops_example()
            # Tokenize
            token_ids = [LISTOPS_VOCAB.get(t, 0) for t in tokens]
            # Pad or truncate
            if len(token_ids) > seq_len:
                token_ids = token_ids[:seq_len]
            else:
                token_ids = token_ids + [0] * (seq_len - len(token_ids))
            
            self.data.append(token_ids)
            self.labels.append(result)
        
        random.setstate(old_state)
        
        self.data = np.array(self.data, dtype=np.int64)
        self.labels = np.array(self.labels, dtype=np.int64)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# ============================================================
# Text (IMDB) Task (seq_len=4096, 2 classes, byte-level)
# ============================================================

class IMDBDataset(Dataset):
    """IMDB sentiment classification with byte-level tokenization.
    Sequence length: 4096 (as per LRA standard, though paper says 1024 for some).
    Actually paper uses 4096 for Text task.
    """
    
    def __init__(self, seq_len=1024, split='train', max_samples=None):
        super().__init__()
        self.seq_len = seq_len
        
        from datasets import load_dataset
        
        if split in ['train', 'val']:
            ds = load_dataset('stanfordnlp/imdb', split='train')
            # Split into train/val (90/10)
            n = len(ds)
            indices = list(range(n))
            random.Random(42).shuffle(indices)
            split_point = int(0.9 * n)
            if split == 'train':
                indices = indices[:split_point]
            else:
                indices = indices[split_point:]
            ds = ds.select(indices)
        else:
            ds = load_dataset('stanfordnlp/imdb', split='test')
        
        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))
        
        self.texts = []
        self.labels = []
        
        for example in ds:
            # Byte-level encoding
            text_bytes = example['text'].encode('utf-8')
            # Convert to list of ints (0-255), shift by 1 for padding token 0
            token_ids = [b + 1 for b in text_bytes]
            
            # Pad or truncate
            if len(token_ids) > seq_len:
                token_ids = token_ids[:seq_len]
            else:
                token_ids = token_ids + [0] * (seq_len - len(token_ids))
            
            self.texts.append(token_ids)
            self.labels.append(example['label'])
        
        self.texts = np.array(self.texts, dtype=np.int64)
        self.labels = np.array(self.labels, dtype=np.int64)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# ============================================================
# Image (Sequential CIFAR-10) Task (seq_len=1024, 10 classes)
# ============================================================

class SequentialCIFAR10Dataset(Dataset):
    """Sequential CIFAR-10: grayscale image flattened to 1D sequence.
    32x32 = 1024 pixels, each pixel is a value 0-255.
    """
    
    def __init__(self, split='train', normalize=True):
        super().__init__()
        import torchvision
        import torchvision.transforms as transforms
        
        is_train = (split == 'train')
        
        # Download CIFAR-10
        # Grayscale conversion: 0.2989*R + 0.5870*G + 0.1140*B
        dataset = torchvision.datasets.CIFAR10(
            root='/workspace/data/cifar10',
            train=is_train,
            download=True
        )
        
        self.data = []
        self.labels = []
        
        for img, label in dataset:
            # Convert to grayscale
            img_np = np.array(img)  # (32, 32, 3)
            gray = 0.2989 * img_np[:,:,0] + 0.5870 * img_np[:,:,1] + 0.1140 * img_np[:,:,2]
            # Flatten to (1024,)
            flat = gray.flatten()
            if normalize:
                flat = flat / 255.0
            self.data.append(flat)
            self.labels.append(label)
        
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        # For val split, take last 5000 from train
        if split == 'val':
            self.data = self.data[-5000:]
            self.labels = self.labels[-5000:]
        elif split == 'train':
            self.data = self.data[:-5000]
            self.labels = self.labels[:-5000]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Return (seq_len,) float tensor and label
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# ============================================================
# Retrieval (AAN) Task - Simplified version
# ============================================================

class RetrievalDataset(Dataset):
    """Simplified retrieval task using IMDB pairs.
    Classify whether two documents have the same sentiment.
    This is NOT the exact AAN task but captures the sequence matching spirit.
    seq_len = 4000 (2 documents concatenated)
    """
    
    def __init__(self, seq_len=4000, split='train', n_samples=147086):
        super().__init__()
        self.seq_len = seq_len
        self.half_len = seq_len // 2
        
        from datasets import load_dataset
        
        if split in ['train', 'val']:
            ds = load_dataset('stanfordnlp/imdb', split='train')
        else:
            ds = load_dataset('stanfordnlp/imdb', split='test')
        
        # Create pairs
        rng = random.Random(42 if split == 'train' else 43 if split == 'val' else 44)
        
        if split == 'val':
            n_samples = n_samples // 10
        elif split == 'test':
            n_samples = n_samples // 10
        
        n_samples = min(n_samples, len(ds) * 2)
        
        self.data = []
        self.labels = []
        
        # Pre-encode all texts
        encoded = []
        labels_list = []
        for example in ds:
            text_bytes = example['text'].encode('utf-8')
            token_ids = [b + 1 for b in text_bytes[:self.half_len]]
            if len(token_ids) < self.half_len:
                token_ids = token_ids + [0] * (self.half_len - len(token_ids))
            encoded.append(token_ids)
            labels_list.append(example['label'])
        
        indices = list(range(len(encoded)))
        
        for _ in range(n_samples):
            i, j = rng.sample(indices, 2)
            # Concatenate two documents
            combined = encoded[i] + encoded[j]
            label = 1 if labels_list[i] == labels_list[j] else 0
            self.data.append(combined)
            self.labels.append(label)
        
        self.data = np.array(self.data, dtype=np.int64)
        self.labels = np.array(self.labels, dtype=np.int64)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# ============================================================
# Dataset factory
# ============================================================

TASK_CONFIGS = {
    'listops': {
        'n_classes': 10,
        'seq_len': 2048,
        'input_type': 'discrete',
        'vocab_size': 16,  # 0-15
        'd_input': 1,
        'batch_size': 32,
        'epochs': 40,
    },
    'text': {
        'n_classes': 2,
        'seq_len': 1024,  # Paper uses 1024 for byte-level with truncation
        'input_type': 'discrete',
        'vocab_size': 257,  # 0=pad, 1-256=bytes
        'd_input': 1,
        'batch_size': 16,
        'epochs': 40,
    },
    'retrieval': {
        'n_classes': 2,
        'seq_len': 4000,
        'input_type': 'discrete',
        'vocab_size': 257,
        'd_input': 1,
        'batch_size': 32,
        'epochs': 30,
    },
    'image': {
        'n_classes': 10,
        'seq_len': 1024,
        'input_type': 'continuous',
        'vocab_size': None,
        'd_input': 1,
        'batch_size': 50,
        'epochs': 200,
    },
}


def get_dataset(task, split='train', **kwargs):
    """Get dataset for a specific LRA task."""
    if task == 'listops':
        return ListOpsDataset(split=split, seq_len=2048, **kwargs)
    elif task == 'text':
        return IMDBDataset(split=split, seq_len=1024, **kwargs)
    elif task == 'retrieval':
        return RetrievalDataset(split=split, seq_len=4000, **kwargs)
    elif task == 'image':
        return SequentialCIFAR10Dataset(split=split, **kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")


def get_dataloaders(task, batch_size=None, num_workers=4):
    """Get train/val/test dataloaders for a task."""
    config = TASK_CONFIGS[task]
    if batch_size is None:
        batch_size = config['batch_size']
    
    train_ds = get_dataset(task, split='train')
    val_ds = get_dataset(task, split='val')
    test_ds = get_dataset(task, split='test')
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, config


if __name__ == '__main__':
    # Test each dataset
    print("Testing ListOps dataset...")
    ds = ListOpsDataset(n_samples=100, split='train')
    x, y = ds[0]
    print(f"  ListOps: x.shape={x.shape}, y={y.item()}, vocab range=[{x.min()}, {x.max()}]")
    print(f"  First 20 tokens: {x[:20].tolist()}")
    print(f"  Label distribution: {np.bincount(ds.labels, minlength=10)}")
    
    print("\nTesting IMDB dataset...")
    ds = IMDBDataset(split='train', seq_len=1024, max_samples=100)
    x, y = ds[0]
    print(f"  IMDB: x.shape={x.shape}, y={y.item()}, vocab range=[{x.min()}, {x.max()}]")
    
    print("\nTesting sCIFAR-10 dataset...")
    ds = SequentialCIFAR10Dataset(split='train')
    x, y = ds[0]
    print(f"  sCIFAR: x.shape={x.shape}, y={y.item()}, value range=[{x.min():.3f}, {x.max():.3f}]")
    print(f"  Train size: {len(ds)}")
    
    print("\nAll dataset tests passed!")
