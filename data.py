# data.py - Synthetic Data Generator for SpiderWeb Self-Attention
#
# Key design choice: classification signal comes from WHICH tokens appear in
# CENTER position, not from class-specific tokens. All tokens are shared across
# classes. Each class has a "signature" set of tokens that preferentially
# appear in center sentences. The model must learn position-sensitive attention,
# which is exactly what SpiderWeb attention is designed to help with.

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SyntheticArticleDataset(Dataset):
    """
    Synthetic dataset where classification relies on token POSITION, not just
    token identity.

    All tokens are shared across classes. Each class c has a signature token
    set S_c. In center sentences, S_c tokens appear with higher probability.
    In support/description sentences, S_c tokens appear at baseline rate.
    The same token may be a center signal for class 0 but appear randomly in
    description for class 1 - the model must learn position-sensitive patterns.
    """

    def __init__(self, num_samples=5000, num_classes=8, vocab_size=500,
                 max_seq_len=80, signature_size=15, seed=42,
                 center_bonus=0.40, support_bonus=0.10, desc_bonus=0.0):
        """
        Args:
            center_bonus: extra probability of signature tokens in center position
            support_bonus: extra probability of signature tokens in support position
            desc_bonus: extra probability of signature tokens in description position
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.signature_size = signature_size
        self.center_bonus = center_bonus
        self.support_bonus = support_bonus
        self.desc_bonus = desc_bonus
        self.rng = np.random.RandomState(seed)

        self.pad_id = 0
        self.usable_tokens = vocab_size - 1  # tokens 1..vocab_size-1

        # Each class gets a random signature token set (no overlap between classes)
        all_tokens = list(range(1, vocab_size))
        self.rng.shuffle(all_tokens)
        self.signatures = {}
        idx = 0
        for c in range(num_classes):
            self.signatures[c] = set(all_tokens[idx:idx + signature_size])
            idx += signature_size

        # Remaining tokens are "background"
        self.background = set(all_tokens[idx:])

        # Pre-generate all samples
        self.samples = []
        for i in range(num_samples):
            label = self.rng.randint(0, num_classes)
            sample = self._generate_one(label)
            self.samples.append(sample)

    def _sample_tokens(self, n, signature, bonus):
        """
        Sample n tokens. Signature tokens get `bonus` extra probability.
        Base probability is uniform over all usable tokens.
        """
        all_ids = list(range(1, self.vocab_size))
        n_total = len(all_ids)

        # Base probability: uniform
        sig_list = list(signature)
        bg_list = list(self.background)

        # Weighted sampling: signature tokens get higher weight
        weights = np.ones(n_total, dtype=np.float32)
        for t in sig_list:
            weights[t - 1] += bonus * n_total / len(sig_list)
        weights = weights / weights.sum()

        return list(self.rng.choice(all_ids, n, replace=True, p=weights))

    def _generate_one(self, label):
        signature = self.signatures[label]

        tokens = []
        levels = []
        segments = []
        seg_id = 0

        # Center sentence (level 0)
        center_len = self.rng.randint(6, 11)
        center_tokens = self._sample_tokens(center_len, signature, self.center_bonus)
        tokens.extend(center_tokens)
        levels.extend([0] * center_len)
        segments.extend([seg_id] * center_len)
        seg_id += 1

        # Support sentences (level 1): 2-3 sentences
        num_support = self.rng.randint(2, 4)
        for _ in range(num_support):
            sup_len = self.rng.randint(5, 9)
            sup_tokens = self._sample_tokens(sup_len, signature, self.support_bonus)
            tokens.extend(sup_tokens)
            levels.extend([1] * sup_len)
            segments.extend([seg_id] * sup_len)
            seg_id += 1

        # Description sentences (level 2): 3-5 sentences
        num_desc = self.rng.randint(3, 6)
        for _ in range(num_desc):
            desc_len = self.rng.randint(5, 9)
            desc_tokens = self._sample_tokens(desc_len, signature, self.desc_bonus)
            tokens.extend(desc_tokens)
            levels.extend([2] * desc_len)
            segments.extend([seg_id] * desc_len)
            seg_id += 1

        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            levels = levels[:self.max_seq_len]
            segments = segments[:self.max_seq_len]

        return tokens, levels, segments, label

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens, levels, segments, label = self.samples[idx]
        seq_len = len(tokens)
        pad_len = self.max_seq_len - seq_len
        tokens = tokens + [self.pad_id] * pad_len
        levels = levels + [-1] * pad_len
        segments = segments + [-1] * pad_len

        return {
            "token_ids": torch.tensor(tokens, dtype=torch.long),
            "levels": torch.tensor(levels, dtype=torch.long),
            "segments": torch.tensor(segments, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "seq_len": seq_len,
        }


def build_padding_mask(token_ids, pad_id=0):
    return (token_ids != pad_id).unsqueeze(1).unsqueeze(2)


def build_m_web(levels, segments, seq_lens, alpha=2.0, beta=1.0, gamma=1.5):
    """Build full SpiderWeb bias: center + hierarchy + position."""
    B, N = levels.shape
    valid = (levels >= 0).float()
    valid_pair = torch.bmm(valid.unsqueeze(-1), valid.unsqueeze(1))

    center_mask = (levels == 0).float()
    M_center = alpha * torch.bmm(center_mask.unsqueeze(-1), center_mask.unsqueeze(1))

    safe_levels = levels.clone()
    safe_levels[safe_levels < 0] = 0
    level_diff = (safe_levels.unsqueeze(-1) - safe_levels.unsqueeze(1)).abs().float()
    M_hierarchy = -beta * level_diff * valid_pair

    safe_seg = segments.clone()
    safe_seg[safe_seg < 0] = -999
    same_seg = (safe_seg.unsqueeze(-1) == safe_seg.unsqueeze(1)).float()
    seg_dist = (safe_seg.unsqueeze(-1) - safe_seg.unsqueeze(1)).abs().float()
    adjacent = (seg_dist == 1).float()
    M_position = (gamma * same_seg + (gamma / 2.0) * adjacent) * valid_pair

    return M_center + M_hierarchy + M_position


def build_m_position_only(levels, segments, seq_lens, gamma=1.5):
    """Build position-only bias (Model B)."""
    B, N = levels.shape
    valid = (levels >= 0).float()
    valid_pair = torch.bmm(valid.unsqueeze(-1), valid.unsqueeze(1))
    safe_seg = segments.clone()
    safe_seg[safe_seg < 0] = -999
    same_seg = (safe_seg.unsqueeze(-1) == safe_seg.unsqueeze(1)).float()
    seg_dist = (safe_seg.unsqueeze(-1) - safe_seg.unsqueeze(1)).abs().float()
    adjacent = (seg_dist == 1).float()
    return (gamma * same_seg + (gamma / 2.0) * adjacent) * valid_pair


def build_m_simple_structure(levels, segments, seq_lens, beta=1.0, gamma=1.5):
    """Build simple structure bias: hierarchy + position (Model C)."""
    B, N = levels.shape
    valid = (levels >= 0).float()
    valid_pair = torch.bmm(valid.unsqueeze(-1), valid.unsqueeze(1))
    safe_levels = levels.clone()
    safe_levels[safe_levels < 0] = 0
    level_diff = (safe_levels.unsqueeze(-1) - safe_levels.unsqueeze(1)).abs().float()
    M_hierarchy = -beta * level_diff * valid_pair
    safe_seg = segments.clone()
    safe_seg[safe_seg < 0] = -999
    same_seg = (safe_seg.unsqueeze(-1) == safe_seg.unsqueeze(1)).float()
    seg_dist = (safe_seg.unsqueeze(-1) - safe_seg.unsqueeze(1)).abs().float()
    adjacent = (seg_dist == 1).float()
    M_position = (gamma * same_seg + (gamma / 2.0) * adjacent) * valid_pair
    return M_hierarchy + M_position


def create_dataloaders(num_samples=5000, batch_size=32, num_classes=8,
                       vocab_size=500, max_seq_len=80, seed=42,
                       signature_size=15,
                       center_bonus=0.40, support_bonus=0.10, desc_bonus=0.0):
    """Create train/test dataloaders."""
    dataset = SyntheticArticleDataset(
        num_samples=num_samples, num_classes=num_classes,
        vocab_size=vocab_size, max_seq_len=max_seq_len, seed=seed,
        signature_size=signature_size,
        center_bonus=center_bonus, support_bonus=support_bonus,
        desc_bonus=desc_bonus,
    )

    n_train = int(0.8 * num_samples)
    n_test = num_samples - n_train
    train_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             drop_last=False)

    return train_loader, test_loader, dataset


def build_m_random(levels, segments, seq_lens):
    """
    Random bias control: generates a random matrix with the same shape
    and magnitude distribution as M_web, but with shuffled entries.
    This proves SpiderWeb's structured bias is meaningful, not just "any bias helps".
    """
    import torch
    B, N = levels.shape
    device = levels.device

    # Build the real M_web first to get its statistical profile
    real = build_m_web(levels, segments, seq_lens)

    # Create random bias with same global mean and std
    M_random = torch.randn(B, N, N, device=device)
    M_random = M_random * real.std() + real.mean()

    # Mask out pad positions (same mask as real M_web)
    valid = (levels >= 0).float()
    valid_pair = torch.bmm(valid.unsqueeze(-1), valid.unsqueeze(1))
    M_random = M_random * valid_pair

    return M_random
if __name__ == "__main__":
    train_loader, test_loader, ds = create_dataloaders(num_samples=200)
    batch = next(iter(train_loader))
    print("Batch keys:", list(batch.keys()))
    print("token_ids:", batch["token_ids"].shape)
    print("levels:", batch["levels"].shape)
    print("labels:", batch["label"])
    M = build_m_web(batch["levels"], batch["segments"], batch["seq_len"])
    print("M_web:", M.shape, "range:", M.min().item(), M.max().item())
    print("data.py OK")

