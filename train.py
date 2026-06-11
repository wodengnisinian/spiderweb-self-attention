# train.py - Training pipeline for SpiderWeb Self-Attention models

import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import time
from collections import defaultdict


def train_epoch(model, loader, optimizer, criterion, device, lambda_=0.5,
                build_bias_fn=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        labels = batch["label"].to(device)
        mask = (token_ids != 0).unsqueeze(1).unsqueeze(2).to(device)

        M_web = None
        if build_bias_fn is not None:
            levels = batch["levels"].to(device)
            segments = batch["segments"].to(device)
            M_web = build_bias_fn(levels, segments, batch["seq_len"])

        logits = model(token_ids, mask, M_web, lambda_)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * token_ids.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += token_ids.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, lambda_=0.5, build_bias_fn=None):
    """Evaluate model. Returns loss, accuracy."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        labels = batch["label"].to(device)
        mask = (token_ids != 0).unsqueeze(1).unsqueeze(2).to(device)

        M_web = None
        if build_bias_fn is not None:
            levels = batch["levels"].to(device)
            segments = batch["segments"].to(device)
            M_web = build_bias_fn(levels, segments, batch["seq_len"])

        logits = model(token_ids, mask, M_web, lambda_)
        loss = criterion(logits, labels)

        total_loss += loss.item() * token_ids.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += token_ids.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_detailed(model, loader, device, lambda_=0.5, build_bias_fn=None):
    """
    Detailed evaluation: returns all predictions, labels, and seq_lens
    for computing Precision, Recall, F1, and length-grouped metrics.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_logits = []
    all_seq_lens = []

    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        labels = batch["label"].to(device)
        mask = (token_ids != 0).unsqueeze(1).unsqueeze(2).to(device)

        M_web = None
        if build_bias_fn is not None:
            levels = batch["levels"].to(device)
            segments = batch["segments"].to(device)
            M_web = build_bias_fn(levels, segments, batch["seq_len"])

        logits = model(token_ids, mask, M_web, lambda_)
        loss = criterion(logits, labels)

        total_loss += loss.item() * token_ids.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += token_ids.size(0)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_logits.append(logits.cpu())
        all_seq_lens.append(batch["seq_len"].clone().detach())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_seq_lens = torch.cat(all_seq_lens)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "preds": all_preds.numpy(),
        "labels": all_labels.numpy(),
        "seq_lens": all_seq_lens.numpy(),
    }


def train_model(model, train_loader, test_loader, epochs=30, lr=1e-3,
                lambda_=0.5, build_bias_fn=None, device="cpu",
                verbose=True):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = defaultdict(list)
    t0 = time.time()

    pbar = tqdm(range(epochs), desc="Training", disable=not verbose)
    for epoch in pbar:
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            lambda_=lambda_, build_bias_fn=build_bias_fn
        )
        test_loss, test_acc = evaluate(
            model, test_loader, device,
            lambda_=lambda_, build_bias_fn=build_bias_fn
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if verbose:
            pbar.set_postfix({
                "tr_loss": f"{train_loss:.3f}",
                "te_acc": f"{test_acc:.3f}",
            })

    history["time"] = time.time() - t0

    # Run detailed evaluation at the end
    detail = evaluate_detailed(
        model, test_loader, device,
        lambda_=lambda_, build_bias_fn=build_bias_fn
    )
    history["detail_preds"] = detail["preds"]
    history["detail_labels"] = detail["labels"]
    history["detail_seq_lens"] = detail["seq_lens"]

    return dict(history)




@torch.no_grad()
def evaluate_with_attention(model, loader, device, lambda_=0.5, build_bias_fn=None, M_web_build_fn=None):
    """
    Evaluate model and return predictions, labels, attention weights from last layer,
    and M_web matrix. Used for generating heatmaps and case studies.
    
    Returns dict with keys:
        preds, labels, seq_lens (numpy arrays)
        attn_weights_last (B, H, N, N) from final layer
        m_web_matrices (B, N, N) or None
        token_ids (numpy)
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_seq_lens = []
    all_token_ids = []
    all_attn = []
    all_m_web = []

    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        labels = batch["label"].to(device)
        mask = (token_ids != 0).unsqueeze(1).unsqueeze(2).to(device)

        M_web = None
        if M_web_build_fn is not None:
            levels = batch["levels"].to(device)
            segments = batch["segments"].to(device)
            M_web = M_web_build_fn(levels, segments, batch["seq_len"])

        logits, attns = model(token_ids, mask, M_web, lambda_, return_attention=True)
        loss = criterion(logits, labels)

        total_loss += loss.item() * token_ids.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += token_ids.size(0)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_seq_lens.append(batch["seq_len"].clone().detach())
        all_token_ids.append(token_ids.cpu())
        all_attn.append(attns[-1].cpu())  # last layer, (B, H, N, N)
        if M_web is not None:
            all_m_web.append(M_web.cpu())

    result = {
        "accuracy": correct / total,
        "loss": total_loss / total,
        "preds": torch.cat(all_preds).numpy(),
        "labels": torch.cat(all_labels).numpy(),
        "seq_lens": torch.cat(all_seq_lens).numpy(),
        "token_ids": torch.cat(all_token_ids).numpy(),
        "attn_weights_last": torch.cat(all_attn).numpy() if all_attn else None,
        "m_web_matrices": torch.cat(all_m_web).numpy() if all_m_web else None,
    }
    return result

if __name__ == "__main__":
    from data import create_dataloaders, build_m_web
    from model import create_model

    train_loader, test_loader, ds = create_dataloaders(num_samples=1000, batch_size=32)
    model = create_model(bias_mode="full")
    results = train_model(
        model, train_loader, test_loader, epochs=5, lr=1e-3, lambda_=0.5,
        build_bias_fn=build_m_web, device="cpu", verbose=True
    )
    print(f"Final test acc: {results['test_acc'][-1]:.4f}")
    print(f"Detail preds shape: {results['detail_preds'].shape}")
    print("train.py OK")


