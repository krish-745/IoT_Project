import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


def train(model, train_dataset, n_epochs=30, batch_size=64, lr=1e-3,
          val_split=0.1, patience=5, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    val_size   = max(1, int(len(train_dataset) * val_split))
    train_size = len(train_dataset) - val_size
    train_sub, val_sub = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_sub,   batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state    = None
    patience_ctr  = 0

    print(f"\n{'─'*55}")
    print(f"  Training GDN  |  epochs={n_epochs}  device={device}")
    print(f"  train_samples={train_size}  val_samples={val_size}")
    print(f"{'─'*55}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()
        val_loss /= len(val_loader)

        print(f"  epoch {epoch:3d}/{n_epochs}  |  train={train_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
        print(f"  ✓ Restored best model  (val_loss={best_val_loss:.5f})")
    print(f"{'─'*55}\n")


@torch.no_grad()
def compute_train_errors(model, train_dataset, batch_size=256, device="cpu"):
    model.eval()
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    all_errors = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        errors = torch.abs(model(x) - y)
        all_errors.append(errors.cpu())
    return torch.cat(all_errors, dim=0)
