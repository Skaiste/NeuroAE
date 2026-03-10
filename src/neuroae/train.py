import torch
import torch.optim as optim
import torch.nn.functional as F


def _dataset_valid_last_dim(dataset):
    if not getattr(dataset, "pad_features", False):
        return None
    original_shape = getattr(dataset, "original_shape", None)
    if original_shape is None or len(original_shape) == 0:
        return None
    valid_last_dim = int(original_shape[-1])
    if valid_last_dim <= 0:
        return None
    return valid_last_dim


def _build_valid_mask(x, valid_last_dim):
    if valid_last_dim is None or x.shape[-1] <= valid_last_dim:
        return None
    mask = torch.zeros_like(x)
    mask[..., :valid_last_dim] = 1.0
    return mask


def _apply_recon_mask(x, model_output, mask):
    if mask is None:
        return model_output

    def _mask_recon(recon):
        return recon * mask + x * (1.0 - mask)

    if isinstance(model_output, dict):
        out = dict(model_output)
        for key in ("x_hat", "recon", "reconstruction"):
            if key in out and torch.is_tensor(out[key]):
                out[key] = _mask_recon(out[key])
                break
        return out

    if isinstance(model_output, tuple):
        if len(model_output) == 0:
            return model_output
        return (_mask_recon(model_output[0]), *model_output[1:])

    if isinstance(model_output, list):
        if len(model_output) == 0:
            return model_output
        out = list(model_output)
        out[0] = _mask_recon(out[0])
        return out

    if torch.is_tensor(model_output):
        return _mask_recon(model_output)

    return model_output


def _masked_mse(x_hat, x, mask):
    if mask is None:
        return F.mse_loss(x_hat, x, reduction="mean")
    if x_hat.shape != x.shape:
        if x_hat.shape[:-1] != x.shape[:-1]:
            raise ValueError(
                f"Cannot align x_hat shape {tuple(x_hat.shape)} with x shape {tuple(x.shape)}."
            )
        common_last_dim = min(x_hat.shape[-1], x.shape[-1], mask.shape[-1])
        x_hat = x_hat[..., :common_last_dim]
        x = x[..., :common_last_dim]
        mask = mask[..., :common_last_dim]
    se = (x_hat - x).pow(2) * mask
    denom = mask.sum().clamp_min(1.0)
    return se.sum() / denom


def loss_params2str(train_params, train_batches, val_params, val_batches):
    def _format_loss_dict(params, type, batches):
        return " | ".join(f"{type} {k}: {float(v/batches):.4f}" for k, v in params.items())

    train_pstr = _format_loss_dict(train_params, "Train", train_batches)
    val_pstr = _format_loss_dict(val_params, "Val", val_batches)
    return f"{train_pstr} | {val_pstr}"

def train_vae(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_dir='./checkpoints',
    name='basicVAE_general',
    pca=None,
    noise=None,
    use_abeta_tau=False
):
    device = torch.device(device)
    model = model.to(device)

    if noise is not None:
        noise = {k:v for p in noise for k,v in p.items()}

    history = {
        'train': {},
        'val': {}
    }
    best_model_losses = None
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_valid_last_dim = _dataset_valid_last_dim(train_loader.dataset)
    val_valid_last_dim = _dataset_valid_last_dim(val_loader.dataset)
    for epoch in range(num_epochs):
        train_loss_params = {}

        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            x = data.to(device)
            valid_mask = _build_valid_mask(x, train_valid_last_dim)

            if noise is not None:
                if noise['type'] == 'gaussian':
                    n = torch.randn_like(x) + float(noise['std'])
                    x += n
                elif noise['type'] == 'mask':
                    n = (torch.rand_like(x) > float(noise['ratio'])).float()
                    x *= n

            optimizer.zero_grad()

            output = model(x)
            output = _apply_recon_mask(x, output, valid_mask)

            if use_abeta_tau:
                _, abeta, tau = labels
                abeta = abeta.to(device)
                tau = tau.to(device)
                loss = model.loss(x, abeta, tau, output)
            else:
                loss = model.loss(x, output)

            for p in loss:
                if p not in train_loss_params:
                    train_loss_params[p] = 0
                train_loss_params[p] += loss[p]

            loss['loss'].backward()
            optimizer.step()

        num_batches = batch_idx + 1
        for p in train_loss_params:
            train_loss_params[p] = float(train_loss_params[p].detach())
            if p not in history['train']:
                history['train'][p] = []
            history['train'][p].append(train_loss_params[p] / num_batches)

        # validation
        model.eval()
        val_loss_params = {}
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(val_loader):
                x = data.to(device)
                valid_mask = _build_valid_mask(x, val_valid_last_dim)
                output = model(x)
                output = _apply_recon_mask(x, output, valid_mask)
            
                if use_abeta_tau:
                    _, abeta, tau = labels
                    abeta = abeta.to(device)
                    tau = tau.to(device)
                    loss = model.loss(x, abeta, tau, output)
                else:
                    loss = model.loss(x, output)

                for p in loss:
                    if p not in val_loss_params:
                        val_loss_params[p] = 0
                    val_loss_params[p] += loss[p]

        num_val_batches = batch_idx + 1
        for p in val_loss_params:
            val_loss_params[p] = float(val_loss_params[p].detach())
            if p not in history['val']:
                history['val'][p] = []
            history['val'][p].append(val_loss_params[p] / num_val_batches)

        print(f"Epoch {epoch}/{num_epochs} | {loss_params2str(train_loss_params, num_batches, val_loss_params, num_val_batches)}")

        # select best model based on validation loss
        avg_val_loss = val_loss_params['loss'] / num_val_batches
        if best_model_losses is None or avg_val_loss < best_model_losses['val']['loss']:
            best_model_losses = {
                "train": {p:train_loss_params[p] / num_batches for p in train_loss_params},
                "val": {p:val_loss_params[p] / num_val_batches for p in val_loss_params}
            }
            torch.save(model.state_dict(), f'{save_dir}/{name}_model.pt')

    # run validation set through pca
    mse_pca = 0
    if pca is not None:
        total_mse_pca = 0
        num_batches = 0
        for batch_idx, (data, _) in enumerate(val_loader):
            x = data.to(device)
            valid_mask = _build_valid_mask(x, val_valid_last_dim)
            z_pca = pca.transform(x.detach().cpu().numpy())
            x_recon_pca = pca.inverse_transform(z_pca)
            x_recon_pca = torch.as_tensor(x_recon_pca, dtype=x.dtype, device=x.device)
            mse_pca = _masked_mse(x_recon_pca, x, valid_mask)
            total_mse_pca += mse_pca.item()
            num_batches += 1
        mse_pca = float(total_mse_pca / num_batches)
            
    print("Training complete!")
    return history, mse_pca
