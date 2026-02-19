import torch
import torch.optim as optim


def loss_params2str(train_params, train_batches, val_params, val_batches):
    def _format_loss_dict(params, type, batches):
        return " | ".join(f"{type} {k}: {float(v/batches):.4f}" for k, v in params.items())

    train_pstr = _format_loss_dict(train_params, "Train", train_batches)
    val_pstr = _format_loss_dict(val_params, "Val", val_batches)
    return f"{train_pstr} | {val_pstr}"

def train_vae_basic(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    learning_rate=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_dir='./checkpoints',
    name='basicVAE_general',
):
    device = torch.device(device)
    model = model.to(device)

    history = {
        'train': {},
        'val': {}
    }
    best_model_losses = None
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        train_loss_params = {}

        model.train()
        for batch_idx, (data, _) in enumerate(train_loader):
            x = data.to(device)

            optimizer.zero_grad()

            output = model(x)
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
        with torch.no_grad():
            val_loss_params = {}
            for batch_idx, (data, _) in enumerate(val_loader):
                x = data.to(device)
                output = model(x)
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
            
    print("Training complete!")
    return history
