import torch
keys = ['layer.0.attention.output.LayerNorm', 'layer.1.attention.output.LayerNorm', 'layer.2.attention.output.LayerNorm', 'layer.3.attention.output.LayerNorm', 'layer.4.attention.output.LayerNorm', 'layer.5.attention.output.LayerNorm', 'layer.6.attention.output.LayerNorm', 'layer.7.attention.output.LayerNorm', 'layer.8.attention.output.LayerNorm', 'layer.9.attention.output.LayerNorm', 'layer.10.attention.output.LayerNorm', 'layer.11.attention.output.LayerNorm']

# keys = ['layer.0.attention.output.LayerNorm']

@torch.no_grad()
def layer_norm(x: torch.Tensor) -> torch.Tensor:
    x_mean = x.mean(dim=0, keepdim=True)
    x_var  = x.var(dim=0, keepdim=True)
    return (x - x_mean) / torch.sqrt(x_var + 1e-5)

rank = 672
for key in keys:
    U: torch.Tensor = torch.load(f"svd_results/{key}.U.pt")[:, :rank]
    X: torch.Tensor = torch.load(f"svd_results/{key}.value.pt")
    Y = U.T @ X

    X_mean = X - X.mean(dim=0)
    Y_mean = Y - Y.mean(dim=0)
    V = X_mean @ torch.linalg.pinv(Y_mean)
    torch.save(V, f"svd_results/{key}.V.pt")

    # x0 = layer_norm(X)
    # x1: torch.Tensor = V @ layer_norm(U.T @ X)

    # print(x0.abs().mean())
    # print(x1.abs().mean())
    # print(x0.std())
    # print(x1.std())
    # print((x0 - x1).abs().mean())
