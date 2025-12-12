# graph_constructor.py
import numpy as np
import torch


def build_chain_graph(full_state, x, device=None):
    """
    full_state: [3, nx] torch or numpy (n, u, E)
    x: numpy [nx]
    Returns:
      node_features [nx,4] torch [n, u, E, x]
      edge_index   [2,2*nx] torch.long
    """
    if isinstance(full_state, np.ndarray):
        n = torch.from_numpy(full_state[0].astype(np.float32))
        u = torch.from_numpy(full_state[1].astype(np.float32))
        E = torch.from_numpy(full_state[2].astype(np.float32))
    else:
        n = full_state[0]
        u = full_state[1]
        E = full_state[2]

    if device is None:
        device = n.device
    else:
        n = n.to(device)
        u = u.to(device)
        E = E.to(device)

    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    nx = n.shape[0]
    node_features = torch.stack([n, u, E, x_t], dim=-1)

    src = torch.arange(nx, dtype=torch.long, device=device)
    dst = (src + 1) % nx
    src_all = torch.cat([src, dst], dim=0)
    dst_all = torch.cat([dst, src], dim=0)
    edge_index = torch.stack([src_all, dst_all], dim=0)
    return node_features, edge_index
