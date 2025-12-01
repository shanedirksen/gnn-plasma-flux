# graph_constructor.py
import numpy as np
import torch


def build_chain_graph(state, x):
    """
    Build a simple 1D chain graph for a given state.

    Args:
        state: np.ndarray of shape [nx], scalar field (e.g., density).
        x: np.ndarray of shape [nx], cell center coordinates.

    Returns:
        node_features: torch.FloatTensor [nx, 2] -> [state, x]
        edge_index: torch.LongTensor [2, num_edges] with periodic connections.
    """
    state = np.asarray(state, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    assert state.shape == x.shape

    nx = state.shape[0]
    node_features = torch.from_numpy(np.stack([state, x], axis=-1))  # [nx, 2]

    # edges: i -> i+1 (periodic)
    src = np.arange(nx, dtype=np.int64)
    dst = (src + 1) % nx
    # undirected: add both directions
    src_all = np.concatenate([src, dst], axis=0)
    dst_all = np.concatenate([dst, src], axis=0)
    edge_index = torch.from_numpy(np.stack([src_all, dst_all], axis=0))  # [2, 2*nx]

    return node_features, edge_index
