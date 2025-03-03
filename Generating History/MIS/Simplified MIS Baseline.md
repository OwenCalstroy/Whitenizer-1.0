# Simplified MIS Baseline

The baseline should be more simpler so that the newly generated algorithm can be more creative.

## Version One

```python
import torch
from torch import Tensor

def get_heatmap_mis_minimal(
        self, xt: Tensor, edges_feature: Tensor, mask: Tensor, t: Tensor
    ) -> Tensor:
    # Core components in 5 essential lines
    excluded = mask & (xt == 0)
    excl_neighbors = (edges_feature @ excluded.float().unsqueeze(-1)).squeeze()
    rem_degree = (edges_feature @ (~mask).float().unsqueeze(-1)).squeeze()
    
    scores = (excl_neighbors + 1) / (rem_degree + 1)  # Exclusion ratio
    blended = torch.exp(-t[:, None]) * xt + (1 - torch.exp(-t[:, None])) * scores
    
    # Strategic softmax application
    final = torch.where(~mask, blended, -1e9).softmax(dim=1)
    return torch.where(mask, xt, final)

```

## Version Two

```python
import torch
from torch import Tensor

def get_heatmap_mis_simple(
        self, xt: Tensor, edges_feature: Tensor, mask: Tensor, t: Tensor
    ) -> Tensor:
    B, N = xt.shape
    device = xt.device

    # Identify excluded nodes (masked and value 0)
    excluded = mask & (xt == 0)
    excluded_float = excluded.float()
    # Count excluded neighbors per node
    excluded_neighbors = torch.bmm(edges_feature, excluded_float.unsqueeze(-1)).squeeze(-1)

    # Calculate remaining degree for undetermined nodes
    undetermined = ~mask
    remaining_degree = torch.bmm(edges_feature, undetermined.float().unsqueeze(-1)).squeeze(-1)

    # Simplified heuristic: ratio of excluded neighbors to remaining degree
    heuristic_score = (excluded_neighbors + 1.0) / (remaining_degree + 1.0)

    # Time-based blending with exponential decay
    blend_weight = torch.exp(-t.view(B, 1))
    blended = blend_weight * xt + (1 - blend_weight) * heuristic_score

    # Apply softmax only to undetermined nodes
    undetermined_scores = torch.where(undetermined, blended, torch.tensor(-1e9, device=device))
    prob_undetermined = torch.softmax(undetermined_scores, dim=1)

    # Combine with determined nodes (0 or 1)
    heatmap = torch.where(mask, xt, prob_undetermined)

    return heatmap

```

