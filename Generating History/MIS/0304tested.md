# 0304 tested

## Brief Overview

| Index | Strategy                                    | Result                                                  |
| ----- | ------------------------------------------- | ------------------------------------------------------- |
| 1     | A conversation between ChatGPT and Deepseek | (17.602, 20.158, 12.559192090607825, 6.399434448292743) |
| 2     |                                             |                                                         |
| 3     |                                             |                                                         |



## Log

We tried to start a conversation between ChatGPT and DeepSeek, so that they may generate a better solution. Here's what they finally come up with:
```python
def get_heatmap_mis(
        self, xt: Tensor, edges_feature: Tensor, mask: Tensor, t: Tensor
    ) -> Tensor:
        B, N = xt.shape
        device = xt.device

        # Structural metrics computation
        undetermined = ~mask
        excluded = mask & (xt == 0)
        undetermined_float, excluded_float = undetermined.float(), excluded.float()
        
        remaining_degree = torch.bmm(edges_feature, undetermined_float.unsqueeze(-1)).squeeze(-1)
        excluded_neighbors = torch.bmm(edges_feature, excluded_float.unsqueeze(-1)).squeeze(-1)
        original_degree = edges_feature.sum(dim=2)
        neighbor_remaining_sum = torch.bmm(edges_feature, remaining_degree.unsqueeze(-1)).squeeze(-1)

        # Core heuristic formulation
        coverage_ratio = (excluded_neighbors + 1) / (remaining_degree + 1)
        structural_impact = (neighbor_remaining_sum + 1) / (original_degree + 1)
        heuristic_score = coverage_ratio * structural_impact

        # Temporal message passing (3 iterations)
        t_expanded = t.view(B, 1)
        for iter in range(3):
            neighbor_avg = torch.bmm(edges_feature, heuristic_score.unsqueeze(-1)).squeeze(-1)
            neighbor_avg /= (original_degree + 1e-9)
            blend_ratio = (0.65 - 0.08*iter - 0.1*t_expanded).clamp(0.45, 0.7)
            heuristic_score = blend_ratio * heuristic_score + (1 - blend_ratio) * neighbor_avg

        # Density-aware suppression
        neighbor_max = (edges_feature * heuristic_score.unsqueeze(1)).max(dim=2).values
        neighbor_avg = (edges_feature * heuristic_score.unsqueeze(1)).sum(dim=2) / (original_degree + 1e-9)
        suppression = (0.38 - 0.12*t_expanded) * (neighbor_max / (neighbor_avg + 1e-6))
        heuristic_score = torch.relu(heuristic_score - suppression)

        # Progressive state integration
        progress = 1 - undetermined_float.mean(dim=1, keepdim=True)
        blend_weight = torch.exp(-4.2 * t_expanded * (progress + 0.2))
        blended = blend_weight * xt + (1 - blend_weight) * heuristic_score

        # Final probability mapping
        undetermined_scores = torch.where(undetermined, blended, torch.tensor(-1e9, device=device))
        return torch.where(mask, xt, torch.softmax(undetermined_scores, dim=1))
```

This achieves **(17.602, 20.158, 12.559192090607825, 6.399434448292743)** result. Not the best, and the middle algorithm during the conversation did not improve greatly, ranging around 12 and 13. 

We need a switch in the strategy.