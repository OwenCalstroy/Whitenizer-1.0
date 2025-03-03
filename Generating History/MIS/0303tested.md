# Brief Overview

| Index | Method                           | Result                                                  |
| ----- | -------------------------------- | ------------------------------------------------------- |
| 1     | by ReEvo's choice                | (17.516, 20.158, 12.990388051621728, 6.310539577449613) |
| 2     | + Self-Iteration T (ChatGPT-o3)  | (17.514, 20.158, 13.003158284186012, 6.381028752560134) |
| 3     | + Self-Iteration T (DeepSeek-R1) | (17.518, 20.158, 12.977891659125337, 6.331099779762719) |



# By ReEvo

```py
def get_heatmap_mis(
        self, xt: Tensor, edges_feature: Tensor, mask: Tensor, t: Tensor
    ) -> Tensor:
        B, N = xt.shape
        device = xt.device

        # Identify excluded nodes (masked and value 0)
        excluded = mask & (xt == 0)  # (B, N)
        excluded_float = excluded.float()
        
        # Calculate the number of excluded neighbors for each node
        excluded_neighbors = torch.bmm(edges_feature, excluded_float.unsqueeze(-1)).squeeze(-1)  # (B, N)

        # Calculate the remaining degree (undetermined neighbors)
        undetermined = ~mask  # (B, N)
        remaining_degree = torch.bmm(edges_feature, undetermined.float().unsqueeze(-1)).squeeze(-1)  # (B, N)

        # Original degree of each node
        original_degree = edges_feature.sum(dim=2)  # (B, N)

        # Calculate the sum of remaining degrees of neighbors
        neighbor_remaining_sum = torch.bmm(edges_feature, remaining_degree.unsqueeze(-1)).squeeze(-1)  # (B, N)

        # Base heuristic score considering coverage efficiency
        heuristic_score = (excluded_neighbors + 1.0) / (remaining_degree + 1.0)
        
        # Incorporate structural impact of neighbor's remaining degrees
        heuristic_score = heuristic_score * (neighbor_remaining_sum + 1.0) / (original_degree + 1.0)

        # Incorporate a local clustering coefficient to prioritize nodes in dense regions
        clustering_coeff = torch.bmm(edges_feature, edges_feature).sum(dim=2) / (original_degree * (original_degree - 1) + 1e-9)
        heuristic_score = heuristic_score * (1.0 + clustering_coeff)

        # Message passing: aggregate neighbor heuristic scores with attention
        neighbor_heuristic_sum = torch.bmm(edges_feature, heuristic_score.unsqueeze(-1)).squeeze(-1)  # (B, N)
        neighbor_heuristic_avg = neighbor_heuristic_sum / (original_degree + 1e-9)  # Prevent division by zero
        attention_weights = torch.softmax(neighbor_heuristic_sum, dim=1)
        heuristic_score = 0.5 * heuristic_score + 0.5 * (attention_weights * neighbor_heuristic_avg)

        # Incorporate degree centrality with adaptive weights
        centrality = original_degree / original_degree.max(dim=1, keepdim=True)[0]
        t_expanded = t.view(B, 1)
        centrality_weight = 0.2 + 0.1 * torch.cos(t_expanded * 0.1)  # Adaptive centrality weight
        heuristic_score = heuristic_score + centrality_weight * centrality

        # Incorporate a global influence factor based on timestep
        global_influence = torch.exp(-t_expanded * 0.5)  # Higher t reduces global influence
        heuristic_score = global_influence * heuristic_score + (1 - global_influence) * torch.mean(heuristic_score, dim=1, keepdim=True)

        # Timestep-modulated blending with exponential decay
        blend_weight = torch.exp(-5.0 * t_expanded)  # Higher t reduces xt influence
        blended = blend_weight * xt + (1 - blend_weight) * heuristic_score

        # Apply softmax only to undetermined nodes
        undetermined_scores = torch.where(undetermined, blended, torch.tensor(-1e9, device=device))
        prob_undetermined = torch.softmax(undetermined_scores, dim=1)

        # Combine with determined nodes (0 or 1)
        heatmap = torch.where(mask, xt, prob_undetermined)

        return heatmap
```

(17.516, 20.158, 12.990388051621728, 6.310539577449613)

# The upper algorithm + Self-Iteration T (ChatGPT-o3)

```python
def get_heatmap_mis(self, xt: Tensor, edges_feature: Tensor, mask: Tensor, t: Tensor) -> Tensor:
        """
        Improved heatmap evolution function that iteratively refines the solution using a fixed maximum
        number of iterations. The original 't' input is preserved for adaptive weighting, while MAX_ITER is
        a constant used solely to drive the evolution process.
        
        Parameters:
        xt            : Tensor of shape (B, N) representing the initial population.
        edges_feature : Tensor of shape (B, N, N) representing the graph edge features.
        mask          : Tensor of shape (B, N) indicating determined nodes.
        t             : Tensor of shape (B, 1) representing the original time or step factor.
        
        Returns:
        Tensor of shape (B, N) as the final evolved heatmap.
        """
        B, N = xt.shape
        device = xt.device
        # Define a constant maximum iteration count (this constant is internal to the evolution process)
        MAX_ITER = 10  # You can adjust this constant as needed
        
        # Initialize the current solution (population) as a clone of the input
        current = xt.clone()
        
        # Iteratively update the solution for MAX_ITER iterations
        for iter in range(MAX_ITER):
            # Form an effective time factor that blends the original 't' with the iteration index
            # This effective_t modulates the adaptive weights in each iteration
            effective_t = t + iter

            # Identify excluded nodes (masked and with value 0)
            excluded = mask & (current == 0)  # (B, N)
            excluded_float = excluded.float()

            # Calculate the number of excluded neighbors for each node
            excluded_neighbors = torch.bmm(edges_feature, excluded_float.unsqueeze(-1)).squeeze(-1)  # (B, N)

            # Determine undetermined nodes (not masked)
            undetermined = ~mask  # (B, N)
            remaining_degree = torch.bmm(edges_feature, undetermined.float().unsqueeze(-1)).squeeze(-1)  # (B, N)

            # Compute the original degree of each node
            original_degree = edges_feature.sum(dim=2)  # (B, N)

            # Sum of remaining degrees of neighbors
            neighbor_remaining_sum = torch.bmm(edges_feature, remaining_degree.unsqueeze(-1)).squeeze(-1)  # (B, N)

            # Base heuristic score considering coverage efficiency
            heuristic_score = (excluded_neighbors + 1.0) / (remaining_degree + 1.0)
            heuristic_score = heuristic_score * (neighbor_remaining_sum + 1.0) / (original_degree + 1.0)

            # Incorporate a local clustering coefficient to prioritize nodes in dense regions
            clustering_coeff = torch.bmm(edges_feature, edges_feature).sum(dim=2) / (original_degree * (original_degree - 1) + 1e-9)
            heuristic_score = heuristic_score * (1.0 + clustering_coeff)

            # Message passing: aggregate neighbor heuristic scores with attention
            neighbor_heuristic_sum = torch.bmm(edges_feature, heuristic_score.unsqueeze(-1)).squeeze(-1)  # (B, N)
            neighbor_heuristic_avg = neighbor_heuristic_sum / (original_degree + 1e-9)  # Prevent division by zero
            attention_weights = torch.softmax(neighbor_heuristic_sum, dim=1)
            heuristic_score = 0.5 * heuristic_score + 0.5 * (attention_weights * neighbor_heuristic_avg)

            # Incorporate degree centrality with adaptive weights using the effective time factor
            centrality = original_degree / original_degree.max(dim=1, keepdim=True)[0]
            centrality_weight = 0.2 + 0.1 * torch.cos(effective_t * 0.1)
            heuristic_score = heuristic_score + centrality_weight * centrality

            # Incorporate a global influence factor based on the effective time factor
            global_influence = torch.exp(-effective_t * 0.5)
            heuristic_score = global_influence * heuristic_score + (1 - global_influence) * torch.mean(heuristic_score, dim=1, keepdim=True)

            # Timestep-modulated blending with exponential decay between current state and heuristic score
            blend_weight = torch.exp(-5.0 * effective_t)
            blended = blend_weight * current + (1 - blend_weight) * heuristic_score

            # Apply softmax only to undetermined nodes to preserve shape
            undetermined_scores = torch.where(undetermined, blended, torch.tensor(-1e9, device=device))
            prob_undetermined = torch.softmax(undetermined_scores, dim=1)

            # Combine with determined nodes (which retain their current values)
            current = torch.where(mask, current, prob_undetermined)
        
        # The final evolved heatmap preserves the (B, N) shape of the input.
        return current
```



(17.514, 20.158, 13.003158284186012, 6.381028752560134)

# The upper algorithm + Self-Iteration T (DeepSeek-R1)

```python
def get_heatmap_mis(self, xt: torch.Tensor, edges_feature: torch.Tensor, mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, N = xt.shape
        device = xt.device

        excluded = mask & (xt == 0)
        excluded_float = excluded.float()
        
        excluded_neighbors = torch.bmm(edges_feature, excluded_float.unsqueeze(-1)).squeeze(-1)
        undetermined = ~mask
        remaining_degree = torch.bmm(edges_feature, undetermined.float().unsqueeze(-1)).squeeze(-1)
        original_degree = edges_feature.sum(dim=2)
        neighbor_remaining_sum = torch.bmm(edges_feature, remaining_degree.unsqueeze(-1)).squeeze(-1)
        
        heuristic_score = (excluded_neighbors + 1.0) / (remaining_degree + 1.0)
        heuristic_score = heuristic_score * (neighbor_remaining_sum + 1.0) / (original_degree + 1.0)
        
        clustering_coeff = torch.bmm(edges_feature, edges_feature).sum(dim=2) / (original_degree * (original_degree - 1) + 1e-9)
        heuristic_score = heuristic_score * (1.0 + clustering_coeff)
        
        neighbor_heuristic_sum = torch.bmm(edges_feature, heuristic_score.unsqueeze(-1)).squeeze(-1)
        neighbor_heuristic_avg = neighbor_heuristic_sum / (original_degree + 1e-9)
        attention_weights = torch.softmax(neighbor_heuristic_sum, dim=1)
        heuristic_score = 0.5 * heuristic_score + 0.5 * (attention_weights * neighbor_heuristic_avg)
        
        centrality = original_degree / original_degree.max(dim=1, keepdim=True)[0]
        t_expanded = t.view(B, 1)
        centrality_weight = 0.2 + 0.1 * torch.cos(t_expanded * 0.1)
        heuristic_score = heuristic_score + centrality_weight * centrality
        
        global_influence = torch.exp(-t_expanded * 0.5)
        heuristic_score = global_influence * heuristic_score + (1 - global_influence) * torch.mean(heuristic_score, dim=1, keepdim=True)
        
        # Start of Graph Neural Evolution integration
        X_transformed = heuristic_score  # φ(X) applied as heuristic computation
        
        # Compute similarity matrix A via cosine similarity
        X_normalized = F.normalize(X_transformed, p=2, dim=1)
        A = torch.bmm(X_normalized.unsqueeze(2), X_normalized.unsqueeze(1))
        
        # Compute degree matrix D and its inverse square root
        D = torch.diag_embed(A.sum(dim=2))
        D_inv_sqrt = torch.zeros_like(D)
        for b in range(B):
            diag = torch.clamp(D[b].diag(), min=1e-8)
            D_inv_sqrt[b] = torch.diag_embed(diag.pow(-0.5))
        
        # Compute normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
        identity = torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1)
        L = identity - torch.bmm(torch.bmm(D_inv_sqrt, A), D_inv_sqrt)
        
        # Eigendecomposition and filter application (g(Λ) as identity here)
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        g_Lambda = torch.diag_embed(eigenvalues)
        
        # Transform population using eigenvectors and eigenvalues
        X_new = torch.bmm(eigenvectors, torch.bmm(g_Lambda, torch.bmm(eigenvectors.transpose(1, 2), X_transformed.unsqueeze(2)))).squeeze(2)
        X_new = torch.tanh(X_new)  # ϕ(X_new) applied as nonlinear transformation
        
        heuristic_score = X_new  # Updated heuristic_score after spectral processing
        # End of integration
        
        blend_weight = torch.exp(-5.0 * t_expanded)
        blended = blend_weight * xt + (1 - blend_weight) * heuristic_score
        
        undetermined_scores = torch.where(undetermined, blended, torch.tensor(-1e9, device=device))
        prob_undetermined = torch.softmax(undetermined_scores, dim=1)
        heatmap = torch.where(mask, xt, prob_undetermined)
        
        return heatmap
```



(17.518, 20.158, 12.977891659125337, 6.331099779762719)