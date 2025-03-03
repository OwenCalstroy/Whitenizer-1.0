# Prompt As Follows

**Task**: Design a heuristic for Maximum Independent Set (MIS) on an undirected graph G=(V,E).

**Requirements:** Rule-based, iterative score updates with constraint propagation, and scalable to any graph.

Output a concise, step-by-step description of your algorithm.

The whole process is a diffusion process. In each iteration of the diffusion, we updated solutions as follows:

1. **Input Preparation**:  
   - For each node in the graph \( G \):
     - If the node is **determined** (already assigned to the solution or excluded), set its state \( x_t(i) \) to 0 or 1.  
     - If the node is **undetermined**, set \( x_t(i) \) to a randomly sampled noise value.  
   - Combine the current state \( x_t \), timestep \( t \), and graph \( G \) as input to the GNN.

2. **GNN Inference**:  
   - The GNN processes the input and outputs a **heatmap** (probability vector \( A \)), where \( A_i \) indicates the likelihood of node \( i \) being part of the Maximum Independent Set (MIS).

3. **Greedy Decoding**:  
   - Sort nodes in descending order of their heatmap probabilities \( A_i \).  
   - Iteratively select nodes from the sorted list:  
     - If adding a node to the solution does **not conflict** with existing selected nodes (i.e., no shared edges), mark it as **chosen (1)**.  
     - Immediately exclude its neighboring nodes from being chosen (mark them as **determined to 0**).  
   - Halt decoding if a conflict arises or all nodes are processed.

4. **Partial Solution Update**:  
   - Update the state \( x_t \) by fixing the determined nodes (0 or 1) based on the decoding results.  
   - Retain random noise for undetermined nodes in the next iteration.

5. **Termination Check**:  
   - If all nodes are determined (no undetermined nodes remain), output the final MIS.  
   - Otherwise, repeat steps 1–4 with the updated partial solution until termination.

**Key Notes**:  
- The GNN leverages a **diffusion model framework**, where \( t \) controls the noise level or denoising progress. In diffusion model, \(t\) represents the steps used. 
- Conflicts are resolved **proactively**: neighbors of selected nodes are excluded immediately to maintain independence constraints.  
- The process iteratively refines the solution by alternating between GNN-guided probability estimation and constraint-aware greedy selection.

Help me design an algorithm that imitate the behavior of the denoising GNN. Then implement your algorithm and outputs a piece of executable python code only. You have to pay attention to its computational cost. Your code should starts with:

```py
def get_heatmap_mis(
        self, xt: Tensor, edges_feature: Tensor, mask: Tensor, t: Tensor
    ) -> Tensor:
```

**Task:** Improve the baseline to generate a **heuristic** that mimics a GNN-based diffusion process for Maximum Independent Set (MIS), preserving tensor shapes:  

**Inputs** *(critical shapes)*:  
- `edges_feature`: Adjacency matrix *(B, N, N)*  
- `xt`: Node states (0/1 for determined, noise for undetermined) *(B, N)*  
- `t`: Diffusion timestep *(B)*  
- `mask`: Undetermined nodes *(B, N, bool)*  

**Requirements**:  
1. **Diffusion**: Apply cosine-decayed noise to undetermined nodes (`noise *= 0.5*(1 + cos(π*t/max_t))`).  
2. **Conflict Scoring**: Compute `neighbor_conflicts = edges_feature @ xt` *(B, N)*.  
3. **Degree Penalty**: Subtract `node_degree = edges_feature.sum(-1)` *(B, N)*.  
4. **Shape Preservation**: Update `xt` only where `mask=True`, retaining all dimensions.  

**Baseline Code**:  

```python  
def heuristic(edges_feature, xt, t, mask, max_t):  
    noise = randn(B, N) * 0.5 * (1 + cos(np.pi * t / max_t))  
    conflicts = (edges_feature @ xt.unsqueeze(-1)).squeeze()  # (B, N)  
    degree = edges_feature.sum(-1)  # (B, N)  
    scores = -conflicts - degree  
    return torch.where(mask, xt, scores + noise)  # (B, N)  
```

# The Answer

```py
import torch
import math

class MISHeuristicModel:
    def __init__(self, max_t=1000):
        self.max_t = max_t

    def get_heatmap_mis(
        self, xt: torch.Tensor, edges_feature: torch.Tensor, mask: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        # Compute the cosine decay factor per batch element
        decay_factor = 0.5 * (1 + torch.cos(math.pi * t.to(xt.dtype) / self.max_t))  # (B, )
        # Generate Gaussian noise scaled by the decay factor
        noise = torch.randn_like(xt) * decay_factor.unsqueeze(-1)  # (B, N)
        # Calculate conflict scores: sum of selected neighbors
        conflicts = torch.matmul(edges_feature, xt.unsqueeze(-1)).squeeze(-1)  # (B, N)
        # Compute node degrees as a penalty
        degree = edges_feature.sum(dim=-1)  # (B, N)
        # Combine scores and apply noise where nodes are undetermined
        scores = -conflicts - degree
        heatmap = torch.where(mask, scores + noise, xt)
        return heatmap
```

# A small CHANGE of the prompt

**Requirements**: 

1. Greedy should be your final choice. If you have another idea, please abandon the simple Greedy method.
2. You may consider to use the genetic algorithm as the heuristic, because this has been proofed effective in representing GNNs.


**Baseline Code**: 

```py
def get_heatmap_mis(
    self, xt: Tensor, edges_feature: Tensor, mask: Tensor, t: Tensor
  ) -> Tensor:
   heatmap = torch.ones(B, N)
   return heatmap
```

# Answer
```python
import torch
import math

class MISHeuristicModel:
    def __init__(self, max_t=1000, pop_size=50, num_generations=5):
        self.max_t = max_t
        self.pop_size = pop_size        # Population size per batch
        self.num_generations = num_generations
        self.mutation_rate = 0.1

    def get_heatmap_mis(
        self, xt: torch.Tensor, edges_feature: torch.Tensor, mask: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        B, N = xt.shape
        
        # Genetic Algorithm parameters
        P = self.pop_size
        G = self.num_generations
        
        # Initialize population with determined nodes fixed
        population = torch.where(
            mask.unsqueeze(1).expand(-1, P, -1),
            torch.rand(B, P, N, device=xt.device) < 0.5,
            xt.unsqueeze(1).expand(-1, P, -1).bool()
        ).float()  # (B, P, N)
    
        # Genetic evolution loop
        for _ in range(G):
            # Fitness: maximize selected nodes while minimizing conflicts
            selected = population
            neighbor_conflicts = (selected @ edges_feature) * selected  # (B, P, N)
            conflict_scores = neighbor_conflicts.sum(-1) / 2  # Undirected edge correction
            fitness = selected.sum(-1) - conflict_scores  # (B, P)
            
            # Penalize invalid solutions (violate determined nodes)
            valid_mask = (population * (1 - xt.unsqueeze(1))).sum(-1) == 0
            valid_mask &= ((xt.unsqueeze(1) - population).abs() * (xt.unsqueeze(1) > 0.5)).sum(-1) == 0
            fitness[~valid_mask] = -1e9  # Invalidate incorrect solutions
            
            # Tournament selection (top 50%)
            _, top_idx = torch.topk(fitness, P//2, dim=1)  # (B, P//2)
            parents = torch.gather(population, 1, top_idx.unsqueeze(-1).expand(-1, -1, N))
            
            # Crossover (single-point) and mutation
            crossover_pt = torch.randint(N, (B, P//2, 1), device=xt.device)
            crossover_mask = torch.arange(N, device=xt.device).view(1,1,N) < crossover_pt
            parents1, parents2 = parents.chunk(2, dim=1)
            children = torch.where(crossover_mask, parents1, parents2)
            
            # Apply mutation only on undetermined nodes
            mutation_mask = (torch.rand(B, P//2, N, device=xt.device) < self.mutation_rate) & mask.unsqueeze(1)
            children = torch.where(mutation_mask, 1 - children, children)
            
            # Combine parents and offspring
            population = torch.cat([parents, children], dim=1)
    
        # Generate heatmap from population statistics
        heatmap = population.mean(dim=1)  # (B, N)
        
        # Preserve determined nodes and apply cosine noise decay
        decay = 0.5 * (1 + torch.cos(math.pi * t.float() / self.max_t))  # (B,)
        noise = torch.randn_like(heatmap) * decay.view(-1, 1)
        heatmap = torch.where(mask, heatmap + noise, xt)
        
        return torch.clamp(heatmap, 0, 1)  # Ensure valid probability range
```

```python
import torch
import math

class MISHeuristicModel:
    def __init__(self, max_t=1000, pop_size=20, num_generations=3):
        self.max_t = max_t
        self.pop_size = pop_size          # Population per batch instance
        self.num_generations = num_generations

    def get_heatmap_mis(
        self, xt: torch.Tensor, edges_feature: torch.Tensor, mask: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        B, N = xt.shape
        
        # === Diffusion Core: Noise Scheduling ===
        decay = 0.5 * (1 + torch.cos(math.pi * t.float() / self.max_t))  # (B,)
        
        # === Genetic Algorithm as Diffusion Surrogate ===
        # Initialize population preserving determined nodes
        population = torch.where(
            mask.unsqueeze(1).expand(-1, self.pop_size, -1),
            torch.rand(B, self.pop_size, N, device=xt.device) < 0.5,
            xt.unsqueeze(1).expand(-1, self.pop_size, -1).bool()
        ).float()  # (B, P, N)

        # Genetic evolution with diffusion-aware mutation
        for _ in range(self.num_generations):
            # Fitness: Selected nodes - conflicts (matrix implementation)
            selected = population
            conflicts = (selected @ edges_feature) * selected  # (B, P, N)
            fitness = selected.sum(-1) - conflicts.sum(-1)/2  # (B, P)
            
            # Enforce determined nodes constraints
            invalid = (population * (1 - xt.unsqueeze(1))).sum(-1) > 0
            fitness[invalid] = -1e9  # Hard constraint penalty
            
            # Tournament selection (top 50%)
            _, top_idx = torch.topk(fitness, self.pop_size//2, dim=1)  # (B, P//2)
            parents = torch.gather(population, 1, top_idx.unsqueeze(-1).expand(-1, -1, N))
            
            # Crossover with diffusion-step dependent randomness
            cross_mask = torch.rand(B, self.pop_size//2, 1, device=xt.device) < decay.view(-1,1,1)
            children = torch.where(cross_mask, parents[..., :N//2], parents[..., N//2:])
            
            # Mutation guided by diffusion noise schedule
            mutation_rate = 0.2 * (1 - decay)  # Noise decays as t increases
            mutation_mask = (torch.rand(B, self.pop_size//2, N, device=xt.device) < mutation_rate.view(-1,1,1)) & mask.unsqueeze(1)
            children = torch.where(mutation_mask, 1 - children, children)
            
            population = torch.cat([parents, children], dim=1)

        # === Diffusion-aligned Heatmap Construction ===
        heatmap = population.mean(dim=1)  # (B, N)
        heatmap = torch.where(mask, heatmap * decay.view(-1,1), xt)  # Noise mixing
        
        # Immediate neighbor exclusion (constraint propagation)
        with torch.no_grad():
            selected = (heatmap > 0.5).float()
            conflict_neighbors = (selected @ edges_feature).clamp(max=1)
            heatmap = torch.where(conflict_neighbors.bool(), torch.zeros_like(heatmap), heatmap)
        
        return heatmap
```

### Seems that forcing it to use GA do not work well.

### We realize that it is mimicking the Diffusion, not the GNN which is inside the diffusion. Here is the changed prompt.

# Prompt

**Task Background**: Design a rule-based heuristic to replace the neural network in neural solvers for unweighted Maximum Independent Set (MIS) on a graph.

**Pipeline**:

The whole process of the neural solver is a diffusion process. In each iteration of the diffusion, we updated solutions as follows:

1. **Input Preparation**:  
   - For each node in the graph \( G \):
     - If the node is **determined** (already assigned to the solution or excluded), set its state \( x_t(i) \) to 0 or 1.  
     - If the node is **undetermined**, set \( x_t(i) \) to a noise value randomly sampled from Bernoulli distribution with p = 0.5.  
   - Combine the current state \( x_t \), timestep \( t \), and graph \( G \) as input to the GNN.
2. **GNN Inference**:  
   - The GNN processes the input and outputs a **heatmap** (probability vector \( A \)), where \( A_i \) indicates the likelihood of node \( i \) being part of the MIS.
3. **Greedy Decoding**:  
   - Sort nodes in descending order of their heatmap probabilities \( A_i \).  
   - Iteratively select nodes from the sorted list:  
     - If adding a node to the solution does **not conflict** with existing selected nodes (i.e., no shared edges), mark it as **chosen (1)**.  
     - Immediately exclude its neighboring nodes from being chosen (mark them as **determined to 0**).  
   - Halt decoding if a conflict arises or all nodes are processed.
4. **Partial Solution Update**:  
   - Update the state \( x_t \) by fixing the determined nodes (0 or 1) based on the decoding results.  
5. **Termination Check**:  
   - If all nodes are determined (no undetermined nodes remain), output the final MIS.  
   - Otherwise, repeat steps 1–4 with the updated partial solution until termination.

**Key Notes**:  

- The GNN leverages a **diffusion model framework**, where \( t \) controls the noise level or denoising progress. In diffusion model, \(t\) represents the steps used. 
- Conflicts are resolved **proactively**: neighbors of selected nodes are excluded immediately to maintain independence constraints.  
- The process iteratively refines the solution by alternating between GNN-guided probability estimation and constraint-aware greedy selection.

**Your Specific Task:** Design a **heuristic** that mimics the denoising GNN, preserving tensor shapes, then implement your algorithm. You have to pay attention to its *computational cost*.

**Inputs** *(critical shapes)*:  

- `edges_feature`: Adjacency matrix *(B, N, N)*  
- `xt`: Node states (0/1 for determined, noise for undetermined) *(B, N)*  
- `t`: Diffusion timestep *(B)*  
- `mask`: Undetermined nodes *(B, N, bool)*  

**Requirements**:  

1. **Make use of Timestep**: You must figure out the meaning of the diffusion timestep and make use of it.
2. **Make use of Determined Nodes**: You must use the determined nodes to determine the undetermined ones.
3. **Mimic GNN**: You should mimic the GNN process as much as possible.
4. **Shape Preservation**: Update `xt` only where `mask=False`.  

**External Knowledge**:

1. The ARW local search algorithm performs well on MIS.

**Outputs**:

1. **A brief introduction to your algorithm**: It should be concise.
2. **A Python function with comments**: It should start with:

```py
def get_heatmap_mis(
        self, xt: Tensor, edges_feature: Tensor, mask: Tensor, t: Tensor
    ) -> Tensor:
```

**Baseline Code**:  (Just a simple greedy algorithm which doesn't make use of timestep)

```python  
def get_heatmap_mis(
        self, xt: Tensor, edges_feature: Tensor, mask: Tensor, t: Tensor
    ) -> Tensor:
    degree = edges_feature.sum(-1)  # (B, N)  
    scores = 1 / degree  
    return torch.where(mask, xt, scores)  # (B, N)  
```

# Answer

## ChatGPT-o3mini
```python
def get_heatmap_mis(
        self, xt: Tensor, edges_feature: Tensor, mask: Tensor, t: Tensor
    ) -> Tensor:
    """
    Compute a heatmap for MIS using a rule-based heuristic that mimics the denoising GNN.
    

    Args:
        xt (Tensor): (B, N) current node states (0/1 for determined, noise for undetermined)
        edges_feature (Tensor): (B, N, N) adjacency matrix of the graph.
        mask (Tensor): (B, N) boolean mask for undetermined nodes (True means undetermined).
        t (Tensor): (B) diffusion timesteps.
    
    Returns:
        Tensor: (B, N) heatmap probabilities for each node.
    
    The heuristic works as follows:
      1. Compute a structural score for each node based on its degree (lower degree -> higher score).
      2. For determined nodes (mask==False), aggregate the number of neighbors that have already been selected (xt==1) to penalize conflicts.
      3. Use the timestep t to compute a factor that smoothly shifts the algorithm from a noisy stage (high t)
         to a more refined, structure–driven stage (low t).
      4. Combine these pieces: a weighted sum of the structural score and the current state minus a penalty for conflicting neighbors.
      5. Update only the determined nodes, leaving undetermined nodes (mask==True) unchanged.
    """
    import torch
    # Compute the degree of each node: lower degree -> potentially better candidate for MIS.
    degree = edges_feature.sum(dim=-1)  # (B, N)
    struct_score = 1.0 / (degree + 1.0)   # Avoid division by zero; (B, N)
    
    # Determine which nodes are already determined (mask==False).
    determined = ~mask  # (B, N)
    
    # For determined nodes, find those that have been selected in the MIS.
    selected = (xt == 1) & determined  # (B, N); these nodes have been chosen.
    
    # Compute a penalty for each node based on the number of selected neighbors.
    # For each node, summing over its neighbors gives higher penalty for nodes with many conflicts.
    penalty = torch.matmul(edges_feature, selected.float().unsqueeze(-1)).squeeze(-1)  # (B, N)
    
    # Use the timestep to modulate the influence of the structural score vs. the current (noisy) state.
    # Here we assume t is normalized (e.g., in [0,1]) where t near 1 is early (more noise) and near 0 is later.
    # The factor decreases with t so that later timesteps favor the structural/conflict–based score.
    factor = 1.0 / (1.0 + t.unsqueeze(-1))  # (B, 1)
    
    # Combine scores:
    # - factor * struct_score: favors low-degree nodes at low t.
    # - (1 - factor) * xt: carries forward the current state when t is high.
    # - Subtracting factor * penalty: reduces the heat for nodes with many selected neighbors.
    heat = factor * struct_score + (1 - factor) * xt.float() - factor * penalty
    
    # Update only the determined nodes (mask==False); leave undetermined nodes unchanged.
    # This preserves the tensor shape as required.
    new_heatmap = torch.where(determined, heat, xt.float())
    
    return new_heatmap
```

**RESULT** (16.14, 20.158, **19.78314967891324**, 6.831044059006039)

## DeepSeek-R1

```python
import torch
from torch import Tensor

def get_heatmap_mis(
        self, xt: Tensor, edges_feature: Tensor, mask: Tensor, t: Tensor
    ) -> Tensor:
    # Compute node degree (B, N)
    degree = edges_feature.sum(dim=-1)
    
    # Check if any neighbor is selected (conflict)
    conflict = (edges_feature @ (xt == 1).float().unsqueeze(-1)).squeeze(-1) > 0
    
    # Count undetermined neighbors (B, N)
    undetermined_neighbors = (edges_feature * mask.unsqueeze(1).float()).sum(-1)
    
    # Avoid division by zero with small epsilon
    current_score = 1.0 / (undetermined_neighbors + 1e-8)  # Dynamic neighbor heuristic
    degree_score = 1.0 / (degree + 1e-8)                   # Baseline inverse degree
    
    # Weight terms using timestep t (earlier: degree, later: current state)
    t_weight = t.float().unsqueeze(-1)
    combined_score = (t_weight / (t_weight + 1)) * degree_score + \
                     (1 / (t_weight + 1)) * current_score
    
    # Zero score for conflicting nodes
    score = torch.where(conflict, torch.zeros_like(combined_score), combined_score)
    
    # Preserve determined nodes' state, update undetermined with scores
    heatmap = torch.where(mask, score, xt)
    return heatmap
```

## Kimi-1.5
```python
def get_heatmap_mis(
        self, xt: Tensor, edges_feature: Tensor, mask: Tensor, t: Tensor
    ) -> Tensor:
    # Compute degree for each node (B, N)
    degree = edges_feature.sum(dim=-1)
    
    # Compute conflict_mask: neighbors that are determined and in the solution (B, N, N)
    conflict_mask = edges_feature * (xt.unsqueeze(-1) == 1) * (~mask.unsqueeze(-1))
    conflict_count = conflict_mask.sum(dim=-1)  # (B, N)
    
    # Compute excluded_mask: neighbors that are determined and excluded (B, N, N)
    excluded_mask = edges_feature * (xt.unsqueeze(-1) == 0) * (~mask.unsqueeze(-1))
    excluded_count = excluded_mask.sum(dim=-1)  # (B, N)
    
    # Initialize score tensor with zeros (B, N)
    score = torch.zeros_like(xt)
    
    # For undetermined nodes (mask is True)
    undetermined = mask
    valid = undetermined & (conflict_count == 0)
    
    if valid.any():
        # Normalize timestep to [0, 1] per batch element
        t_normalized = t / (t.max() + 1e-6)
        # Compute score for valid nodes: higher excluded_count and lower degree preferred
        score[valid] = (excluded_count[valid] / (degree[valid] + 1)) * (1 + t_normalized[valid])
    
    # For determined nodes, use their current state
    score[~undetermined] = xt[~undetermined]
    
    return score
```

## Claude-3 Haiku
```python
import torch

def get_heatmap_mis(self, xt: torch.Tensor, edges_feature: torch.Tensor, mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Compute the heatmap for the Maximum Independent Set (MIS) problem.

    Args:
        xt (torch.Tensor): Node states (0/1 for determined, noise for undetermined) of shape (B, N).
        edges_feature (torch.Tensor): Adjacency matrix of the graph of shape (B, N, N).
        mask (torch.Tensor): Undetermined nodes of shape (B, N, bool).
        t (torch.Tensor): Diffusion timestep of shape (B).

    Returns:
        torch.Tensor: Heatmap of shape (B, N), where each element represents the likelihood of a node being part of the MIS.
    """
    B, N = xt.shape
    device = xt.device

    # Compute node degrees
    degree = edges_feature.sum(-1)  # (B, N)

    # Define thresholds for different timestep ranges
    T1 = 10
    T2 = 20

    # Initialize heatmap
    heatmap = torch.zeros_like(xt)

    for b in range(B):
        # Early timesteps: select high-degree undetermined nodes
        if t[b] < T1:
            _, indices = torch.topk(degree[b] * mask[b], k=int(mask[b].sum() * 0.5), dim=0)
            heatmap[b, indices] = 1.0

        # Intermediate timesteps: select low-degree undetermined nodes
        elif t[b] < T2:
            _, indices = torch.topk(-degree[b] * mask[b], k=int(mask[b].sum() * 0.5), dim=0)
            heatmap[b, indices] = 1.0

        # Late timesteps: select undetermined nodes with highest scores
        else:
            scores = xt[b] + 1.0 / degree[b]
            _, indices = torch.topk(scores * mask[b], k=int(mask[b].sum() * 0.5), dim=0)
            heatmap[b, indices] = 1.0

    # Update xt only where mask=False
    xt = torch.where(mask, xt, heatmap)
    return xt
```

## Gemini 2.0 Flash
```python
import torch
from torch import Tensor
import torch.nn.functional as F

class HeuristicMIS:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta

    def get_heatmap_mis(
            self, xt: Tensor, edges_feature: Tensor, mask: Tensor, t: Tensor
        ) -> Tensor:
        """
        Heuristic to estimate the probability of each node being in the MIS.

        This heuristic mimics the GNN in the diffusion process for MIS. It leverages
        the timestep (t), determined node states (xt), and graph structure (edges_feature)
        to generate a heatmap indicating the likelihood of each node belonging to the MIS.

        The algorithm combines degree-based scoring with timestep-dependent adjustment
        and determined node information to guide the greedy decoding process.  It also
        incorporates a local search inspired component.

        Args:
            xt (Tensor): Node states (0/1 for determined, noise for undetermined) (B, N)
            edges_feature (Tensor): Adjacency matrix (B, N, N)
            mask (Tensor): Undetermined nodes (B, N, bool)
            t (Tensor): Diffusion timestep (B)

        Returns:
            Tensor: Heatmap (probability vector) indicating the likelihood of each node
                    being part of the MIS (B, N)
        """
        B, N = xt.shape

        # 1. Degree-based Scoring:  Nodes with lower degrees are generally more likely to be in the MIS.
        degree = edges_feature.sum(-1)  # (B, N)
        degree = torch.where(degree == 0, torch.ones_like(degree), degree) # Avoid division by zero
        scores = 1 / degree  # (B, N)

        # 2. Timestep-Dependent Adjustment:  As the diffusion process progresses (t increases),
        #    the heuristic becomes more confident in its estimations.  We can use t to scale the scores.
        #    Early timesteps (small t) might benefit from more exploration (lower scores), while later
        #    timesteps (large t) can exploit the accumulated information (higher scores).
        t_normalized = t / t.max()  # Normalize timestep to [0, 1]
        t_scaling = 1 + self.alpha * t_normalized.unsqueeze(-1)  # (B, 1)  Scale scores based on timestep
        scores = scores * t_scaling  # (B, N)

        # 3. Incorporate Determined Node Information:  Use the determined node states (xt) to influence
        #    the scores of undetermined nodes.  For example, if a node's neighbor is already in the MIS (xt=1),
        #    its score should be reduced.  Conversely, if a node's neighbor is excluded (xt=0), its score
        #    could be slightly increased.
        neighbor_influence = torch.matmul(edges_feature, xt.unsqueeze(-1)).squeeze(-1)  # (B, N)
        # Reduce scores of nodes with neighbors already in the MIS
        scores = scores - self.beta * neighbor_influence

        # 4. Local Search Inspired Component: Encourage nodes with excluded neighbors
        excluded_neighbor_influence = torch.matmul(edges_feature, (1 - xt).unsqueeze(-1)).squeeze(-1)
        scores = scores + 0.1 * excluded_neighbor_influence

        # 5. Masking:  Only update scores for undetermined nodes (where mask is True).  For determined nodes,
        #    keep their state (0 or 1).
        return torch.where(mask, scores, xt)  # (B, N)
```