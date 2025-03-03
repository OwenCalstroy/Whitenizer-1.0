# TSP General Prompt

Now we have a TSP task. Instead of directly solving it, we want to solve it in a novel way: designing heuristics to imitate the neural TSP solver.
Here’s the pipeline of the neural solver:
The TSP solving task is separated into two parts: training stage and decision stage. 
The training stage was initially accomplished by a Consistency-Model-guided GNN which is inherently a denoising network. This is what you’ll imitate. 
And the decision stage is an autoregressive structure like Mask-Predict. In each iteration, we updated solutions as follows:

1. We get a partial solution with determined edges and undetermined edges from the last iteration.
2. We directly feed (xt, t, G) into the GNN, where xt(i) equals a randomly-sampled noise when edge (i) is undetermined and it equals the determined state (0 or 1) when edge (i) is determined, t is the timestep same as the diffusion model and G denotes the problem instance. The GNN will generate a solution heatmap (a (n, n) shape vector A, where Aij denotes the possibility that the final solution contains an edge between node i and node j). 
3. The heatmap is decoded greedily until a conflict takes place. Specifically, we sort all the possible edges in decreasing order of the inferred likelihood to be chosen and choose each edge until the choice won’t satisfy the constraint of TSP. (It’s worth mentioning that once we determine that an edge should be chosen, the nodes that are linked to it should not have a node degree better than 2)  If all edges are determined, we jump out of the iteration.
4. The partial solution is updated accordingly.

Finally, we get the optimal solution. Note that the decision stage is for your reference only, you’re not allowed to revise this process.
Help me design an algorithm that imitate the behavior of the denoising GNN. Then implement your algorithm and outputs a piece of executable python code only. You have to pay attention to its computational cost.
Your code should starts with:

```py
def get_heatmap_tsp(
        nodes_feature: torch.Tensor, xt: torch.Tensor, graph: torch.Tensor, 
        t: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
```

In the `get_heatmap_tsp` function, the shape of `nodes_feature` is (B, N, C), where B is the batch size, N is the number of nodes, and C is the dimension of node features. The shape of `xt` is (B, N, N), representing the feature matrix between nodes. The shape of `graph` is (B, N, N), representing the adjacency matrix of the graph. The shape of `t` is (B, 1), representing the time step. The shape of `mask` is (B, N, N), representing the mask matrix used to indicate which edges have been determined. The output of the function, `heatmap`, has the shape (B, N, N), representing the generated heatmap that indicates the connection probabilities between nodes.
Warning: your algorithm MUST combine both knowledge of the TSP problem and the diffusion process.
We offer you the baseline, please adjust the following code to make it closer to the function of GNN, which should result a lot better than this baseline.

```python
# baseline as follows:
def get_heatmap_tsp(
        nodes_feature: torch.Tensor, xt: torch.Tensor, graph: torch.Tensor, 
        t: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
    return torch.where(mask, xt, (1 / graph).clamp(0,1))
```

