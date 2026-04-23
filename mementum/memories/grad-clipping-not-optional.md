❌ When porting a model between frameworks (PyTorch→MLX), gradient
clipping is not a nicety — it's load-bearing. v5 had clip_grad_norm_
(1.0) buried at line 354. v6 omitted it. Result: embedding weights
diverged in ~400 steps. Tied weight matrices (embed = output projection)
create positive feedback loops that are invisible until they explode.
Always grep the source model's training script for `clip_grad` before
declaring a port complete.
