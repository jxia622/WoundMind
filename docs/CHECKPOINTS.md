# Checkpoint Placement

WoundMind expects local model weights under `checkpoints/`. These files are large and are intentionally ignored by git.

Copy the current local checkpoints into this layout:

```text
checkpoints/
  convnext_tiny_best.pt
  unetpp_efficientnet_b0_512.pt
  DFU_convnext_tiny/
    convnext_tiny_best.pt
  DFU_convnext_tiny_mask4/
    convnext_tiny_mask4_best.pt
  DFU_convnext_tiny_depth4/
    convnext_tiny_depth4_best.pt
  DFU_convnext_tiny_depth_mask5/
    convnext_tiny_depth_mask5_best.pt
  PI_convnext_tiny/
    convnext_tiny_pi_best.pt
  PI_convnext_tiny_mask4/
    convnext_tiny_mask4_pi_best.pt
  PI_convnext_tiny_depth4/
    convnext_tiny_depth4_pi_best.pt
  PI_convnext_tiny_depth_mask5/
    convnext_tiny_depth_mask5_pi_best.pt
```

In the source workspace, these files came from:

```text
/Users/jackxia/Desktop/Python/DFU Project/Condition Classification/condition-classifier-deployment/checkpoints/
```

The total checkpoint directory is about 2.6 GB, so it should be handled with external artifact storage rather than normal git history.
