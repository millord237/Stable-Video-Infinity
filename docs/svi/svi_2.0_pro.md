### [10 Dec 2025] Difference of SVI 2.0 Pro

Compared with **SVI 2.0**, this **Pro version** introduces three key improvements:

1. **Anchor Redesign**  
   Redesigned the anchor frame mechanism to potentially alleviate its conflict with the **lightx2v LoRA**.

2. **Latent Conditioning**  
   Replaces the last-frame conditioning with **last-latent conditioning**, avoiding repeated encoding/decoding of the last frames.

3. **Data Scaling**  
   Expands the training set by adding additional videos generated from closed-source models, improving data diversity and robustness.

The architectural design is as follows. For each generation, the input latent is the concatation of:

- the **first-frame latent** (shared across all clips),
- the **last latent from the previous clip**,
- followed by **zero-value latents**.

<p align="center">
  <img src="svi_pro.png" alt="SVI" width="900"/>
</p>



