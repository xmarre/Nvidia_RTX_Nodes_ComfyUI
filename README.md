# Nvidia_RTX_Nodes_ComfyUI

Contains the **RTX Video Super Resolution** node for upscaling images and videos with NVIDIA RTX Video Effects. Only NVIDIA RTX GPUs are supported.

Search **RTX** in the ComfyUI Manager to install it.

## Memory behavior

High-resolution video outputs can be much larger than their inputs. The node processes VSR one frame at a time so CUDA scratch memory does not scale with an arbitrary frame batch.

`output_device` controls where the complete returned `IMAGE` tensor lives:

- `auto` (default): keeps the result on CUDA when enough free VRAM exists for the full output plus conservative scratch/headroom; otherwise uses CPU.
- `cuda`: always keep the complete result in VRAM. This is useful for large video chains when VRAM is plentiful and avoids materializing another full-resolution tensor in system RAM.
- `cpu`: preserve host-memory output behavior.

Common video encoders such as VideoHelperSuite consume CUDA `IMAGE` tensors frame by frame and transfer individual frames to CPU while encoding, so a CUDA result does not require a second full-video CPU copy there.
