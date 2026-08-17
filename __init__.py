import torch
import nvvfx
from enum import Enum
from typing import TypedDict
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


class UpscaleType(str, Enum):
    SCALE_BY = "scale by multiplier"
    TARGET_DIMENSIONS = "target dimensions"


class OutputDevice(str, Enum):
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


_GIB = 1024**3
_AUTO_CUDA_MIN_HEADROOM = 2 * _GIB
_AUTO_CUDA_HEADROOM_FRACTION = 0.10


def _tensor_nbytes(shape, dtype: torch.dtype) -> int:
    elements = 1
    for dimension in shape:
        elements *= int(dimension)
    return elements * torch.empty((), dtype=dtype).element_size()


def _resolve_output_device(
    images: torch.Tensor,
    output_shape: tuple[int, int, int, int],
    preference: str,
) -> torch.device:
    preference = str(preference).lower()
    if preference == OutputDevice.CPU.value:
        return torch.device("cpu")

    if not torch.cuda.is_available():
        if preference == OutputDevice.CUDA.value:
            raise RuntimeError("RTX Video Super Resolution output_device=cuda requires CUDA")
        return torch.device("cpu")

    cuda_device = images.device if images.device.type == "cuda" else torch.device("cuda")
    if preference == OutputDevice.CUDA.value:
        return cuda_device
    if preference != OutputDevice.AUTO.value:
        raise ValueError(f"Unsupported output device: {preference}")

    # Keeping an already-CUDA IMAGE on the same device avoids a pointless full-video
    # round trip through system RAM. For CPU inputs, only choose CUDA when the full
    # result, one float32 input/output scratch frame, and a conservative CUDA
    # headroom all fit in currently free VRAM.
    if images.device.type == "cuda":
        return cuda_device

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(cuda_device)
    except Exception:
        return torch.device("cpu")

    _, output_height, output_width, channels = output_shape
    output_bytes = _tensor_nbytes(output_shape, images.dtype)
    scratch_bytes = (
        int(images.shape[1]) * int(images.shape[2]) * int(images.shape[3]) * 4
        + output_height * output_width * channels * 4
    )
    headroom = max(
        _AUTO_CUDA_MIN_HEADROOM,
        int(total_bytes * _AUTO_CUDA_HEADROOM_FRACTION),
    )
    if free_bytes >= output_bytes + scratch_bytes + headroom:
        return cuda_device
    return torch.device("cpu")


class RTXVideoSuperResolution(io.ComfyNode):
    class UpscaleTypedDict(TypedDict):
        resize_type: UpscaleType
        scale: float
        width: int
        height: int

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RTXVideoSuperResolution",
            display_name="RTX Video Super Resolution",
            category="image/upscaling",
            search_aliases=["rtx", "nvidia", "upscale", "super resolution", "vsr"],
            inputs=[
                io.Image.Input("images"),
                io.DynamicCombo.Input(
                    "resize_type",
                    tooltip="Choose to scale by a multiplier or to exact target dimensions.",
                    options=[
                        io.DynamicCombo.Option(UpscaleType.SCALE_BY, [
                            io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.01, tooltip="Scale factor (e.g., 2.0 doubles the size)."),
                        ]),
                        io.DynamicCombo.Option(UpscaleType.TARGET_DIMENSIONS, [
                            io.Int.Input("width", default=1920, min=64, max=8192, step=8, tooltip="Target width in pixels."),
                            io.Int.Input("height", default=1080, min=64, max=8192, step=8, tooltip="Target height in pixels.")
                        ])
                    ],
                ),
                io.Combo.Input("quality", options=["LOW", "MEDIUM", "HIGH", "ULTRA"], default="ULTRA"),
                io.Combo.Input(
                    "output_device",
                    options=[
                        OutputDevice.AUTO.value,
                        OutputDevice.CUDA.value,
                        OutputDevice.CPU.value,
                    ],
                    default=OutputDevice.AUTO.value,
                    tooltip=(
                        "auto keeps the complete upscaled video in VRAM when enough CUDA memory "
                        "is free, avoiding a second full-resolution system-RAM tensor. cpu keeps "
                        "legacy host-memory output behavior."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output("upscaled_images"),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        resize_type: UpscaleTypedDict,
        quality: str,
        output_device: str = OutputDevice.AUTO.value,
    ) -> io.NodeOutput:
        if not torch.is_tensor(images) or images.ndim != 4:
            raise ValueError("images must be an IMAGE tensor [frames,height,width,channels]")

        frame_count, h, w, c = images.shape
        if frame_count < 1:
            return io.NodeOutput(images)

        selected_type = resize_type["resize_type"]
        if selected_type == UpscaleType.SCALE_BY:
            scale = resize_type["scale"]
            output_width = int(w * scale)
            output_height = int(h * scale)
        elif selected_type == UpscaleType.TARGET_DIMENSIONS:
            output_width = resize_type["width"]
            output_height = resize_type["height"]
        else:
            raise ValueError(f"Unsupported resize type: {selected_type}")

        output_width = max(8, round(output_width / 8) * 8)
        output_height = max(8, round(output_height / 8) * 8)
        output_shape = (int(frame_count), output_height, output_width, int(c))
        result_device = _resolve_output_device(images, output_shape, output_device)

        quality_mapping = {
            "LOW": nvvfx.effects.QualityLevel.LOW,
            "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM,
            "HIGH": nvvfx.effects.QualityLevel.HIGH,
            "ULTRA": nvvfx.effects.QualityLevel.ULTRA,
        }
        selected_quality = quality_mapping.get(quality, nvvfx.effects.QualityLevel.HIGH)
        cuda_device = images.device if images.device.type == "cuda" else torch.device("cuda")

        with torch.inference_mode(), nvvfx.VideoSuperRes(selected_quality) as sr:
            sr.output_width = output_width
            sr.output_height = output_height
            sr.load()

            # The returned IMAGE must be one contiguous tensor, but there is no reason
            # to pre-stage a multi-frame float32 CUDA batch as well. Process exactly one
            # frame at a time so temporary memory is O(one frame) instead of O(batch).
            out_tensor = torch.empty(
                output_shape,
                device=result_device,
                dtype=images.dtype,
            )
            for index in range(int(frame_count)):
                input_frame = (
                    images[index]
                    .movedim(-1, 0)
                    .to(device=cuda_device, dtype=torch.float32)
                    .contiguous()
                )
                dlpack_out = sr.run(input_frame).image
                output_frame = torch.from_dlpack(dlpack_out).movedim(0, -1)
                out_tensor[index].copy_(
                    output_frame.to(device=result_device, dtype=images.dtype)
                )
                del input_frame, output_frame, dlpack_out

        return io.NodeOutput(out_tensor)


class NVVFXVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            RTXVideoSuperResolution,
        ]


async def comfy_entrypoint() -> NVVFXVideoExtension:
    return NVVFXVideoExtension()

# hack so registry picks up the node name
if False:
    NODE_CLASS_MAPPINGS = {"RTXVideoSuperResolution": RTXVideoSuperResolution}
