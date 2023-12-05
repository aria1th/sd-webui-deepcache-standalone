"""
Patched forward_timestep_embed function to support the following:
@source https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/ldm/modules/diffusionmodules/openaimodel.py
"""
from ldm.modules.attention import SpatialTransformer
try:
    from ldm.modules.attention import SpatialVideoTransformer
except (ImportError, ModuleNotFoundError):
    SpatialVideoTransformer = None
from ldm.modules.diffusionmodules.openaimodel import TimestepBlock, TimestepEmbedSequential, Upsample
try:
    from ldm.modules.diffusionmodules.openaimodel import VideoResBlock
except (ImportError, ModuleNotFoundError):
    VideoResBlock = None
import torch.nn.functional as F

def forward_timestep_embed(ts:TimestepEmbedSequential, x, emb, context=None, output_shape=None, time_context=None, num_video_frames=None, image_only_indicator=None):
    for layer in ts:
        if VideoResBlock and isinstance(layer, VideoResBlock):
            x = layer(x, emb, num_video_frames, image_only_indicator)
        elif isinstance(layer, TimestepBlock):
            x = layer(x, emb)
        elif SpatialVideoTransformer and isinstance(layer, SpatialVideoTransformer):
            x = layer(x, context, time_context, num_video_frames, image_only_indicator)
        elif isinstance(layer, SpatialTransformer):
            x = layer(x, context)
        elif isinstance(layer, Upsample):
            x = forward_upsample(layer, x, output_shape=output_shape)
        else:
            x = layer(x)
    return x

def forward_upsample(self:Upsample, x, output_shape=None):
    assert x.shape[1] == self.channels
    if self.dims == 3:
        shape = [x.shape[2], x.shape[3] * 2, x.shape[4] * 2]
        if output_shape is not None:
            shape[1] = output_shape[3]
            shape[2] = output_shape[4]
    else:
        shape = [x.shape[2] * 2, x.shape[3] * 2]
        if output_shape is not None:
            shape[0] = output_shape[2]
            shape[1] = output_shape[3]

    x = F.interpolate(x, size=shape, mode="nearest")
    if self.use_conv:
        x = self.conv(x)
    return x
