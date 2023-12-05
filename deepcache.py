from __future__ import annotations

from dataclasses import dataclass

import torch
from ldm.modules.diffusionmodules.openaimodel import timestep_embedding
from scripts.forward_timestep_embed_patch import forward_timestep_embed

@dataclass
class DeepCacheParams:
    cache_in_start: int = 600
    cache_in_start2: int = 400
    cache_mid_start: int = 800
    cache_out_start: int = 0
    cache_in_block: int = 6
    cache_in_block2: int = 4
    cache_out_block: int = 3
    cache_disable_step: int = 0
    full_run_step_rate: int = 1000

class DeepCacheSession:
    """
    Session for DeepCache, which holds cache data and provides functions for hooking the model.
    """
    def __init__(self) -> None:
        self.CACHE_LAST = {'ts' : 1000}
        self.stored_forward = None
        self.unet_reference = None
        self.cache_success_count = 0
        self.cache_fail_count = 0

    def report(self):
        # report cache success rate
        total = self.cache_success_count + self.cache_fail_count
        if total == 0:
            return
        print(f"DeepCache success rate: {self.cache_success_count / total * 100}% ({self.cache_success_count}/{total})")

    def deepcache_hook_model(self, unet, params:DeepCacheParams):
        """
        Hooks the given unet model to use DeepCache.
        """
        cache_in_start = params.cache_in_start
        cache_in_start2 = params.cache_in_start2
        cache_mid_start = params.cache_mid_start
        cache_out_start = params.cache_out_start
        cache_in_block = params.cache_in_block
        cache_in_block2 = params.cache_in_block2
        cache_out_block = params.cache_out_block
        cache_disable_step = params.cache_disable_step
        full_run_step_rate = params.full_run_step_rate
        if getattr(unet, '_deepcache_hooked', False):
            return  # already hooked
        CACHE_LAST = self.CACHE_LAST
        self.stored_forward = unet.forward
        def hijacked_unet_forward(x, timesteps=None, context=None, y=None, **kwargs):
            assert (y is not None) == (
                hasattr(unet, 'num_classes') and unet.num_classes is not None #v2 or xl
            ), "must specify y if and only if the model is class-conditional"
            hs = []
            t_emb = timestep_embedding(timesteps, unet.model_channels, repeat_only=False).to(unet.dtype)
            emb = unet.time_embed(t_emb)

            if hasattr(unet, 'num_classes') and unet.num_classes is not None:
                assert y.shape[0] == x.shape[0]
                emb = emb + unet.label_emb(y)

            h = x.type(unet.dtype)

            timestep_index = float(timesteps[0])

            do_full = False # to run without cache
            if timestep_index > cache_in_start and timestep_index > cache_mid_start:
                CACHE_LAST['ts'] = timestep_index
            elif timestep_index < CACHE_LAST['ts'] - full_run_step_rate:
                CACHE_LAST['ts'] = timestep_index
                do_full = True

            for id, module in enumerate(unet.input_blocks):
                cache_key = f'in{id}'
                if cache_key in CACHE_LAST and id > cache_in_block and timestep_index < cache_in_start and timestep_index > cache_disable_step and not do_full:
                    h = CACHE_LAST[cache_key]
                    self.cache_success_count += 1
                elif cache_key in CACHE_LAST and id > cache_in_block2 and timestep_index < cache_in_start2 and timestep_index > cache_disable_step and not do_full:
                    h = CACHE_LAST[cache_key]
                    self.cache_success_count += 1
                else:
                    self.cache_fail_count += 1
                    h = forward_timestep_embed(module, h, emb, context)
                    CACHE_LAST[cache_key] = h
                    #print(f"in {id} is {h.mean()}")
                hs.append(h)

            if 'mid' in CACHE_LAST and timestep_index < cache_mid_start and timestep_index > cache_disable_step and not do_full:
                h = CACHE_LAST['mid']
                self.cache_success_count += 1
            else:
                self.cache_fail_count += 1
                h = forward_timestep_embed(unet.middle_block, h, emb, context)
                CACHE_LAST['mid'] = h
                #print(f"mid is {h.mean()}")

            for id, module in enumerate(unet.output_blocks):
                hsp = hs.pop()
                cache_key = f'out{id}'
                if id < cache_out_block and timestep_index < cache_out_start and timestep_index > cache_disable_step and not do_full:
                    h = CACHE_LAST[cache_key]
                    self.cache_success_count += 1
                else:
                    self.cache_fail_count += 1
                    h = torch.cat([h, hsp], dim=1)
                    del hsp
                    if len(hs) > 0:
                        output_shape = hs[-1].shape
                    else:
                        output_shape = None
                    h = forward_timestep_embed(module, h, emb, context, output_shape=output_shape)
                    CACHE_LAST[cache_key] = h
                    #print(f"out {id} is {h.mean()}")
            h = h.type(x.dtype)

            if unet.predict_codebook_ids:
                return unet.id_predictor(h)
            else:
                return unet.out(h)
        unet.forward = hijacked_unet_forward
        unet._deepcache_hooked = True
        self.unet_reference = unet

    def detach(self):
        if self.unet_reference is None:
            return
        if not getattr(self.unet_reference, '_deepcache_hooked', False):
            return
        # detach
        self.unet_reference.forward = self.stored_forward
        self.unet_reference._deepcache_hooked = False
        self.unet_reference = None
        self.stored_forward = None
        self.CACHE_LAST = {'ts' : 1000}
        self.cache_fail_count = self.cache_success_count = 0
