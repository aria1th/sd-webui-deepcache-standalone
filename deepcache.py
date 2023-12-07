from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

import torch
from ldm.modules.diffusionmodules.openaimodel import timestep_embedding
from scripts.forward_timestep_embed_patch import forward_timestep_embed

@dataclass
class DeepCacheParams:
    cache_in_level: int = 0
    cache_enable_step: int = 0
    full_run_step_rate: int = 5

class DeepCacheSession:
    """
    Session for DeepCache, which holds cache data and provides functions for hooking the model.
    """
    def __init__(self) -> None:
        self.CACHE_LAST = {"timestep": {0}}
        self.stored_forward = None
        self.unet_reference = None
        self.cache_success_count = 0
        self.cache_fail_count = 0
        self.fail_reasons = defaultdict(int)

    def log_skip(self, reason:str = 'disabled_by_default'):
        self.fail_reasons[reason] += 1
        self.cache_fail_count += 1

    def report(self):
        # report cache success rate
        total = self.cache_success_count + self.cache_fail_count
        if total == 0:
            return
        print(f"DeepCache success rate: {self.cache_success_count / total * 100}% ({self.cache_success_count}/{total})")
        for fail_reasons, count in self.fail_reasons.items():
            print(f"  {fail_reasons}: {count}")

    def deepcache_hook_model(self, unet, params:DeepCacheParams):
        """
        Hooks the given unet model to use DeepCache.
        """
        caching_level = params.cache_in_level
        # caching level 0 = no caching, idx for resnet layers
        cache_enable_step = params.cache_enable_step
        full_run_step_rate = params.full_run_step_rate # '5' means run full model every 5 steps
        if getattr(unet, '_deepcache_hooked', False):
            return  # already hooked
        CACHE_LAST = self.CACHE_LAST
        self.stored_forward = unet.forward
        enumerated_timestep = -1
        valid_caching_in_level = min(caching_level, len(unet.input_blocks) - 1)
        valid_caching_out_level = min(valid_caching_in_level, len(unet.output_blocks) - 1)
        # set to max if invalid
        caching_level = valid_caching_out_level
        valid_cache_timestep_range = 50 # total 1000, 50
        def put_cache(h:torch.Tensor, timestep:int, real_timestep:float):
            """
            Registers cache
            """
            if timestep < min(CACHE_LAST.get("timestep", {0})):
                # reset cache, we are going back in time
                print(f"Resetting cache for timestep {timestep}")
                CACHE_LAST.clear()
                CACHE_LAST["timestep"] = {0}
            CACHE_LAST["timestep"].add(timestep)
            assert h is not None, f"Cannot cache None"
            CACHE_LAST["last"] = h
            CACHE_LAST["real_timestep"] = real_timestep
        def get_cache(current_timestep:int, real_timestep:float) -> Optional[torch.Tensor]:
            """
            Returns the cached tensor for the given timestep and cache key.
            """
            if current_timestep < cache_enable_step:
                self.fail_reasons['disabled'] += 1
                self.cache_fail_count += 1
                return None
            elif full_run_step_rate < 1:
                self.fail_reasons['full_run_step_rate_disabled'] += 1
                self.cache_fail_count += 1
                return None
            elif current_timestep % full_run_step_rate == 0:
                self.fail_reasons['full_run_step_rate_division'] += 1
                self.cache_fail_count += 1
                return None
            elif CACHE_LAST.get("real_timestep", 0) + valid_cache_timestep_range < real_timestep:
                self.fail_reasons['cache_outdated'] += 1
                self.cache_fail_count += 1
                return None
            if "last" in CACHE_LAST:
                self.cache_success_count += 1
                return CACHE_LAST["last"]
            self.fail_reasons['not_cached'] += 1
            self.cache_fail_count += 1
            return None
        def hijacked_unet_forward(x, timesteps=None, context=None, y=None, **kwargs):
            cache_cond = lambda : enumerated_timestep % full_run_step_rate == 0 or enumerated_timestep > cache_enable_step
            use_cache_cond = lambda : enumerated_timestep > cache_enable_step and enumerated_timestep % full_run_step_rate != 0
            nonlocal enumerated_timestep, CACHE_LAST
            assert (y is not None) == (
                hasattr(unet, 'num_classes') and unet.num_classes is not None #v2 or xl
            ), "must specify y if and only if the model is class-conditional"
            hs = []
            t_emb = timestep_embedding(timesteps, unet.model_channels, repeat_only=False).to(unet.dtype)
            emb = unet.time_embed(t_emb)

            if hasattr(unet, 'num_classes') and unet.num_classes is not None:
                assert y.shape[0] == x.shape[0]
                emb = emb + unet.label_emb(y)
            real_timestep = timesteps[0].item()
            h = x.type(unet.dtype)
            cached_h = get_cache(enumerated_timestep, real_timestep)
            for id, module in enumerate(unet.input_blocks):
                self.log_skip('run_before_cache_input_block')
                h = forward_timestep_embed(module, h, emb, context)
                hs.append(h)
                if cached_h is not None and use_cache_cond() and id == caching_level:
                    break
            if not use_cache_cond():
                self.log_skip('run_before_cache_middle_block')
                h = forward_timestep_embed(unet.middle_block, h, emb, context)
            relative_cache_level = len(unet.output_blocks) - caching_level - 1
            for idx, module in enumerate(unet.output_blocks):
                if cached_h is not None and use_cache_cond() and idx == relative_cache_level:
                    # use cache
                    h = cached_h
                elif cache_cond() and idx == relative_cache_level:
                    # put cache
                    put_cache(h, enumerated_timestep, real_timestep)
                elif cached_h is not None and use_cache_cond() and idx < relative_cache_level:
                    # skip, h is already cached
                    continue
                hsp = hs.pop()
                h = torch.cat([h, hsp], dim=1)
                del hsp
                if len(hs) > 0:
                    output_shape = hs[-1].shape
                else:
                    output_shape = None
                h = forward_timestep_embed(module, h, emb, context, output_shape=output_shape)
            h = h.type(x.dtype)
            enumerated_timestep += 1
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
        self.CACHE_LAST.clear()
        self.cache_fail_count = self.cache_success_count = 0#
