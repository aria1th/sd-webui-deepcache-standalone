from modules import scripts, script_callbacks, shared
from deepcache import DeepCacheSession, DeepCacheParams
from scripts.deepcache_xyz import add_axis_options

class ScriptDeepCache(scripts.Script):

    name = "DeepCache"
    session: DeepCacheSession = None

    def title(self):
        return self.name

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def get_deepcache_params(self) -> DeepCacheParams:
        return DeepCacheParams(
            cache_in_start=shared.opts.deepcache_cache_in_start,
            cache_in_start2=shared.opts.deepcache_cache_in_start2,
            cache_mid_start=shared.opts.deepcache_cache_mid_start,
            cache_out_start=shared.opts.deepcache_cache_out_start,
            cache_in_block=shared.opts.deepcache_cache_in_block,
            cache_in_block2=shared.opts.deepcache_cache_in_block2,
            cache_out_block=shared.opts.deepcache_cache_out_block,
            cache_disable_step=shared.opts.deepcache_cache_disable_step,
            full_run_step_rate=shared.opts.deepcache_full_run_step_rate,
        )

    def process(self, p, *args):
        self.detach_deepcache()
        if shared.opts.deepcache_enable:
            self.configure_deepcache(self.get_deepcache_params())

    def before_hr(self, p, *args):
        self.detach_deepcache()
        if shared.opts.deepcache_enable:
            self.configure_deepcache(self.get_deepcache_params())

    def postprocess(self, p, processed, *args):
        self.detach_deepcache()

    def configure_deepcache(self, params:DeepCacheParams):
        if self.session is None:
            self.session = DeepCacheSession()
        self.session.deepcache_hook_model(
            shared.sd_model.model.diffusion_model, #unet_model
            params
        )

    def detach_deepcache(self):
        if self.session is None:
            return
        self.session.report()
        self.session.detach()
        self.session = None

def on_ui_settings():
    import gradio as gr
    options = {
        "deepcache_explanation": shared.OptionHTML("""
    <a href='https://github.com/horseee/DeepCache'>DeepCache</a> optimizes by caching the results of mid-blocks, which is known for high level features, and reusing them in the next forward pass.
    """),

        "deepcache_enable": shared.OptionInfo(False, "Enable DeepCache").info("noticeable change in details of the generated picture"),
        "deepcache_cache_in_start": shared.OptionInfo(600, "TimeStep - Cache In Start", gr.Slider, {"minimum": 0, "maximum": 1000, "step": 1}).info("Timestep to start caching in in-blocks"),
        "deepcache_cache_in_start2": shared.OptionInfo(400, "TimeStep - Cache In Start 2", gr.Slider, {"minimum": 0, "maximum": 1000, "step": 1}).info("Timestep to start caching in in-blocks 2"),
        "deepcache_cache_mid_start": shared.OptionInfo(800, "TimeStep - Cache Mid Start", gr.Slider, {"minimum": 0, "maximum": 1000, "step": 1}).info("Timestep to start caching in mid-blocks"),
        "deepcache_cache_out_start": shared.OptionInfo(400, "TimeStep - Cache Out Start", gr.Slider, {"minimum": 0, "maximum": 1000, "step": 1}).info("Timestep to start caching in out-blocks"),
        "deepcache_cache_in_block": shared.OptionInfo(6, "Cache In Block Index", gr.Slider, {"minimum": 0, "maximum": 8, "step": 1}).info("In-blocks index"),
        "deepcache_cache_in_block2": shared.OptionInfo(4, "Cache In Block 2 Index", gr.Slider, {"minimum": 0, "maximum": 8, "step": 1}).info("In-blocks index 2"),
        "deepcache_cache_out_block": shared.OptionInfo(3, "Cache Out Block Index", gr.Slider, {"minimum": 0, "maximum": 8, "step": 1}).info("Out-blocks index"),
        "deepcache_cache_disable_step": shared.OptionInfo(0, "TimeStep - Do not use cache after", gr.Slider, {"minimum": 0, "maximum": 1000, "step": 1}).info("Timestep to stop using cache"),
        "deepcache_full_run_step_rate": shared.OptionInfo(1000, "TimeStep - Log cache until", gr.Slider, {"minimum": 0, "maximum": 1000, "step": 1}).info("Timestep to start to use cache"),
    }
    for name, opt in options.items():
        opt.section = ('deepcache', "DeepCache")
        shared.opts.add_option(name, opt)

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_before_ui(add_axis_options)
