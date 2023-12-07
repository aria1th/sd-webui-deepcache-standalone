# sd-webui-deepcache-standalone
New implementation - should be tested though

512x704 test, with 40% disable for initial steps
```
Steps: 23, Sampler: DPM++ SDE Karras, CFG scale: 8, Seed: 3335110679, Size: 512x704, Model hash: 8c838299ab, VAE hash: 79e225b92f, VAE: blessed2.vae.pt, Denoising strength: 0.5, Hypertile U-Net: True, Hypertile U-Net max depth: 2, Hypertile U-Net max tile size: 64, Hypertile U-Net swap size: 12, Hypertile VAE: True, Hypertile VAE swap size: 2, Hires upscale: 2, Hires upscaler: R-ESRGAN 4x+ Anime6B, Version: v1.7.0-RC-16-geb2b1679
```
![image](https://github.com/AUTOMATIC1111/stable-diffusion-webui/assets/35677394/66868212-87b5-4734-989d-c7c4882069ee)
**Enabled, Reusing cache for HR steps**
![grid-0671-3335110679-1girl](https://github.com/AUTOMATIC1111/stable-diffusion-webui/assets/35677394/295bf626-e89c-45c8-8f3c-ab03767b5dad)
**5.68it/s**

**Enabled:**
![grid-0660-3335110679-1girl](https://github.com/AUTOMATIC1111/stable-diffusion-webui/assets/35677394/2dd00da6-5588-4275-bbb1-8b825d661dce)
**4.66it/s**

**Vanilla with Hypertile:**
![grid-0661-3335110679-1girl](https://github.com/AUTOMATIC1111/stable-diffusion-webui/assets/35677394/31b91e9c-6977-4043-ad29-3b82fffb75bd)
**2.21it/s**

**Vanilla without Hypertile**

![grid-0664-3335110679-1girl](https://github.com/AUTOMATIC1111/stable-diffusion-webui/assets/35677394/bfa59048-adc4-4549-ae92-287a2540667a)
**1.21it/s**
**Vanilla with DeepCache Only**
![grid-0665-3335110679-1girl](https://github.com/AUTOMATIC1111/stable-diffusion-webui/assets/35677394/9921e411-1b0f-46da-a113-7b8b9495821d)
**2.83it/s**



SD XL : Being tested.


**Literally achieves 4.5x boost!**
