# Pipe Config (short)
RUN_T2I = $run_t2i

if RUN_T2I:
    import os
    import torch
    import diffusers
    $pipelines
    base = "$base".strip()
    pipeline = $pipeline

    checkpoint = "$final".strip()
    ext = "safetensors"
    model_type = "fp16"
    scheduler = "euler_a"

    SCHEDULERS = {
        "unipc": [diffusers.schedulers.UniPCMultistepScheduler, {}, "UniPC"],
        "euler_a": [diffusers.schedulers.EulerAncestralDiscreteScheduler, {}, "Euler a"],
        "euler": [diffusers.schedulers.EulerDiscreteScheduler, {}, "Euler"],
        "ddim": [diffusers.schedulers.DDIMScheduler, {}, "DDIM"],
        "ddpm": [diffusers.schedulers.DDPMScheduler, {}, "DDPM"],
        "deis": [diffusers.schedulers.DEISMultistepScheduler, {}, "DEIS"],
        "dpm++_2m": [diffusers.schedulers.DPMSolverMultistepScheduler, {}, "DPM++ 2M"],
        "dpm++_2m_karras": [diffusers.schedulers.DPMSolverMultistepScheduler, {"use_karras_sigmas": True}, "DPM++ 2M Karras"],
    }
    mt = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}

    cpath = os.path.join(r"$model_dir", f"{checkpoint}.{ext}")
    dtype = mt[model_type]
    scheduler_cls, scheduler_kwargs, scheduler_name = SCHEDULERS[scheduler]

    if checkpoint and os.path.exists(cpath):
        base_pipe = pipeline.from_single_file(cpath, torch_dtype=dtype, use_safetensors=True, variant="fp16")
        scd = scheduler_cls.from_config(base_pipe.scheduler.config, **scheduler_kwargs)
        pipe = pipeline.from_single_file(cpath, torch_dtype=dtype, scheduler=scd, use_safetensors=True, variant="fp16")
        if base == "Anima":
            pipe.scheduler.set_sampling_config(
                sampler="euler_a_rf",      # flowmatch_euler | euler | euler_a_rf | euler_ancestral_rf
                sigma_schedule="normal",     # uniform | beta | simple | normal
            )
        pipe.safety_checker = None
        pipe = pipe.to("cuda:0" if torch.cuda.is_available() else "cpu")
        init_pipe = pipe
        scd_name = scheduler_name
        print(f"Loaded pipeline for {checkpoint}")
    else:
        init_pipe = None
        init_refiner = None
        scd_name = "N/A"
        print("No final checkpoint found for t2i.")
