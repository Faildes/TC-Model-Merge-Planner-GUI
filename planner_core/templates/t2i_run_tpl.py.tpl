# t2i
RUN_T2I = $run_t2i
T2I_SETTINGS_JSON = r"""$t2i_settings_json"""

def bpro(prompt):
    k = prompt.split(",")
    thu = []
    for g in k:
        f = g.count(" ")
        thu.append([g, f+1])
    off = 0
    nl = []
    t = 0
    for x in thu:
        if "BREAK" in x[0]:
            tok = t+off
            add = tok % 75
            nl += [" "]*add
            off += add
            continue
        t += x[1]
        nl.append(x[0])
    return ",".join(nl)
    
if RUN_T2I:
    import os
    import json
    import random
    from PIL import Image
    from IPython.display import display
    from sd_embed.embedding_funcs import $encoder
    
    base = "$base".strip()
    pipe = globals().get("init_pipe")
    encoder = $encoder
    default_settings = {
        "prompts": [{"name": "default", "prompt": "masterpiece, best quality, scenery", "negative": "lowres, bad anatomy, watermark"}],
        "seed": -1,
        "steps": 20,
        "width": 768,
        "height": 1152,
        "cfg": 4.5,
        "num_gen": 1,
    }
    try:
        loaded_settings = json.loads(T2I_SETTINGS_JSON) if T2I_SETTINGS_JSON.strip() else {}
        if isinstance(loaded_settings, dict):
            default_settings.update(loaded_settings)
    except Exception as e:
        print(f"[t2i] Failed to parse settings, using defaults: {e}")
    settings = default_settings
    prompt_items = settings.get("prompts") or default_settings["prompts"]
    if isinstance(prompt_items, dict):
        prompt_items = [prompt_items]
    prompt_items = [p for p in prompt_items if isinstance(p, dict)] or default_settings["prompts"]
    w = int(settings.get("width") or settings.get("w") or 768)
    h = int(settings.get("height") or settings.get("h") or 1152)
    steps = int(settings.get("steps") or 20)
    guidance = float(settings.get("cfg", settings.get("guidance", 4.5)) or 4.5)
    num_gen = max(1, int(settings.get("num_gen") or 1))
    base_seed = int(settings.get("seed", -1) if str(settings.get("seed", "")).strip() else -1)
    _t2i_root = globals().get("WORKING_DIR", r"$workpath")
    idir = os.path.join(_t2i_root, "t2i_images")
    os.makedirs(idir, exist_ok=True)

    if pipe is None:
        print("t2i helper idle. Configure or build a final model first.")
    else:
        manifest = []
        for prompt_idx, item in enumerate(prompt_items):
            prompt = str(item.get("prompt") or "").strip() or default_settings["prompts"][0]["prompt"]
            neg = str(item.get("negative") or item.get("neg") or settings.get("negative") or default_settings["prompts"][0]["negative"])
            if base != "SDXL":
                embeds, negative_embeds=encoder(pipe,prompt=bpro(prompt),neg_prompt=neg)
            else:
                embeds, negative_embeds, pooled, neg_pooled=encoder(pipe,prompt=bpro(prompt),neg_prompt=neg)
            name = str(item.get("name") or f"prompt_{prompt_idx+1}").strip() or f"prompt_{prompt_idx+1}"
            safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)[:64] or f"prompt_{prompt_idx+1}"
            for i in range(num_gen):
                seed = base_seed if base_seed >= 0 else random.randrange(4294967294)
                if base_seed >= 0 and (prompt_idx or i):
                    seed = (base_seed + prompt_idx * max(1, num_gen) + i) % 4294967294
                generator = torch.Generator(device="cpu").manual_seed(seed)
                if base != "SDXL":
                    image = pipe(
                        prompt=None, 
                        prompt_embeds=embeds, 
                        negative_prompt_embeds=negative_embeds, 
                        height=h, width=w, 
                        num_inference_steps=steps, 
                        guidance_scale=guidance, 
                        generator=generator).images[0]
                else:
                    image = pipe(
                        prompt=None, 
                        prompt_embeds=embeds, 
                        pooled_prompt_embeds=pooled, 
                        negative_prompt_embeds=negative_embeds, 
                        negative_pooled_prompt_embeds=neg_pooled, 
                        height=h, width=w, 
                        num_inference_steps=steps, 
                        guidance_scale=guidance, 
                        generator=generator).images[0]
                out_path = os.path.join(idir, f"{safe_name}_{i:03d}_{seed}.png")
                image.save(out_path)
                manifest.append({"path": out_path, "prompt_name": name, "prompt": prompt, "negative": neg, "seed": seed, "width": w, "height": h, "steps": steps, "cfg": guidance})
                display(image.resize((max(1, w // 2), max(1, h // 2)), Image.Resampling.LANCZOS))
                print(f"Saved: {out_path}")
        manifest_path = os.path.join(idir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[t2i] Manifest: {manifest_path}")
