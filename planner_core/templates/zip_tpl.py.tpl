# Image ZIP
RUN_T2I = $run_t2i

if RUN_T2I:
    import os
    import zipfile
    from pathlib import Path

    name = "download"
    _zip_root = globals().get("WORKING_DIR", r"$workpath")
    dst = os.path.join(_zip_root, f"{name}.zip")
    idir = os.path.join(_zip_root, "t2i_images")
    if os.path.exists(dst):
        os.remove(dst)
    paths = [str(p) for p in Path(idir).rglob("*") if p.is_file()]
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, os.path.join(name, os.path.relpath(p, idir)))
    print(f"Done! -> {dst}")
