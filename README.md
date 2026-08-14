# TC Model Merge Planner GUI

A Tkinter-based planner UI for building, validating, exporting, and running model merge workflows.
It is designed to sit on top of **Chattiori-Model-Merger** and gives you a visual editor for checkpoint merges, LoRA bake steps, download/local model registration, notebook generation, execution monitoring, and optional Hugging Face upload.

---

## Features

- **Plan Creator UI** for editing merge workflows line by line.
- **Structured plan editing** with support for:
  - Checkpoint Merge
  - LoRA Bake
  - LoRA Merge

  The below functions is integrated to the system automatically
  - Download Model
  - Local Model
  - Remove Model
- **Base model switching** for multiple families:
  - Stable Diffusion 1.5
  - Stable Diffusion XL
  - Flux
  - Z-Image
  - Anima (Anima 2.9B as well)
  - Krea2
- **Ratio modes**:
  - Single
  - Block weight
  - Elemental
- **Legacy plan text compatibility** (`CM`, `LB`, `LC`, `+`, `-`) alongside structured internal plan data.
- **Notebook export and execution** using generated Jupyter notebooks.
- **Execution console** with:
  - IDLE log view
  - raw Jupyter output
  - rendered notebook output view
  - progress tracking
  - stop button for running jobs
- **Pre-validation / issue explanation UI** to help locate missing references, duplicate aliases, unreferenced outputs, and similar plan problems.
- **Optional Hugging Face upload** for the newest generated `.safetensors` file.
- **Optional T2I run stage** after merge execution.
- **Auto-detection / assisted install** of the backend merger repository.

---

## Repository Contents

This planner is currently composed of the following main files:

- `main.py` — desktop UI, validation, console, notebook execution integration, upload helper, and general planner interaction.
- `plan.py` — plan normalization, legacy text parsing/export, command generation, notebook compilation, runtime helper code.
- `install_planner_deps.py` — dependency installer and backend repository bootstrapper.

After setup, the planner expects the Chattiori backend to exist at one of these locations:

- `tools/chattiori_model_merge`
- `tools/chattiori_model_merger`

The default installer target is:

- `tools/chattiori_model_merger`

---

## Requirements

### Python

- Python 3.10+ is recommended.

### Python packages

The included installer script installs these packages automatically:

- `requests`
- `filelock`
- `fake_useragent`
- `huggingface_hub`
- `pillow`
- `papermill`
- `jupyter`
- `nbconvert`
- `nbformat`
- `ipython`
- `ipykernel`

The generated notebook runtime may also install additional runtime packages such as `torch`, `torchvision`, `diffusers`, `torchsde`, `peft`, and `torchao` depending on your environment and run settings.

### System tools

The generated notebook install stage attempts to use system package managers to install:

- `aria2`
- `git`

Supported install paths are included for Linux, macOS, and Windows package managers where available.

### GUI note

This project uses **Tkinter**. On some Linux distributions, you may need to install the Tk runtime separately if it is not bundled with your Python installation.

---

## Initial Setup

### 1. Clone this repository

```bash
git clone https://github.com/Faildes/TC-Model-Merge-Planner-GUI
cd TC-Model-Merge-Planner-GUI
```

### 2. Install planner dependencies and backend merger

```bash
python install_planner_deps.py
```

This script will:

1. install the Python packages needed by the planner,
2. clone or update **Chattiori-Model-Merger**,
3. install that repository's `requirements.txt` if present.

### Optional installer flags

```bash
python install_planner_deps.py --skip-pip
python install_planner_deps.py --skip-repo
python install_planner_deps.py --check-update
python install_planner_deps.py --force-update
```

### 3. Launch the planner

```bash
python main.py
```

---

## First Launch Checklist

When the UI opens, start with the left panel:

1. **Base Model**  
   Choose the model family you are planning for.

2. **Plan Text Path**  
   Set the `.txt` file used for loading/saving the plan.

3. **Workspace Path**  
   Set the working directory used by generated notebooks.

4. **Model Dir (Opt.)**  
   Optional custom checkpoint directory.

5. **VAE Dir (Opt.)**  
   Optional custom VAE directory.

6. **Notebook Title**  
   Sets the output notebook base name.

7. **HuggingFace Token / CivitAI API**  
   Fill these only if you need authenticated downloads or uploads.

8. **User/Repo ID**  
   Set this when you want to upload the final result to Hugging Face.

9. **Notebook Run Options**  
   - Ignore Install Deps  
   - Upload After Merge  
   - Run T2I

The planner stores session-style settings in `config.tccm`.

---

## How to Use

### Basic workflow

A typical workflow is:

1. Create or load a plan text file.
2. Add the necessary source models.
3. Build merge and/or LoRA bake steps in **Plan Creator**.
4. Save the plan text.
5. Run pre-validation and fix any unresolved aliases or missing references.
6. Export a notebook or run it directly.
7. Review progress in the console.
8. Optionally upload the newest result to Hugging Face.

### Main action buttons

The planner includes these main actions:

- **Run Merge Notebook** — generate and execute a notebook from the current plan.
- **Save Plan Text** — write the current in-memory plan back to the plan file.
- **Export as notebook** — create a Jupyter notebook without executing it.
- **Export as txt** — export the compiled text plan.
- **Show Console** — open the execution console.
- **Upload Latest Model** — upload the newest `.safetensors` from the model directory.

---

## Plan Creator Guide

### Line types

Each plan line has a **Model Merge Type**. The current implementation supports five line types.

#### 1. Checkpoint Merge
Create a merge step using one of the available merge modes.

Fields include:
- Merge Mode
- Model 0
- Model 1
- Model 2 (only when required)
- Alpha
- Beta (only when required)
- Output Name
- Additional Signatures

#### 2. LoRA Bake
Bake one or more LoRAs / LyCORIS models into a checkpoint.

Fields include:
- Checkpoint
- one or more LoRA blocks
- per-LoRA ratio
- Output Name
- Additional Signatures

#### 3. LoRA Merge
Merge two or more LoRAs / LyCORIS models into a LoRA / LyCORIS.

Fields include:
- two or more LoRA blocks
- per-LoRA ratio
- Output Name
- Additional Signatures

---

## Ratio Guide

### Single
A single scalar value such as:

```text
0.35
```

### Block weight
One value per block. This is useful when you want finer layer-level control over transfer strength.

Example shape:

```text
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```

### Elemental
Free-form elemental text such as layer / element / strength style expressions.
The planner also includes popup assistance for elemental editing and resolves candidate JSON files per base model, for example:

- `elemental_candidates_sdxl.json`
- `elemental_candidates_anima.json`
- `elemental_candidates_zimage.json`
- `elemental_candidates_sd15.json`
- `elemental_candidates_flux.json`
- `elemental_candidates_krea2.json`

Example style:

```text
IN04:attn:0.12,OUT03:res:0.08
```

---

## Additional Signatures

The planner passes extra CLI-style options through the **Additional Signatures** field.
Recognized `@` style tokens include:

- `@c` / `@cosine`
- `@f` / `@fine`
- `@s` / `@seed`
- `@m` / `@mode`
- `@p` / `@precision`
- `@rank`
- `@arch`

These are normalized by the plan compiler and appended to the backend commands.

---

## Legacy Plan Text Format

The planner can load and export a legacy text-style plan format.
This is useful if you want to version-control plans in plain text.

### Supported legacy records

#### Download Model
```text
+ModelAlias, https://example.com/model
```

LoRA / LyCORIS style:

```text
+MyLoRA, https://example.com/lora, %LR
```

#### Local Model
```text
LC, /path/to/model.safetensors, Checkpoint
```

#### Remove Model
```text
-OldAlias
```

#### Checkpoint Merge
Example style:

```text
CM Base + Donor 0.25 Result @mode WS
```

Difference / triple-input / beta-using lines are also supported internally and exported according to the active merge mode.

#### LoRA Bake
```text
LB BaseModel StyleLoRA:0.8,PoseLoRA:0.35 FinalModel
```

---

## Example Workflow

Below is a minimal example plan:

```text
+BaseModel, https://example.com/base-model
+PoseDonor, https://example.com/pose-model
+HandsLoRA, https://example.com/hands-lora, %LR
CM BaseModel + PoseDonor 0.20 PoseMixed
LB PoseMixed HandsLoRA:0.75 FinalOutput
```

What this does:

1. registers a base checkpoint,
2. registers a donor checkpoint,
3. registers a LoRA,
4. merges the base and donor into `PoseMixed`,
5. bakes the LoRA into `PoseMixed` to create `FinalOutput`.

---

## Notebook Execution Flow

When you use **Run Merge Notebook**, the planner will:

1. save or materialize the current plan,
2. compile the plan into notebook cells,
3. prepare install/runtime helper cells,
4. execute merge and bake commands,
5. stream progress back into the console,
6. optionally upload the final model,
7. optionally run T2I validation.

The generated notebook runtime also includes support for:

- VAE download and registration,
- progress-style command streaming,
- runtime path resolution for executables,
- model registry tracking,
- cleanup / cache flush helpers.

---

## Execution Console

The **Planner Runner** console is designed for long-running jobs.
It provides:

- current state,
- current step,
- progress text / percent,
- IDLE-style log output,
- raw Jupyter output,
- rendered notebook output,
- a stop button for the current process.

This is especially useful for download-heavy or multi-step notebook workflows.

---

## Validation and Troubleshooting

The planner performs plan validation and highlights problem rows in the plan list.
Examples of detected issues include:

- missing checkpoint references,
- missing LoRA references,
- duplicate produced aliases,
- outputs that are never used later,
- disabled lines that break later references.

The pre-validation UI can also show:

- cause details,
- available aliases before a failing line,
- suggested fixes,
- current line payload details.

### Common fixes

#### A referenced model is not available
Make sure the model alias is introduced by an earlier active line and not removed later.

#### An output is unreferenced
Either connect it into a later step or remove the unused line.

#### A line is disabled
Disabled lines are excluded from export/runtime. Re-enable them if later steps depend on them.

#### Upload fails
Check that:
- `User/Repo ID` is set,
- `HuggingFace Token` is valid,
- your model directory contains at least one `.safetensors` file.

#### Backend repo is missing
Run:

```bash
python install_planner_deps.py
```

again and verify that `tools/chattiori_model_merger` exists.

---

## Recommended GitHub Structure

A practical repository layout is:

```text
.
├─ main.py
├─ plan.py
├─ install_planner_deps.py
├─ README.md
├─ tools/
│  └─ chattiori_model_merger/
├─ elemental_candidates_sdxl.json
├─ elemental_candidates_flux.json
└─ ...
```

You do not need all elemental candidate JSON files immediately, but the planner is already prepared to use them.

---

## Credits

### Core merge backend
This planner is designed to work with:

- **Faildes / Chattiori-Model-Merger**

### Libraries / ecosystem
This project also relies on or integrates with:

- Python / Tkinter
- Jupyter / IPython / nbformat / nbconvert / papermill
- Pillow
- huggingface_hub
- requests
- filelock
- fake_useragent
- PyTorch / diffusers ecosystem during notebook runtime

### Planner implementation
- Planner UI / structured plan workflow / notebook wrapper: this repository
