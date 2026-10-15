"""Input requirements and incompatible options shared by the compiler and GUI."""


def required_model_slots(entry):
    options = entry.get("cli_options") or {}
    mode = str(entry.get("merge_mode") or "WS").upper()
    turbo = bool(options.get("turbo") or options.get("deturbo"))
    slots = ["model0"]
    if turbo or options.get("clipxor") or mode not in {"NOIN", "RM", "COMP"}:
        slots.append("model1")
    if turbo or mode in {"SAD", "AD", "TRS", "ST", "TD", "SIM", "MD", "HUB"}:
        slots.append("model2")
    return slots


def turbo_problem(entry):
    options = entry.get("cli_options") or {}
    if not (options.get("turbo") or options.get("deturbo")):
        return None
    if options.get("turbo") and options.get("deturbo"):
        return "Turbo and DeTurbo cannot be enabled together"
    mode = str(entry.get("merge_mode") or "WS").upper()
    if mode in {"RM", "COMP", "SWAP", "CLIPXOR", "XDARE"}:
        return f"Turbo is incompatible with mode {mode}"
    conflicts = [key for key in ("cosine0", "cosine1", "cosine2", "clipxor",
                                "use_dif_10", "use_dif_20", "use_dif_21", "rebasin")
                 if options.get(key) not in (None, "", False)]
    if conflicts:
        return "Turbo cannot be combined with " + ", ".join(conflicts)
    return None
