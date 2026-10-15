"""Elemental controls and callbacks."""
from __future__ import annotations

from .constants import DEFAULT_ELEMENTAL_ELEMENTS
from .constants import ELEMENTAL_CANDIDATE_JSON_FILES
from .constants import SDXL_BLOCKS
from .paths import RESOURCE_ROOT
from pathlib import Path
from tkinter import Text
from typing import Any
from typing import Dict
from typing import List
import json
import re


class ElementalMixin:
    """Elemental behavior for the Planner."""

    def _elemental_candidate_json_filename(self) -> str:
        base_model = self.base_model_var.get() if hasattr(self, "base_model_var") else self.config.get("base_model", "SDXL")
        return ELEMENTAL_CANDIDATE_JSON_FILES.get(base_model, "elemental_candidates_sdxl.json")


    def _elemental_candidate_json_path(self) -> Path:
        return Path((str(RESOURCE_ROOT / "main.py"))).resolve().parent / self._elemental_candidate_json_filename()


    def _load_elemental_catalog(self) -> Dict[str, Any]:
        path = self._elemental_candidate_json_path()
        cache_sig = None
        try:
            if path.exists():
                stat = path.stat()
                cache_sig = (str(path), int(stat.st_mtime_ns), int(stat.st_size))
        except Exception:
            cache_sig = None
        if cache_sig is not None and getattr(self, "_elemental_catalog_cache_sig", None) == cache_sig:
            cached = getattr(self, "_elemental_catalog_cache", None)
            if isinstance(cached, dict):
                return cached

        layers: List[Dict[str, str]] = []
        elements: List[Dict[str, str]] = []
        layer_elements: Dict[str, List[Dict[str, str]]] = {}
        layer_aliases: Dict[str, str] = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                layers = self._normalize_named_effect_items(data.get("layers") or data.get("blocks") or data.get("hierarchy") or [])
                elements = self._normalize_named_effect_items(data.get("elements") or data.get("elementals") or data.get("parts") or [])
                raw_map = data.get("layer_elements") or data.get("elements_by_layer") or {}
                element_lookup = self._element_lookup_from_items(elements)
                if isinstance(raw_map, dict):
                    for layer_name, raw_items in raw_map.items():
                        norm = self._normalize_elemental_layer_name(layer_name)
                        if not norm:
                            continue
                        layer_elements[norm] = self._resolve_layer_element_items(raw_items, element_lookup)
                raw_aliases = data.get("layer_aliases") or {}
                if isinstance(raw_aliases, dict):
                    for alias_name, canonical in raw_aliases.items():
                        alias_norm = str(alias_name or "").strip().upper()
                        canonical_norm = str(canonical or "").strip().upper()
                        if alias_norm and canonical_norm:
                            layer_aliases[alias_norm] = canonical_norm
            except Exception:
                layers = []
                elements = []
                layer_elements = {}
                layer_aliases = {}
        if not layers:
            layers = [{"name": name, "effect": self._block_effect_hint(name)} for name in (self._current_block_names() or list(SDXL_BLOCKS))]
        if not layer_elements:
            fallback_items = [{"name": name, "effect": effect} for name, effect in DEFAULT_ELEMENTAL_ELEMENTS.items()]
            for layer in layers:
                layer_name = self._normalize_elemental_layer_name(layer.get("name", ""))
                if layer_name:
                    layer_elements[layer_name] = [dict(item) for item in fallback_items]
        if not elements:
            if layer_elements:
                dedup: Dict[str, Dict[str, str]] = {}
                for items in layer_elements.values():
                    for item in items:
                        key = str(item.get("name", "")).strip().lower()
                        if key and key not in dedup:
                            dedup[key] = {
                                "name": str(item.get("name", "")).strip(),
                                "effect": str(item.get("effect", "")).strip(),
                            }
                elements = list(dedup.values())
            else:
                elements = [{"name": name, "effect": effect} for name, effect in DEFAULT_ELEMENTAL_ELEMENTS.items()]

        catalog: Dict[str, Any] = {
            "layers": layers,
            "elements": elements,
            "layer_elements": layer_elements,
            "layer_aliases": layer_aliases,
            "path": [{"name": self._elemental_candidate_json_filename(), "effect": "base-model-specific candidate file loaded for elemental popup suggestions"}],
        }
        self._elemental_catalog_cache = catalog
        self._elemental_catalog_cache_sig = cache_sig
        return catalog


    def _normalize_elemental_layer_name(self, name: str) -> str:
        raw = str(name or "").strip()
        if not raw:
            return ""
        upper = raw.upper()
        alias_map = {"MID": "M00", "MID00": "M00", "MIDDLE": "M00"}
        cached = getattr(self, "_elemental_catalog_cache", None)
        if isinstance(cached, dict):
            for alias_name, canonical in (cached.get("layer_aliases") or {}).items():
                alias_map[str(alias_name).strip().upper()] = str(canonical).strip().upper()
        upper = alias_map.get(upper, upper)
        known = {str(x).strip().upper() for x in (self._current_block_names() or [])}
        known.update({"BASE", "VAE", "M00"})
        if upper in known:
            return upper
        return upper


    def _elemental_valid_elements_for_layer(self, layer_name: str) -> List[Dict[str, str]]:
        catalog = self._load_elemental_catalog()
        norm = self._normalize_elemental_layer_name(layer_name)
        layer_map = catalog.get("layer_elements") or {}
        if isinstance(layer_map, dict) and norm in layer_map:
            return list(layer_map.get(norm) or [])
        return list(catalog.get("elements") or [])


    def _elemental_alias_lookup(self, layer_name: str | None = None) -> Dict[str, List[Dict[str, str]]]:
        items = self._elemental_valid_elements_for_layer(layer_name) if layer_name else list((self._load_elemental_catalog().get("elements") or []))
        lookup: Dict[str, List[Dict[str, str]]] = {}
        seen = set()
        for item in items:
            canonical = str(item.get("name", "")).strip()
            effect = str(item.get("effect", "")).strip()
            aliases = item.get("aliases") if isinstance(item, dict) else None
            if not canonical or not isinstance(aliases, list):
                continue
            for alias in aliases:
                alias_str = str(alias).strip()
                if not alias_str or alias_str.lower() == canonical.lower():
                    continue
                key = (alias_str.lower(), canonical.lower(), effect)
                if key in seen:
                    continue
                seen.add(key)
                lookup.setdefault(alias_str.lower(), []).append({
                    "alias": alias_str,
                    "canonical": canonical,
                    "effect": effect,
                })
        return lookup


    def _elemental_layer_order(self) -> List[str]:
        catalog = self._load_elemental_catalog()
        out: List[str] = []
        for item in catalog.get("layers", []):
            name = self._normalize_elemental_layer_name(item.get("name", ""))
            if name and name not in out:
                out.append(name)
        if not out:
            out = [self._normalize_elemental_layer_name(x) for x in self._current_block_names()]
        return out


    def _expand_elemental_layer_token(self, token: str) -> List[str]:
        token = str(token or "").strip()
        if not token:
            return []
        order = self._elemental_layer_order()
        if "-" in token and not token.startswith("-") and not token.endswith("-"):
            left, right = token.split("-", 1)
            left_n = self._normalize_elemental_layer_name(left)
            right_n = self._normalize_elemental_layer_name(right)
            if left_n in order and right_n in order:
                li = order.index(left_n)
                ri = order.index(right_n)
                if li <= ri:
                    return order[li:ri + 1]
                return order[ri:li + 1]
        norm = self._normalize_elemental_layer_name(token)
        return [norm] if norm in order else []


    def _resolve_elemental_layers_expr(self, expr: str) -> List[str]:
        expr = str(expr or "")
        out: List[str] = []
        seen = set()
        for token in expr.split():
            for layer_name in self._expand_elemental_layer_token(token):
                key = layer_name.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(layer_name)
        return out


    def _elemental_valid_elements_for_layers(self, layer_names: List[str]) -> List[Dict[str, str]]:
        catalog = self._load_elemental_catalog()
        layer_map = catalog.get("layer_elements") or {}
        out: List[Dict[str, str]] = []
        seen = set()
        if layer_names:
            for layer_name in layer_names:
                items = layer_map.get(self._normalize_elemental_layer_name(layer_name), []) if isinstance(layer_map, dict) else []
                for item in items:
                    name = str(item.get("name", "")).strip()
                    if not name:
                        continue
                    key = name.lower()
                    if key not in seen:
                        seen.add(key)
                        entry = {"name": name, "effect": str(item.get("effect", "")).strip()}
                        aliases = item.get("aliases") if isinstance(item, dict) else None
                        if isinstance(aliases, list) and aliases:
                            entry["aliases"] = [str(x).strip() for x in aliases if str(x).strip()]
                        out.append(entry)
        else:
            for item in catalog.get("elements", []):
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                key = name.lower()
                if key not in seen:
                    seen.add(key)
                    entry = {"name": name, "effect": str(item.get("effect", "")).strip()}
                    aliases = item.get("aliases") if isinstance(item, dict) else None
                    if isinstance(aliases, list) and aliases:
                        entry["aliases"] = [str(x).strip() for x in aliases if str(x).strip()]
                    out.append(entry)
        return out


    def _elemental_separator_for_widget(self, widget: Text | None) -> str:
        sep = str(getattr(widget, "_elemental_separator", ",") or ",") if widget is not None else ","
        return sep if sep in {",", ";"} else ","


    def _elemental_context(self, widget: Text) -> Dict[str, Any]:
        sep = self._elemental_separator_for_widget(widget)
        before = widget.get("1.0", "insert")
        abs_end = len(before)
        seg_start = max(before.rfind(sep), before.rfind("\n"))
        segment = before[seg_start + 1:]
        base_abs = seg_start + 1
        first_colon = segment.find(":")
        second_colon = segment.find(":", first_colon + 1) if first_colon >= 0 else -1

        def token_info(text_part: str, offset: int):
            if not text_part or text_part.endswith((" ", "\t")):
                pos = f"1.0+{abs_end}c"
                return "", pos, pos
            m = re.search(r"(\S+)$", text_part)
            if not m:
                pos = f"1.0+{abs_end}c"
                return "", pos, pos
            start_abs = base_abs + offset + m.start(1)
            return m.group(1), f"1.0+{start_abs}c", f"1.0+{abs_end}c"

        if first_colon < 0:
            probe, start, end = token_info(segment, 0)
            return {
                "stage": "layer",
                "probe": probe,
                "start": start,
                "end": end,
                "segment": segment,
                "layer_part": segment,
                "layer_expr": segment,
                "element_part": "",
                "target_layers": self._resolve_elemental_layers_expr(segment),
            }

        layer_part = segment[:first_colon]
        if second_colon < 0:
            element_part = segment[first_colon + 1:]
            probe, start, end = token_info(element_part, first_colon + 1)
            return {
                "stage": "element",
                "probe": probe,
                "start": start,
                "end": end,
                "segment": segment,
                "layer_part": layer_part,
                "layer_expr": layer_part,
                "element_part": element_part,
                "target_layers": self._resolve_elemental_layers_expr(layer_part),
            }

        return {
            "stage": "strength",
            "probe": "",
            "start": None,
            "end": None,
            "segment": segment,
            "layer_part": layer_part,
            "layer_expr": layer_part,
            "element_part": segment[first_colon + 1:second_colon],
            "target_layers": self._resolve_elemental_layers_expr(layer_part),
        }


    def _elemental_suggestion_items(self, widget: Text) -> List[Dict[str, str]]:
        catalog = self._load_elemental_catalog()
        ctx = self._elemental_context(widget)
        probe = str(ctx.get("probe") or "").strip().lower()

        if ctx.get("stage") == "layer":
            out = []
            for item in catalog.get("layers", []):
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                if probe and probe not in name.lower():
                    continue
                out.append(self._autocomplete_item(name, item.get("effect", ""), label=f"layer  {name}    — {item.get('effect', '')}"))
            return out[:40]

        if ctx.get("stage") == "element":
            out = []
            for item in self._elemental_valid_elements_for_layers(ctx.get("target_layers") or []):
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                aliases = [str(x).strip().lower() for x in (item.get("aliases") or []) if str(x).strip()]
                haystack = [name.lower()] + aliases
                if probe and not any(probe in x for x in haystack):
                    continue
                out.append(self._autocomplete_item(name, item.get("effect", ""), label=f"element {name}    — {item.get('effect', '')}"))
            return out[:40]

        return []


    def _on_elemental_text_motion(self, widget: Text, event):
        token, start, _end, _index = self._extract_elemental_hover_token(widget, event)
        if not token:
            self._hide_elemental_alias_hover()
            return None
        try:
            full_text = widget.get("1.0", "end-1c")
            seg_abs = widget.count("1.0", start, "chars")[0]
        except Exception:
            self._hide_elemental_alias_hover()
            return None
        left_text = full_text[:seg_abs]
        segment_start = max(left_text.rfind(","), left_text.rfind(";"), left_text.rfind("\n")) + 1
        segment_left = full_text[segment_start:seg_abs]
        if segment_left.count(":") != 1:
            self._hide_elemental_alias_hover()
            return None
        layer_expr = segment_left.split(":", 1)[0]
        layer_names = self._resolve_elemental_layers_expr(layer_expr)
        alias_info = self._elemental_alias_lookup(layer_names).get(token.lower()) or self._elemental_alias_lookup().get(token.lower())
        if alias_info:
            self._show_elemental_alias_hover_popup(widget, alias_info, event)
        else:
            self._hide_elemental_alias_hover()
        return None


    @staticmethod
    def _normalize_named_effect_items(raw) -> List[Dict[str, str]]:
        items: List[Dict[str, str]] = []
        if isinstance(raw, dict):
            iterable = [{"name": k, "effect": v} for k, v in raw.items()]
        elif isinstance(raw, list):
            iterable = raw
        else:
            iterable = []
        for item in iterable:
            if isinstance(item, str):
                name = item.strip()
                effect = ""
            elif isinstance(item, dict):
                name = str(item.get("name") or item.get("key") or item.get("id") or item.get("label") or "").strip()
                effect = str(item.get("effect") or item.get("what") or item.get("description") or item.get("detail") or item.get("help") or "").strip()
            else:
                continue
            if name:
                entry = {"name": name, "effect": effect}
                if isinstance(item, dict):
                    aliases = item.get("aliases")
                    if isinstance(aliases, list) and aliases:
                        entry["aliases"] = [str(x).strip() for x in aliases if str(x).strip()]
                items.append(entry)
        return items


    @staticmethod
    def _element_lookup_from_items(items: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        lookup: Dict[str, List[Dict[str, str]]] = {}
        for item in items or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            entry = {
                "name": name,
                "effect": str(item.get("effect", "")).strip(),
            }
            aliases = item.get("aliases")
            if isinstance(aliases, list) and aliases:
                entry["aliases"] = [str(x).strip() for x in aliases if str(x).strip()]
            lookup.setdefault(name, []).append(entry)
        return lookup


    def _resolve_layer_element_items(self, raw_items, element_lookup: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        resolved: List[Dict[str, str]] = []
        seen = set()
        for raw in raw_items or []:
            candidates: List[Dict[str, str]] = []
            if isinstance(raw, str):
                candidates = list(element_lookup.get(raw.strip(), []))
            elif isinstance(raw, dict):
                raw_name = str(raw.get("name", "")).strip()
                if raw_name and raw_name in element_lookup:
                    candidates = list(element_lookup.get(raw_name, []))
                elif raw_name:
                    entry = {
                        "name": raw_name,
                        "effect": str(raw.get("effect", "")).strip(),
                    }
                    aliases = raw.get("aliases")
                    if isinstance(aliases, list) and aliases:
                        entry["aliases"] = [str(x).strip() for x in aliases if str(x).strip()]
                    candidates = [entry]
            for item in candidates:
                key = (str(item.get("name", "")).strip().lower(), str(item.get("effect", "")).strip())
                if key in seen or not key[0]:
                    continue
                seen.add(key)
                resolved.append(dict(item))
        return resolved


    def _block_effect_hint_fallback(self, block_name: str) -> str:
        name = str(block_name or "").upper()
        if name == "BASE":
            return "global base behavior, broad style baseline, and shared foundation layers"
        if name.startswith("IN"):
            try:
                idx = int(name[2:])
            except Exception:
                idx = 0
            if idx <= 2:
                return "very coarse composition, silhouette, camera distance, and large structure"
            if idx <= 5:
                return "mid-scale shapes, pose flow, anatomy rhythm, and scene layout"
            return "deeper feature extraction, local forms, and transition into detailed latent features"
        if name.startswith("MID"):
            return "bottleneck/global mixing, concept fusion, and whole-image latent coordination"
        if name.startswith("OUT"):
            try:
                idx = int(name[3:])
            except Exception:
                idx = 0
            if idx <= 2:
                return "large decoded structures, major reconstruction, and broad rendering direction"
            if idx <= 5:
                return "surface shaping, material feel, lighting transitions, and regional detail"
            return "fine finishing, micro-detail, crisp edges, and final texture cleanup"
        return "model-specific feature group; effect depends on architecture and training"


    def _block_effect_hint(self, block_name: str) -> str:
        name = self._normalize_elemental_layer_name(block_name)
        path = self._elemental_candidate_json_path()
        cache_sig = None
        try:
            if path.exists():
                stat = path.stat()
                cache_sig = (str(path), int(stat.st_mtime_ns), int(stat.st_size))
        except Exception:
            cache_sig = None

        effect_map = getattr(self, "_block_effect_hint_cache", None)
        if not isinstance(effect_map, dict) or getattr(self, "_block_effect_hint_cache_sig", None) != cache_sig:
            effect_map = {}
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    layer_items = self._normalize_named_effect_items(
                        data.get("layers") or data.get("blocks") or data.get("hierarchy") or []
                    )
                    for item in layer_items:
                        layer_name = self._normalize_elemental_layer_name(item.get("name", ""))
                        effect = str(item.get("effect") or "").strip()
                        if layer_name and effect:
                            effect_map[layer_name] = effect
                except Exception:
                    effect_map = {}
            self._block_effect_hint_cache = effect_map
            self._block_effect_hint_cache_sig = cache_sig

        effect = str(effect_map.get(name, "") or "").strip() if isinstance(effect_map, dict) else ""
        if effect:
            return effect
        return self._block_effect_hint_fallback(name)


    def _current_block_names(self) -> List[str]:
        base_model = self.base_model_var.get() if hasattr(self, "base_model_var") else self.config.get("base_model", "SDXL")
        names = self.block_sets.get(base_model) or self.block_sets.get("SDXL") or list(SDXL_BLOCKS)
        return [str(name) for name in names]
