"""Local LLM pair generation (`transformers` and `exllamav3`) for H3 control vectors."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch

import control_vector_nodes as cv_nodes
import control_vectors as cv

from .helpers import REPO_ROOT, _display_path

def _extract_system_prompt_text() -> str:
    candidates = [
        REPO_ROOT / "docs" / "wiki" / "system_prompts" / "h3_control_vector_system_prompt.md",
        REPO_ROOT / "docs" / "research" / "control_vectors" / "2026-11-05_control_vector_system_prompt.md",
    ]
    open_fence = "````markdown\n"
    for sp_path in candidates:
        if not sp_path.is_file():
            continue
        raw = sp_path.read_text(encoding="utf-8")
        fence_start = raw.find(open_fence)
        if fence_start >= 0:
            body_start = fence_start + len(open_fence)
            fence_end = raw.find("\n````", body_start)
            if fence_end >= 0:
                return raw[body_start:fence_end].strip()
    return ""


SUPPORTED_LLM_PROVIDERS: tuple[str, ...] = ("transformers", "exllamav3")


def _resolve_exl3_library_root(explicit_root: str | None = None) -> Path | None:
    """Locate the `exllamav3` Python package root from `explicit_root`, `EXLLAMAV3_ROOT`, or sibling directories."""
    for raw in (explicit_root, os.environ.get("EXLLAMAV3_ROOT", "")):
        s = str(raw or "").strip()
        if s and (Path(s).expanduser() / "exllamav3").is_dir():
            return Path(s).expanduser().resolve()
    for parent in [REPO_ROOT.parent, *REPO_ROOT.parents]:
        if (parent / "exllamav3").is_dir():
            return parent
    base = REPO_ROOT.parents[2] if len(REPO_ROOT.parents) >= 3 else REPO_ROOT.parent
    for cand in base.glob("*/*"):
        if cand.name == "exllamav3" and (cand / "exllamav3").is_dir():
            return cand.resolve()
    return None


def _generate_pairs_with_llm(
    *,
    provider: str,
    model_spec: str,
    concept_prompt: str,
    system_prompt: str | None = None,
    load_in_4bit: bool = True,
    max_new_tokens: int = 3200,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    rep_p: float = 1.05,
    seed: int = 42,
    exl3_root: str | None = None,
    llm_cache: dict[tuple[str, str, bool], Any] | None = None,
) -> dict[str, Any]:
    """Generate H3 control-vector Markdown pairs using a user-selected `provider` (`transformers` or `exllamav3`)
    and any user-specified `model_spec` (local directory path or HuggingFace repo ID).
    """
    prov = str(provider or "").strip().lower()
    if prov in ("exl3", "exllama", "exllamav3"):
        prov = "exllamav3"
    elif prov in ("hf", "huggingface", "transformers"):
        prov = "transformers"
    else:
        raise ValueError(
            f"unsupported LLM provider {provider!r}; expected one of {SUPPORTED_LLM_PROVIDERS}"
        )

    mspec = str(model_spec or "").strip()
    if not mspec:
        mspec = (
            os.environ.get("H3_LLM_MODEL", "").strip()
            or os.environ.get("H3_EXL3_MODEL_DIR", "").strip()
        )
    if not mspec:
        raise ValueError(
            "LLM model path or HuggingFace model ID is required (pass --llm-model, set H3_LLM_MODEL, or enter it in the Studio UI)"
        )

    concept = str(concept_prompt or "").strip()
    if not concept:
        raise ValueError("concept description is required to generate pairs")

    sys_txt = (system_prompt or _extract_system_prompt_text()).strip()
    if not sys_txt:
        sys_txt = (
            "You are an expert MiniMax H3 prompt engineer. Author 10 contrasting Prompt A (Baseline) / "
            "Prompt B (Steered) pairs in Markdown using `#### Pair N`, `##### Prompt A (Baseline)`, and "
            "`##### Prompt B (Steered)` headings where only the target concept clause changes."
        )

    user_msg = (
        f"Target Control Vector Concept / Axis:\n{concept}\n\n"
        "Generate the complete Markdown control-vector pack (`### Test Scene Prompt` followed by "
        "`#### Pair 1` through `#### Pair 10` with `##### Prompt A (Baseline)` and `##### Prompt B (Steered)`)."
    )

    t0 = time.perf_counter()
    cache_store = llm_cache if llm_cache is not None else {}
    cache_key = (prov, mspec, bool(load_in_4bit))

    if prov == "exllamav3":
        ex_root = _resolve_exl3_library_root(exl3_root)
        if ex_root is not None and str(ex_root) not in sys.path:
            sys.path.insert(0, str(ex_root))
        try:
            from exllamav3 import Cache, CacheLayer_quant, ComboSampler, Config, Model, Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "ExLlamaV3 ('exllamav3') is not importable; set EXLLAMAV3_ROOT to your exllamav3 checkout or select the 'transformers' provider."
            ) from exc

        model_dir = Path(mspec).expanduser()
        if not model_dir.is_dir():
            raise ValueError(f"ExLlamaV3 model directory not found: {mspec!r}")

        if cache_key in cache_store:
            model, cache, tokenizer = cache_store[cache_key]
        else:
            config = Config.from_directory(str(model_dir))
            tokenizer = Tokenizer.from_config(config)
            model = Model.from_config(config)
            cache = Cache(
                model,
                max_num_tokens=6144,
                layer_type=CacheLayer_quant,
                k_bits=4,
                v_bits=4,
            )
            model.load(progressbar=False)
            cache_store[cache_key] = (model, cache, tokenizer)

        chatml = (
            f"<|im_start|>system\n{sys_txt}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        context_ids = tokenizer.encode(
            chatml,
            add_bos=False,
            add_eos=False,
            encode_special_tokens=True,
        )
        sampler = ComboSampler(
            rep_p=float(rep_p),
            temperature=float(temperature),
            min_p=0.0,
            top_k=int(top_k),
            top_p=float(top_p),
        )
        stop_ids: set[int] = set()
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            stop_ids.add(int(tokenizer.eos_token_id))
        for tok_s in ("<|im_end|>", "<|endoftext|>"):
            try:
                enc_s = tokenizer.encode(tok_s, add_bos=False, add_eos=False, encode_special_tokens=True)
                if enc_s.numel() == 1:
                    stop_ids.add(int(enc_s.item()))
            except Exception:
                pass

        prefill_params: dict[str, Any] = {
            "attn_mode": "flash_attn",
            "cache": cache,
            "past_len": 0,
            "batch_shape": (1, cache.max_num_tokens),
        }
        if context_ids.shape[-1] > 1:
            model.prefill(input_ids=context_ids[:, :-1], params=prefill_params)

        gen_ids: list[int] = []
        for step_idx in range(int(max_new_tokens)):
            step_params: dict[str, Any] = {
                "attn_mode": "flash_attn",
                "cache": cache,
                "past_len": context_ids.shape[-1] - 1,
                "batch_shape": (1, cache.max_num_tokens),
            }
            logits = model.forward(input_ids=context_ids[:, -1:], params=step_params)
            rand_u32 = int.from_bytes(
                hashlib.blake2b(f"{seed}:{step_idx}".encode("utf-8"), digest_size=4).digest(),
                "little",
            )
            sample = sampler.forward(
                logits,
                sequence_ids=context_ids,
                rand_u32=rand_u32,
                tokenizer=tokenizer,
            )
            tid = int(sample.item())
            if tid in stop_ids:
                break
            gen_ids.append(tid)
            context_ids = torch.cat((context_ids, sample.cpu()), dim=-1)

        out_tensor = torch.tensor([gen_ids], dtype=torch.long) if gen_ids else torch.zeros((1, 0), dtype=torch.long)
        markdown_out = tokenizer.decode(out_tensor, decode_special_tokens=False)[0].strip() if gen_ids else ""
    else:
        try:
            import transformers
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "HuggingFace 'transformers' is not importable in this Python environment."
            ) from exc

        resolved_target = str(Path(mspec).expanduser()) if Path(mspec).expanduser().exists() else mspec
        if cache_key in cache_store:
            hf_model, hf_tokenizer = cache_store[cache_key]
        else:
            hf_tokenizer = AutoTokenizer.from_pretrained(resolved_target, trust_remote_code=True)
            load_kwargs: dict[str, Any] = {
                "device_map": "auto",
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
            }
            if load_in_4bit and torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig

                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4",
                    )
                except Exception:
                    pass
            cfg = AutoConfig.from_pretrained(resolved_target, trust_remote_code=True)
            arch_names = " ".join(getattr(cfg, "architectures", None) or ())
            model_cls = AutoModelForCausalLM
            if "Qwen3VL" in arch_names and hasattr(transformers, "Qwen3VLForConditionalGeneration"):
                model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration")
            elif "Qwen2_5_VL" in arch_names and hasattr(transformers, "Qwen2_5_VLForConditionalGeneration"):
                model_cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration")
            hf_model = model_cls.from_pretrained(resolved_target, **load_kwargs)
            hf_model.eval()
            cache_store[cache_key] = (hf_model, hf_tokenizer)

        messages = [
            {"role": "system", "content": sys_txt},
            {"role": "user", "content": user_msg},
        ]
        if hasattr(hf_tokenizer, "apply_chat_template"):
            try:
                prompt_str = hf_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                prompt_str = hf_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        else:
            prompt_str = (
                f"<|im_start|>system\n{sys_txt}<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        inputs = hf_tokenizer(prompt_str, return_tensors="pt")
        target_dev = next(hf_model.parameters()).device
        inputs = {k: v.to(target_dev) for k, v in inputs.items()}
        with torch.inference_mode():
            out_ids = hf_model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=float(temperature) > 0.0,
                temperature=max(float(temperature), 1e-4),
                top_p=float(top_p),
                top_k=int(top_k),
                repetition_penalty=float(rep_p),
            )
        new_tokens = out_ids[0, inputs["input_ids"].shape[-1] :]
        markdown_out = hf_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    elapsed_s = round(time.perf_counter() - t0, 2)
    return {
        "ok": True,
        "provider": prov,
        "model": mspec,
        "markdown": markdown_out,
        "elapsed_s": elapsed_s,
    }



def cmd_generate_pairs(args: argparse.Namespace) -> int:
    res = _generate_pairs_with_llm(
        provider=args.llm_provider,
        model_spec=args.llm_model,
        concept_prompt=args.concept,
        load_in_4bit=not bool(getattr(args, "no_4bit", False)),
        max_new_tokens=int(getattr(args, "max_new_tokens", 3200)),
        temperature=float(getattr(args, "temperature", 0.7)),
        exl3_root=getattr(args, "exl3_root", None) or None,
    )
    md_text = res["markdown"]
    if getattr(args, "out_md", ""):
        out_p = Path(args.out_md)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(md_text + "\n", encoding="utf-8")
        print(f"wrote {out_p} ({res['elapsed_s']}s via {res['provider']})")
    else:
        print(md_text)
    if getattr(args, "save_toml", False) and getattr(args, "name", ""):
        saved = cv.write_markdown_to_toml(
            md_text,
            cv_nodes.PAIR_BANK_DIR,
            name=str(args.name).strip(),
            overwrite=bool(getattr(args, "overwrite", False)),
            bank_dir=cv_nodes.PROMPT_BANK_DIR,
        )
        print(f"wrote {_display_path(saved)}")
    return 0

