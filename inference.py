#!/usr/bin/env python3
"""
Zero-shot inference script for Moderator final step based on dev directory.

Usage:
  Basic usage (output defaults to sibling *_Ilocal directory):
    python inference.py --dev_dir output/dev/dev_xxx

  Specify output directory and limit to first N samples:
    python inference.py --dev_dir output/dev/dev_xxx \
      --out_dir output/dev_inference/dev_xxx_Ilocal \
      --limit 100

Optional:
  Model loading: --base_model --device_map --dtype --load_in_4bit --load_in_8bit --trust_remote_code
  Generation parameters: --max_new_tokens --temperature --top_p --top_k --do_sample
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm
import multiprocessing as mp

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

try:
    # peft is required to load LoRA adapters
    from peft import PeftModel, AutoPeftModelForCausalLM
except Exception as e:  # pragma: no cover
    AutoPeftModelForCausalLM = None
    PeftModel = None



def _str_to_dtype(dtype_str: str):
    if dtype_str == "auto":
        return None
    if dtype_str.lower() in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_str.lower() in {"fp16", "float16"}:
        return torch.float16
    if dtype_str.lower() in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def _read_chat_template(adapter_dir: str) -> Optional[str]:
    template_path = os.path.join(adapter_dir, "chat_template.jinja")
    if os.path.isfile(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def generate(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    eos_token_id: Optional[int] = None,
):
    # Prepare inputs with chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = inputs.to(model.device)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": bool(do_sample),
        "pad_token_id": pad_id,
    }
    if bool(do_sample):
        gen_kwargs.update({
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
        })
    else:
        # Deterministic decoding: greedy with full nucleus and no top-k filter
        gen_kwargs.update({
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
        })
    if eos_token_id is None and tokenizer.eos_token_id is not None:
        eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        gen_kwargs["eos_token_id"] = eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            **gen_kwargs,
        )

    # Decode only the generated continuation
    generated = outputs[0]
    prompt_len = input_ids.shape[-1]
    new_tokens = generated[prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text


def load_tokenizer(adapter_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    # Prefer local chat template if present
    local_template = _read_chat_template(adapter_dir)
    if local_template:
        tokenizer.chat_template = local_template
        pass
    return tokenizer


def load_model(
    adapter_dir: str,
    base_model_override: Optional[str] = None,
    device_map: str = "auto",
    dtype_str: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
):
    torch_dtype = _str_to_dtype(dtype_str)

    if base_model_override:
        # Explicitly load base then attach adapter
        base = AutoModelForCausalLM.from_pretrained(
            base_model_override,
            device_map=device_map,
            torch_dtype=torch_dtype if torch_dtype is not None else None,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
        )
        if PeftModel is None:
            raise RuntimeError("peft is required to attach LoRA adapter. Please `pip install peft`.")
        model = PeftModel.from_pretrained(base, adapter_dir)
    else:
        # Let PEFT auto-resolve base from adapter_config.json
        if AutoPeftModelForCausalLM is None:
            raise RuntimeError("peft is required to load LoRA adapter. Please `pip install peft`.")

        model = AutoPeftModelForCausalLM.from_pretrained(
            adapter_dir,
            device_map=device_map,
            torch_dtype=torch_dtype if torch_dtype is not None else None,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
        )
    try:
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
    except Exception:
        pass
    return model


def _strip_think_tags(text: str) -> str:
    try:
        import re as _re
        return _re.sub(r"<think>[\s\S]*?</think>\s*", "", text)
    except Exception:
        return text


def _extract_balanced_json(text: str) -> Optional[str]:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def extract_json_dict_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return None
    cleaned = _strip_think_tags(text).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        candidate = _extract_balanced_json(cleaned)
        if isinstance(candidate, str):
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                try:
                    import ast as _ast
                    obj = _ast.literal_eval(candidate)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    pass
    except Exception:
        pass
    try:
        import ast as _ast
        return _ast.literal_eval(cleaned)
    except Exception:
        return None


def _regex_extract_v_and_j(text: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        import re as _re
        v = None
        j = None
        # Verdict: match 'Verdict': '...'/"..."
        m_v = _re.search(r"[\"']Verdict[\"']\s*:\s*[\"']([^\"']+)[\"']", text, flags=_re.IGNORECASE)
        if m_v:
            v = m_v.group(1).strip()
        # Justification for Verdict: allow multiline
        m_j = _re.search(r"[\"']Justification for Verdict[\"']\s*:\s*[\"']([\s\S]*?)[\"']\s*(,|\})", text, flags=_re.IGNORECASE)
        if m_j:
            j = m_j.group(1).strip()
        return v, j
    except Exception:
        return None, None


def _load_prompts_for_sample(sample_fp: str) -> Optional[Dict[str, Any]]:
    try:
        base = os.path.basename(sample_fp)
        name, ext = os.path.splitext(base)
        cfg_path = os.path.join(os.path.dirname(sample_fp), f"{name}-config.json")
        if not os.path.isfile(cfg_path):
            return None
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        prompts = cfg.get("prompts")
        if isinstance(prompts, dict):
            return prompts
        return None
    except Exception:
        return None


def _fill_template(template: str, variables: Dict[str, str]) -> str:
    if not isinstance(template, str):
        return ""
    result = template
    try:
        for k, v in variables.items():
            result = result.replace(f"##{k}##", v if v is not None else "")
        return result
    except Exception:
        return template


def _extract_sample_data(sample_data: Dict[str, Any]) -> Tuple[Optional[str], str, str, str]:
    """Extract sample data, returns (claim_text, evidence_text, aff_final, neg_final)"""
    claim_text = None
    try:
        if isinstance(sample_data.get("claim"), dict):
            claim_text = sample_data["claim"].get("claim") or sample_data["claim"].get("text")
        elif isinstance(sample_data.get("claim"), str):
            claim_text = sample_data.get("claim")
    except Exception:
        claim_text = None
    evidence_text = ""
    try:
        ev = sample_data.get("evidence")
        if isinstance(ev, list) and len(ev) > 0:
            items = []
            for idx, e in enumerate(ev):
                q = e.get("question") or ""
                a = e.get("answer") or ""
                items.append(f"Evidence_ID: {idx}")
                items.append(f"Content:{{Question: {q}")
                items.append(f"Answer: {a}}}")
                items.append("")
            if items and items[-1] == "":
                items.pop()
            evidence_text = "\n".join(items)
    except Exception:
        evidence_text = ""
    aff_final = ""
    neg_final = ""
    try:
        rounds = sample_data.get("debate_rounds")
        if isinstance(rounds, list) and len(rounds) > 0:
            last_round = rounds[-1]
            aff_final = str(last_round.get("affirmative_response") or "")
            neg_final = str(last_round.get("negative_response") or "")
    except Exception:
        pass

    assert len(claim_text) > 0, f"claim_text is empty: {sample_data}"
    assert len(evidence_text) > 0, f"evidence_text is empty: {sample_data}"
    assert len(aff_final) > 0, f"aff_final is empty: {sample_data}"
    assert len(neg_final) > 0, f"neg_final is empty: {sample_data}"
    return claim_text, evidence_text, aff_final, neg_final


def _parse_verdict_and_justification(output: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse verdict and justification from output"""
    parsed = extract_json_dict_from_text(output)
    if not isinstance(parsed, dict):
        v_fallback, j_fallback = _regex_extract_v_and_j(output)
        if v_fallback and j_fallback:
            parsed = {"Verdict": v_fallback, "Justification for Verdict": j_fallback}
        else:
            return None, None
    
    v = parsed.get("Verdict") or parsed.get("verdict")
    j = parsed.get("Justification for Verdict") or parsed.get("justification")
    return v, j


def _build_final_judgment_messages_simple(sample_data: Dict[str, Any]) -> Tuple[str, str, str]:
    claim_text, evidence_text, aff_final, neg_final = _extract_sample_data(sample_data)

    system_prompt ="""You are the judge in a fact-checking debate where two debaters examine the truthfulness of a given claim by presenting credible supporting or opposing evidence. Your goal is to facilitate a fact-based evaluation of the claim, ensuring each side maintains a strong, evidence-backed stance.\n\nResponsibilities:\n- Guide each debate round, ensuring arguments remain evidence-based.\n- Assess the relevance and strength of the credible evidence presented by both sides.\n- Determine if further rounds are essential based on the new insights provided. End the debate if the both sides are just repeating their previous arguments without bringing new insights.\n\nClaim under Review:\n##debate_topic##\n\nDetailed Verdict Criteria:\n- Supported: The knowledge from the fact-check supports or at least strongly implies the Claim. Mere plausibility is not enough for this decision.\n- Not Enough Evidence: The fact-check does not contain sufficient information to come to a conclusion. For example, there is substantial lack of evidence. In this case, state which information exactly is missing. In particular, if no RESULTS or sources are available, pick this decision.\n- Refuted: The knowledge from the fact-check clearly refutes the Claim. The mere absence or lack of supporting evidence is not enough reason for being refuted (argument from ignorance).\n- Conflicting Evidence: The knowledge from the fact-check contains conflicting evidence from multiple RELIABLE sources. Even trying to resolve the conflicting sources through additional investigation was not successful.\n\nCredible Evidence Set:\n##evidence_set##"""
    judge_prompt_last1 ="""Affirmative Argument:\n##aff_ans##\n\nNegative Argument:\n##neg_ans##\n\nSummarize the primary insights gathered throughout the entire debate concisely."""
    judge_prompt_last2 ="""After reviewing both sides' arguments on the claim:\n##debate_topic##\n\nSelect a Verdict from the following labels based on the credible evidence. Remember the detailed criteria:\n- Supported: The knowledge from the fact-check supports or at least strongly implies the Claim. Mere plausibility is not enough for this decision.\n- Not Enough Evidence: The fact-check does not contain sufficient information to come to a conclusion. For example, there is substantial lack of evidence. In this case, state which information exactly is missing. In particular, if no RESULTS or sources are available, pick this decision.\n- Refuted: The knowledge from the fact-check clearly refutes the Claim. The mere absence or lack of supporting evidence is not enough reason for being refuted (argument from ignorance).\n- Conflicting Evidence: The knowledge from the fact-check contains conflicting evidence from multiple RELIABLE sources. Even trying to resolve the conflicting sources through additional investigation was not successful.\n\nIgnore any irrelevant or unsupported information.\n\nProvide your verdict its Justification in JSON format:\n{\n  \"Verdict\": \"...\",\n  \"Justification for Verdict\": \"...\"\n}"""
    
    system_prompt_filled = _fill_template(system_prompt, {
        "debate_topic": claim_text or "",
        "evidence_set": evidence_text or "",
    })
    judge_prompt_last1_filled = _fill_template(judge_prompt_last1, {
        "aff_ans": aff_final,
        "neg_ans": neg_final,
    })
    judge_prompt_last2_filled = _fill_template(judge_prompt_last2, {
        "debate_topic": claim_text or "",
    })

    return system_prompt_filled, judge_prompt_last1_filled, judge_prompt_last2_filled


def _build_judge_messages_step1(sample_data: Dict[str, Any], prompts: Dict[str, Any]) -> List[Dict[str, str]]:
    claim_text, evidence_text, aff_final, neg_final = _extract_sample_data(sample_data)

    judge_meta = prompts.get("judge_meta_prompt", "")
    judge_meta_filled = _fill_template(judge_meta, {
        "debate_topic": claim_text or "",
        "evidence_set": evidence_text or "",
    })
    judge_last1 = prompts.get("judge_prompt_last1", "")
    judge_last1_filled = _fill_template(judge_last1, {
        "aff_ans": aff_final,
        "neg_ans": neg_final,
    })

    msgs = []
    if judge_meta_filled:
        msgs.append({"role": "system", "content": judge_meta_filled})
    msgs.append({"role": "user", "content": judge_last1_filled})
    return msgs


def extract_moderator_context_and_target(debate_data: Dict[str, Any]) -> Tuple[List[Dict[str, str]], Optional[Dict[str, Any]]]:
    participants = debate_data.get("participants", {})
    if len(participants) == 0:
        participants = debate_data.get("players", {})
    moderator_hist = participants.get("Moderator") or participants.get("moderator")
    if not isinstance(moderator_hist, list) or len(moderator_hist) == 0:
        return [], None
    last_assistant_idx = -1
    for i in range(len(moderator_hist) - 1, -1, -1):
        if moderator_hist[i].get("role") == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx < 0:
        return moderator_hist, None
    context = moderator_hist[:last_assistant_idx]
    target = moderator_hist[last_assistant_idx]
    context_msgs: List[Dict[str, str]] = []
    for m in context:
        role = m.get("role")
        content = m.get("content")
        if role is None or content is None:
            continue
        context_msgs.append({"role": role, "content": content})
    return context_msgs, target


def iter_sample_files(dev_dir: str) -> List[str]:
    files: List[str] = []
    try:
        for name in os.listdir(dev_dir):
            if not name.endswith(".json"):
                continue
            if name.endswith("-config.json"):
                continue
            files.append(os.path.join(dev_dir, name))
    except Exception:
        return []

    def _num_key(path: str) -> int:
        base = os.path.basename(path)
        name, _ = os.path.splitext(base)
        try:
            return int(name)
        except Exception:
            parts = name.split("-")
            for p in parts:
                if p.isdigit():
                    return int(p)
            return 1 << 30

    files.sort(key=_num_key)
    return files


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _extract_claim_id_and_success(obj: Dict[str, Any]) -> Optional[str]:
    try:
        cid = obj.get("claim_id")
        success = obj.get("success")
        if cid is None and isinstance(obj.get("claim"), dict):
            cid = obj["claim"].get("id") or obj["claim"].get("claim_id")
        if cid is None:
        # Support alternative field names
            cid = obj.get("id")
        return str(cid) if cid is not None else None, success
    except Exception:
        return None, None


def _collect_done_claim_ids(out_dir: str) -> set:
    done = set()
    if not os.path.isdir(out_dir):
        return done
    try:
        for name in os.listdir(out_dir):
            if not name.endswith(".json"):
                continue
            fp = os.path.join(out_dir, name)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cid, success = _extract_claim_id_and_success(data)
                if cid is not None and success:
                    done.add(cid)
            except Exception:
                pass
    except Exception:
        pass
    return done


def run_moderator_last_step(
    model,
    tokenizer,
    dev_dir: str,
    out_dir: Optional[str],
    files: Optional[List[str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
):
    if out_dir is None or len(str(out_dir).strip()) == 0:
        parent = os.path.dirname(dev_dir)
        base = os.path.basename(dev_dir)
        out_dir = os.path.join(parent, f"{base}_Ilocal")
    ensure_dir(out_dir)

    if files is None:
        files = iter_sample_files(dev_dir)
    
    processed = 0
    skipped = 0
    for fp in tqdm(files, desc="Inferencing Claims"):
        out_fp = os.path.join(out_dir, os.path.basename(fp))
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            skipped += 1
            continue

        context_msgs, _target = extract_moderator_context_and_target(data)

        if len(context_msgs) == 0:
            raise ValueError(f"Moderator context is empty or invalid: {fp}, claim_id: {data.get('claim_id')}")

        success = False
        last_output = None
        try:
            output = generate(
                model,
                tokenizer,
                context_msgs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
            )
            last_output = output
            v, j = _parse_verdict_and_justification(output)
            if not v or not j:
                raise ValueError(f"Cannot parse verdict result in {fp}")
            
            parsed = {"Verdict": v, "Justification for Verdict": j}
            data["verdict"] = v
            data["justification"] = j
            success = True
        except Exception:
            last_output = None

        if not success:
            try:
                system_prompt_filled, judge_prompt_last1_filled, judge_prompt_last2_filled = _build_final_judgment_messages_simple(data)
                fj_messages_step1 = [
                    {"role": "system", "content": system_prompt_filled},
                    {"role": "user", "content": judge_prompt_last1_filled},
                ]
                fj_output_step1 = generate(
                    model,
                    tokenizer,
                    fj_messages_step1,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                )
                fj_messages_step2 = [
                    {"role": "system", "content": system_prompt_filled},
                    {"role": "user", "content": judge_prompt_last1_filled},
                    {"role": "assistant", "content": fj_output_step1},
                    {"role": "user", "content": judge_prompt_last2_filled},
                ]
                fj_output_step2 = generate(
                    model,
                    tokenizer,
                    fj_messages_step2,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                )
                v, j = _parse_verdict_and_justification(fj_output_step2)
                if not v or not j:
                    raise ValueError(f"Final judgment missing fields: {fj_output_step2}")
                data["verdict"] = v
                data["justification"] = j
                output = fj_output_step2
                parsed = {"Verdict": v, "Justification for Verdict": j}
            except Exception as fj_err:
                raise ValueError(f"Fallback judgment failed, skipping: {fp} error: {fj_err}") 
        try:
            if isinstance(data.get("debate_rounds"), list) and len(data["debate_rounds"]) > 0:
                data["debate_rounds"][-1]["moderator_response"] = parsed
        except Exception:
            pass

        try:
            participants = data.get("participants", {})
            mod_hist = participants.get("Moderator") or participants.get("moderator")
            if isinstance(mod_hist, list):
                last_idx = -1
                for i in range(len(mod_hist) - 1, -1, -1):
                    if mod_hist[i].get("role") == "assistant":
                        last_idx = i
                        break
                if last_idx >= 0:
                    mod_hist[last_idx]["content"] = output
        except Exception:
            pass



        data["inference_raw_output"] = output
        data["inference_raw_parsed"] = parsed

        try:
            with open(out_fp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            processed += 1
        except Exception:
            skipped += 1


def _create_task_pool(files: List[str], out_dir: str) -> List[str]:
    """Create task pool, skip already processed files"""
    if not files:
        return []
    
    done_claim_ids = _collect_done_claim_ids(out_dir)
    
    unprocessed_files = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            claim_id, _ = _extract_claim_id_and_success(data)
            if claim_id is None or claim_id not in done_claim_ids:
                unprocessed_files.append(fp)
        except Exception:
            unprocessed_files.append(fp)
    
    return unprocessed_files


def _worker_process(task_queue: mp.Queue, done_set: Any, args_dict: Dict[str, Any], process_id: int) -> None:
    try:
        # Torch settings per process
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_grad_enabled(False)

        pass
        
        tokenizer = load_tokenizer(args_dict["adapter_dir"])
        model = load_model(
            adapter_dir=args_dict["adapter_dir"],
            base_model_override=args_dict.get("base_model"),
            device_map=args_dict["device_map"],
            dtype_str=args_dict["dtype"],
            load_in_8bit=args_dict["load_in_8bit"],
            load_in_4bit=args_dict["load_in_4bit"],
            trust_remote_code=args_dict["trust_remote_code"],
        )

        processed = 0
        while True:
            try:
                file_path = task_queue.get(timeout=1)
                if file_path is None:
                    break
                
                if file_path in done_set:
                    continue
                
                done_set[file_path] = True
                run_moderator_last_step(
                    model=model,
                    tokenizer=tokenizer,
                    dev_dir=args_dict["dev_dir"],
                    out_dir=args_dict["out_dir"],
                    files=[file_path],  # Process single file only
                    max_new_tokens=args_dict["max_new_tokens"],
                    temperature=args_dict["temperature"],
                    top_p=args_dict["top_p"],
                    top_k=args_dict["top_k"],
                    do_sample=args_dict["do_sample"],
                )
                processed += 1
                
            except Exception as e:
                error_str = str(e)
                error_type = str(type(e))
                
                if "Empty" in error_type or "timeout" in error_str.lower():
                    continue
                else:
                    continue
                
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Moderator final step inference (based on dev directory)")
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="",
        help="Path to LoRA adapter directory",
    )
    parser.add_argument("--base_model", type=str, default=None, help="Override base model repo or path (optional)")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map, e.g., 'auto' or 'cuda' or 'cpu'")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"], help="Torch dtype for loading")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit (requires bitsandbytes)")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit (requires bitsandbytes)")
    parser.add_argument("--trust_remote_code", action="store_true", help="Set True if base model requires custom code")
    parser.add_argument("--num_procs", type=int, default=1, help="Number of parallel processes, >1 enables multiprocessing")

    parser.add_argument("--dev_dir", type=str, required=True, help="Path to dev results directory")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for inference results (optional)")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N samples; 0 means all")

    # Generation options
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling (default off for deterministic)")

    args = parser.parse_args()

    # Defaults: sampling off unless explicitly enabled
    do_sample = bool(args.do_sample)

    # Torch settings
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    # Dev mode only
    dev_dir = args.dev_dir
    if not os.path.isdir(dev_dir):
        sys.exit(2)
    files = iter_sample_files(dev_dir)
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    if isinstance(args.num_procs, int) and args.num_procs > 1:
        try:
            mp.set_start_method("spawn", force=True)
        except Exception:
            pass

        out_dir = args.out_dir
        if out_dir is None or len(str(out_dir).strip()) == 0:
            parent = os.path.dirname(dev_dir)
            base = os.path.basename(dev_dir)
            out_dir = os.path.join(parent, f"{base}_Ilocal")
        ensure_dir(out_dir)
        
        task_files = _create_task_pool(files, out_dir)
        
        if not task_files:
            return
            
        manager = mp.Manager()
        task_queue = mp.Queue()
        done_set = manager.dict()
        
        for file_path in task_files:
            task_queue.put(file_path)
        
        for _ in range(args.num_procs):
            task_queue.put(None)

        args_dict: Dict[str, Any] = {
            "adapter_dir": args.adapter_dir,
            "base_model": args.base_model,
            "device_map": args.device_map,
            "dtype": args.dtype,
            "load_in_8bit": bool(args.load_in_8bit),
            "load_in_4bit": bool(args.load_in_4bit),
            "trust_remote_code": bool(args.trust_remote_code),
            "dev_dir": dev_dir,
            "out_dir": out_dir,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "do_sample": do_sample,
        }

        processes: List[mp.Process] = []
        for i in range(args.num_procs):
            p = mp.Process(target=_worker_process, args=(task_queue, done_set, args_dict, i))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            
        return
    tokenizer = load_tokenizer(args.adapter_dir)
    model = load_model(
        adapter_dir=args.adapter_dir,
        base_model_override=args.base_model,
        device_map=args.device_map,
        dtype_str=args.dtype,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
    )

    run_moderator_last_step(
        model=model,
        tokenizer=tokenizer,
        dev_dir=dev_dir,
        out_dir=args.out_dir,
        files=files,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=do_sample,
    )


if __name__ == "__main__":
    main()




