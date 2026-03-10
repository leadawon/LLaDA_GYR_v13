import math
import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def top_p_filtering(logits, top_p):
    if top_p is None or top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = torch.zeros_like(sorted_indices_to_remove, dtype=torch.bool)
    indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    return logits.masked_fill(indices_to_remove, -torch.inf)


def _as_bool(v, default=False):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off", ""}:
        return False
    return default


def _parse_proxy_cfg(raw):
    cfg = {
        "name": "qk_log_softmax",
        "layer_set": "last_only",
        "head_reduce": "mean",
        "layer_reduce": "mean",
        "threshold": 0.2,
        "yg_threshold": None,
        "sym": "max",
        "yg_mode": "directed",
        "outgoing_lambda": 0.0,
        "outgoing_metric": "max",
        "outgoing_include_green": False,
        "attn_temp": 1.0,
    }
    if raw is None:
        return cfg

    data = {}
    if isinstance(raw, dict):
        data = dict(raw)
    else:
        txt = str(raw).strip()
        if txt:
            for item in txt.split(";"):
                item = item.strip()
                if not item or "=" not in item:
                    continue
                k, v = item.split("=", 1)
                data[k.strip()] = v.strip()

    for k, v in data.items():
        if k in {"threshold", "yg_threshold", "outgoing_lambda", "attn_temp"}:
            try:
                cfg[k] = float(v)
            except Exception:
                pass
        elif k in {"outgoing_include_green"}:
            cfg[k] = _as_bool(v, default=cfg[k])
        else:
            cfg[k] = v

    cfg["name"] = str(cfg.get("name", "qk_log_softmax")).strip().lower()
    cfg["layer_set"] = str(cfg.get("layer_set", "last_only")).strip().lower()
    cfg["head_reduce"] = str(cfg.get("head_reduce", "mean")).strip().lower()
    cfg["layer_reduce"] = str(cfg.get("layer_reduce", "mean")).strip().lower()
    cfg["sym"] = str(cfg.get("sym", "max")).strip().lower()
    cfg["yg_mode"] = str(cfg.get("yg_mode", "directed")).strip().lower()
    cfg["outgoing_metric"] = str(cfg.get("outgoing_metric", "max")).strip().lower()

    if cfg["yg_mode"] not in {"directed", "symmetric"}:
        cfg["yg_mode"] = "directed"
    if cfg["head_reduce"] not in {"mean", "max"}:
        cfg["head_reduce"] = "mean"
    if cfg["layer_reduce"] not in {"mean", "max", "min"}:
        cfg["layer_reduce"] = "mean"
    if cfg["sym"] not in {"none", "max", "mean", "min"}:
        cfg["sym"] = "max"
    if cfg["outgoing_metric"] not in {"max", "mean", "topk_mean", "topk_mean:2", "topk_mean:3"}:
        cfg["outgoing_metric"] = "max"

    return cfg


def _resolve_layer_indices(num_layers, layer_set):
    if num_layers <= 0:
        return []
    if layer_set == "last_only":
        return [num_layers - 1]
    if layer_set == "mid_only":
        return [num_layers // 2]
    if layer_set == "early_mid_late":
        return sorted(set([0, num_layers // 2, num_layers - 1]))
    return [num_layers - 1]


def _unwrap_model_for_hooks(model):
    return model.module if hasattr(model, "module") else model


def _discover_qk_layers(core_model):
    layers = []
    seen = set()
    for name, mod in core_model.named_modules():
        if not hasattr(mod, "q_proj") or not hasattr(mod, "k_proj"):
            continue
        if not isinstance(getattr(mod, "q_proj"), torch.nn.Module):
            continue
        if not isinstance(getattr(mod, "k_proj"), torch.nn.Module):
            continue
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        layers.append((name, mod))
    return layers


def _register_qk_hooks(model, capture_layer_ids):
    core = _unwrap_model_for_hooks(model)
    discovered = _discover_qk_layers(core)
    if not discovered:
        raise RuntimeError("[yellow] proxy_qk requested but q_proj/k_proj layers were not found.")

    total_layers = len(discovered)
    valid = []
    for lid in capture_layer_ids:
        lid = int(lid)
        if 0 <= lid < total_layers:
            valid.append(lid)
    valid = sorted(set(valid))
    if not valid:
        raise RuntimeError("[yellow] proxy_qk requested but no valid capture layers were resolved.")

    captures = {lid: {"q": None, "k": None} for lid in valid}
    attn_mods = {lid: discovered[lid][1] for lid in valid}
    hooks = []

    def _make_hook(layer_id, key):
        def _hook(_module, _inp, out):
            captures[layer_id][key] = out
            return None
        return _hook

    for lid in valid:
        m = attn_mods[lid]
        hooks.append(m.q_proj.register_forward_hook(_make_hook(lid, "q")))
        hooks.append(m.k_proj.register_forward_hook(_make_hook(lid, "k")))

    return {
        "captures": captures,
        "attn_mods": attn_mods,
        "hooks": hooks,
        "total_layers": total_layers,
        "layer_ids": valid,
    }


def _clear_qk_captures(state):
    if state is None:
        return
    for lid in state["layer_ids"]:
        state["captures"][lid]["q"] = None
        state["captures"][lid]["k"] = None


def _remove_qk_hooks(state):
    if state is None:
        return
    for h in state.get("hooks", []):
        try:
            h.remove()
        except Exception:
            pass


def _reshape_qk_from_proj(attn_mod, q_lin, k_lin):
    bsz, seq_len, q_dim = q_lin.shape
    _, _, k_dim = k_lin.shape

    num_heads = None
    kv_heads = None
    head_dim = None

    cfg = getattr(attn_mod, "config", None)
    if cfg is not None:
        num_heads = int(getattr(cfg, "n_heads", 0) or 0)
        kv_heads = int(getattr(cfg, "effective_n_kv_heads", 0) or 0)
        if num_heads > 0:
            head_dim = int(getattr(cfg, "d_model", q_dim) // num_heads)

    if num_heads is None or num_heads <= 0:
        num_heads = int(getattr(attn_mod, "num_heads", 0) or 0)
    if num_heads <= 0:
        hd = int(getattr(attn_mod, "head_dim", 0) or 0)
        if hd > 0:
            num_heads = max(1, q_dim // hd)
        else:
            num_heads = 1

    if head_dim is None or head_dim <= 0:
        hd = int(getattr(attn_mod, "head_dim", 0) or 0)
        if hd > 0:
            head_dim = hd
        else:
            head_dim = max(1, q_dim // num_heads)

    if kv_heads is None or kv_heads <= 0:
        kv_heads = max(1, k_dim // head_dim)

    q = q_lin.view(bsz, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    k = k_lin.view(bsz, seq_len, kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    if kv_heads != num_heads:
        if num_heads % kv_heads == 0:
            rep = num_heads // kv_heads
            k = k.repeat_interleave(rep, dim=1)
        else:
            rep = int(math.ceil(num_heads / kv_heads))
            k = k.repeat(1, rep, 1, 1)[:, :num_heads]

    return q, k


def _collect_captured_qk(state, batch_size):
    out = {}
    for lid in state["layer_ids"]:
        rec = state["captures"][lid]
        q_lin = rec.get("q")
        k_lin = rec.get("k")
        if q_lin is None or k_lin is None:
            continue
        q, k = _reshape_qk_from_proj(state["attn_mods"][lid], q_lin, k_lin)
        if batch_size is not None and q.shape[0] >= batch_size:
            q = q[:batch_size]
            k = k[:batch_size]
        out[lid] = (q, k)
    return out


def _reduce_layer_scores(stacked, reduce_mode):
    if reduce_mode == "max":
        return stacked.max(dim=0).values
    if reduce_mode == "min":
        return stacked.min(dim=0).values
    return stacked.mean(dim=0)


def _symmetrize_square(scores, sym):
    if sym == "none":
        return scores
    if sym == "mean":
        return 0.5 * (scores + scores.transpose(0, 1))
    if sym == "min":
        return torch.minimum(scores, scores.transpose(0, 1))
    return torch.maximum(scores, scores.transpose(0, 1))


def _symmetrize_rect(yg, gy_t, sym):
    if sym == "none":
        return yg
    if sym == "mean":
        return 0.5 * (yg + gy_t)
    if sym == "min":
        return torch.minimum(yg, gy_t)
    return torch.maximum(yg, gy_t)


def _compute_proxy_scores_row(row_layers, src_idx, dst_idx, proxy_cfg):
    if src_idx.numel() == 0 or dst_idx.numel() == 0:
        return None

    head_reduce = proxy_cfg.get("head_reduce", "mean")
    layer_reduce = proxy_cfg.get("layer_reduce", "mean")
    name = proxy_cfg.get("name", "qk_log_softmax")
    attn_temp = max(float(proxy_cfg.get("attn_temp", 1.0)), 1e-6)

    layer_scores = []
    for _, (q_row, k_row) in row_layers.items():
        q_sel = q_row.index_select(1, src_idx)
        k_sel = k_row.index_select(1, dst_idx)

        raw_h = torch.einsum("hmd,hnd->hmn", q_sel, k_sel) / math.sqrt(q_row.shape[-1])
        if head_reduce == "max":
            sc = raw_h.max(dim=0).values
        else:
            sc = raw_h.mean(dim=0)
        layer_scores.append(sc)

    if not layer_scores:
        return None

    stacked = torch.stack(layer_scores, dim=0)
    scores = _reduce_layer_scores(stacked, layer_reduce)

    if name == "qk_log_softmax":
        scores = torch.log_softmax(scores.float() / attn_temp, dim=-1)
    elif name == "qk_softmax":
        scores = torch.softmax(scores.float() / attn_temp, dim=-1)
    else:
        scores = scores.float()

    return scores


def _reduce_outgoing_scores(values, metric="max"):
    rows = int(values.shape[0]) if values.dim() >= 1 else 0
    if rows == 0:
        return torch.zeros((0,), device=values.device, dtype=values.dtype)
    cols = int(values.shape[1]) if values.dim() >= 2 else 0
    if cols == 0:
        return torch.zeros((rows,), device=values.device, dtype=values.dtype)

    m = str(metric or "max").strip().lower()
    if m == "mean":
        return values.mean(dim=-1)
    if m.startswith("topk_mean"):
        k = 2
        if ":" in m:
            try:
                k = int(m.split(":", 1)[1].strip())
            except Exception:
                k = 2
        k = max(1, min(k, cols))
        return torch.topk(values, k, dim=-1).values.mean(dim=-1)
    return values.max(dim=-1).values


def _compute_outgoing_dependency(yy_dir, yg_vals, metric="max", include_green=False):
    m = int(yy_dir.shape[0])
    if m <= 0:
        return torch.zeros((0,), device=yy_dir.device, dtype=yy_dir.dtype)

    if m > 1:
        offdiag_mask = ~torch.eye(m, dtype=torch.bool, device=yy_dir.device)
        yy_vals = yy_dir[offdiag_mask].view(m, m - 1)
    else:
        yy_vals = yy_dir.new_empty((m, 0))

    combined = yy_vals
    if include_green and yg_vals is not None and yg_vals.numel() > 0:
        combined = torch.cat([combined, yg_vals], dim=-1) if combined.numel() > 0 else yg_vals

    return _reduce_outgoing_scores(combined, metric)


def _topk_from_values(values, k):
    if k <= 0 or values.numel() == 0:
        return values.new_zeros((0,), dtype=torch.long)
    k = min(int(k), int(values.numel()))
    return torch.topk(values, k=k, largest=True).indices


def _select_transfer_positions_standard(
    row_conf,
    row_mask,
    base_k,
    gr_enabled,
    yellow_enabled,
    green_conf_thresh,
    yellow_conf_thresh,
    probabilistic,
    prob_tau,
):
    row_positions = torch.where(row_mask)[0]
    if row_positions.numel() == 0:
        return row_positions.new_zeros((0,), dtype=torch.long)

    base_k = min(int(base_k), int(row_positions.numel()))
    if base_k <= 0:
        return row_positions.new_zeros((0,), dtype=torch.long)

    conf = row_conf[row_positions]

    if not gr_enabled:
        local = _topk_from_values(conf, base_k)
        return row_positions[local]

    green_local = conf >= float(green_conf_thresh)
    yellow_local = yellow_enabled & (conf >= float(yellow_conf_thresh)) & (~green_local)
    red_local = ~(green_local | yellow_local)

    high_cnt = int((green_local | yellow_local).sum().item())
    k_target = base_k
    if probabilistic and high_cnt > base_k:
        aggressive = max(0.0, min(1.0, float(prob_tau) * 6.0))
        bump = int((high_cnt - base_k) * aggressive)
        k_target = min(int(row_positions.numel()), base_k + bump)

    if probabilistic:
        tau = max(float(prob_tau), 1e-6)
        logits = conf / tau
        color_bonus = torch.zeros_like(logits)
        color_bonus[green_local] = 0.6
        color_bonus[yellow_local] = 0.2
        color_bonus[red_local] = -0.25
        probs = torch.softmax(logits + color_bonus, dim=0)
        if not torch.isnan(probs).any() and float(probs.sum().item()) > 0.0:
            sampled = torch.multinomial(probs, num_samples=k_target, replacement=False)
            return row_positions[sampled]

    ordered = []
    g_idx = torch.where(green_local)[0]
    y_idx = torch.where(yellow_local)[0]
    r_idx = torch.where(red_local)[0]

    if g_idx.numel() > 0:
        ordered.append(g_idx[torch.argsort(conf[g_idx], descending=True)])
    if y_idx.numel() > 0:
        ordered.append(y_idx[torch.argsort(conf[y_idx], descending=True)])
    if r_idx.numel() > 0:
        ordered.append(r_idx[torch.argsort(conf[r_idx], descending=True)])

    if not ordered:
        local = _topk_from_values(conf, k_target)
    else:
        local = torch.cat(ordered, dim=0)[:k_target]
    return row_positions[local]


def _select_transfer_positions_proxy(
    row_conf,
    row_mask,
    base_k,
    green_conf_thresh,
    yellow_conf_thresh,
    probabilistic,
    prob_tau,
    row_layers,
    proxy_cfg,
    committed_global,
):
    row_positions = torch.where(row_mask)[0]
    if row_positions.numel() == 0:
        return row_positions.new_zeros((0,), dtype=torch.long)

    base_k = min(int(base_k), int(row_positions.numel()))
    if base_k <= 0:
        return row_positions.new_zeros((0,), dtype=torch.long)

    conf = row_conf[row_positions]
    green_local = conf >= float(green_conf_thresh)
    yellow_local = (conf >= float(yellow_conf_thresh)) & (~green_local)

    high_cnt = int((green_local | yellow_local).sum().item())
    k_target = base_k
    if probabilistic and high_cnt > base_k:
        aggressive = max(0.0, min(1.0, float(prob_tau) * 6.0))
        bump = int((high_cnt - base_k) * aggressive)
        k_target = min(int(row_positions.numel()), base_k + bump)

    selected = []

    green_idx = torch.where(green_local)[0]
    if green_idx.numel() > 0:
        g_ord = green_idx[torch.argsort(conf[green_idx], descending=True)]
        for gi in g_ord.tolist():
            if len(selected) >= k_target:
                break
            selected.append(int(gi))

    rem = k_target - len(selected)
    if rem > 0 and yellow_local.any():
        y_idx = torch.where(yellow_local)[0]
        y_global = row_positions[y_idx]

        # Dream directional idea: use Y->G, not G->Y, unless yg_mode=symmetric.
        safe_mask = torch.ones((y_idx.numel(),), dtype=torch.bool, device=row_conf.device)
        yg_threshold = float(
            proxy_cfg.get("yg_threshold") if proxy_cfg.get("yg_threshold") is not None else proxy_cfg.get("threshold", 0.2)
        )
        yg_mode = str(proxy_cfg.get("yg_mode", "directed")).strip().lower()

        yg_vals_for_outgoing = None
        if committed_global.numel() > 0:
            yg_dir = _compute_proxy_scores_row(row_layers, y_global, committed_global, proxy_cfg)
            if yg_dir is None:
                raise RuntimeError("[yellow] proxy_qk failed to build Y->G scores; aborting to avoid fallback.")

            if yg_mode == "symmetric":
                gy_dir = _compute_proxy_scores_row(row_layers, committed_global, y_global, proxy_cfg)
                if gy_dir is None:
                    raise RuntimeError("[yellow] proxy_qk failed to build G->Y scores; aborting to avoid fallback.")
                yg_eval = _symmetrize_rect(yg_dir, gy_dir.transpose(0, 1), proxy_cfg.get("sym", "max"))
            else:
                yg_eval = yg_dir

            dep_to_green = yg_eval >= yg_threshold
            safe_mask = ~torch.any(dep_to_green, dim=1)
            yg_vals_for_outgoing = yg_dir

        y_safe_idx = y_idx[safe_mask]
        if y_safe_idx.numel() > 0:
            y_safe_global = row_positions[y_safe_idx]
            yy_dir = _compute_proxy_scores_row(row_layers, y_safe_global, y_safe_global, proxy_cfg)
            if yy_dir is None:
                raise RuntimeError("[yellow] proxy_qk failed to build Y->Y scores; aborting to avoid fallback.")

            yy_sym = _symmetrize_square(yy_dir, proxy_cfg.get("sym", "max"))
            dep_mask = yy_sym >= float(proxy_cfg.get("threshold", 0.2))
            safe_adj = ~dep_mask
            safe_adj.fill_diagonal_(True)

            outgoing_metric = str(proxy_cfg.get("outgoing_metric", "max"))
            outgoing_lambda = float(proxy_cfg.get("outgoing_lambda", 0.0))
            include_green = bool(proxy_cfg.get("outgoing_include_green", False))

            yg_subset = None
            if include_green and yg_vals_for_outgoing is not None and committed_global.numel() > 0:
                yg_subset = yg_vals_for_outgoing[safe_mask]

            outgoing = _compute_outgoing_dependency(
                yy_dir=yy_dir,
                yg_vals=yg_subset,
                metric=outgoing_metric,
                include_green=include_green,
            )

            adjusted_conf = conf[y_safe_idx]
            if float(outgoing_lambda) != 0.0 and outgoing.numel() == adjusted_conf.numel():
                adjusted_conf = adjusted_conf - (float(outgoing_lambda) * outgoing)

            order = torch.argsort(adjusted_conf, descending=True)
            clique = []
            for ord_i in order.tolist():
                cand = int(ord_i)
                if not clique:
                    clique.append(cand)
                    continue
                clique_t = torch.tensor(clique, device=row_conf.device, dtype=torch.long)
                if bool(torch.all(safe_adj[cand, clique_t]).item()):
                    clique.append(cand)

            if clique:
                y_pick = y_safe_idx[torch.tensor(clique, device=row_conf.device, dtype=torch.long)]
                for yi in y_pick.tolist():
                    if len(selected) >= k_target:
                        break
                    v = int(yi)
                    if v not in selected:
                        selected.append(v)

    if len(selected) < k_target:
        selected_set = set(selected)
        rest = torch.arange(row_positions.numel(), device=row_conf.device)
        keep = torch.tensor([int(i) not in selected_set for i in rest.tolist()], device=row_conf.device, dtype=torch.bool)
        rest = rest[keep]
        if rest.numel() > 0:
            rest_ord = rest[torch.argsort(conf[rest], descending=True)]
            for ri in rest_ord.tolist():
                if len(selected) >= k_target:
                    break
                selected.append(int(ri))

    if not selected:
        return row_positions.new_zeros((0,), dtype=torch.long)

    selected_t = torch.tensor(selected, device=row_conf.device, dtype=torch.long)
    return row_positions[selected_t]


@ torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False,
             top_p=1.0, alg=None, return_nfe=False,
             enable_green_red_policy=False,
             enable_yellow_policy=False,
             green_conf_thresh=0.0,
             yellow_conf_thresh=-0.1,
             enable_probabilistic_gyr_policy=False,
             prob_gyr_tau=0.03,
             yellow_dependency_method='forward',
             yellow_proxy_cfg=None,
             enable_early_stop_when_no_mask=True,
             early_stop_only_when_gr_enabled=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    if alg is None:
        remasking_strategy = remasking
    elif alg == 'entropy':
        remasking_strategy = 'low_confidence'
    elif alg == 'random':
        remasking_strategy = 'random'
    else:
        raise ValueError(f"Unsupported alg: {alg}")

    gr_enabled = _as_bool(enable_green_red_policy, default=False)
    yellow_enabled = bool(gr_enabled and _as_bool(enable_yellow_policy, default=False))
    probabilistic = bool(gr_enabled and _as_bool(enable_probabilistic_gyr_policy, default=False))
    proxy_cfg = _parse_proxy_cfg(yellow_proxy_cfg)
    yellow_dep_method = str(yellow_dependency_method).strip().lower()
    proxy_enabled = bool(yellow_enabled and yellow_dep_method == 'proxy_qk')

    early_stop_enabled = _as_bool(enable_early_stop_when_no_mask, default=True)
    early_stop_only_when_gr = _as_bool(early_stop_only_when_gr_enabled, default=False)

    qk_state = None
    if proxy_enabled:
        # Dream-style: capture Q/K projections directly, do not rely on output_attentions.
        probe_layers = _discover_qk_layers(_unwrap_model_for_hooks(model))
        layer_ids = _resolve_layer_indices(len(probe_layers), proxy_cfg.get("layer_set", "last_only"))
        qk_state = _register_qk_hooks(model, layer_ids)

    nfe = 0

    try:
        for num_block in range(num_blocks):
            block_start = prompt.shape[1] + num_block * block_length
            block_end = prompt.shape[1] + (num_block + 1) * block_length
            block_mask_index = (x[:, block_start:block_end] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

            for i in range(steps):
                mask_index = (x == mask_id)
                eligible_mask_index = mask_index.clone()
                eligible_mask_index[:, block_end:] = False

                if qk_state is not None:
                    _clear_qk_captures(qk_state)

                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    if attention_mask is not None:
                        attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                    else:
                        attention_mask_ = None
                    logits_all = model(x_, attention_mask=attention_mask_).logits
                    logits, un_logits = torch.chunk(logits_all, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x, attention_mask=attention_mask).logits
                nfe += 1

                if logits_eos_inf:
                    logits[:, :, 126081] = -torch.inf

                logits = top_p_filtering(logits, top_p=top_p)
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if confidence_eos_eot_inf:
                    logits_with_noise[:, :, 126081] = -torch.inf
                    logits_with_noise[:, :, 126348] = -torch.inf

                if remasking_strategy == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                elif remasking_strategy == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking_strategy)

                x0_p[:, block_end:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                confidence = torch.where(eligible_mask_index, confidence, torch.tensor(-np.inf, device=confidence.device))

                qk_by_layer = None
                if proxy_enabled:
                    qk_by_layer = _collect_captured_qk(qk_state, batch_size=x.shape[0])
                    if not qk_by_layer:
                        raise RuntimeError("[yellow] proxy_qk requested but Q/K capture is empty; aborting to avoid fallback.")

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    row_mask = eligible_mask_index[j]
                    if int(row_mask.sum().item()) <= 0:
                        continue

                    base_k = int(num_transfer_tokens[j, i].item())
                    committed_global = torch.where(~mask_index[j])[0]

                    if proxy_enabled:
                        row_layers = {lid: (q[j], k[j]) for lid, (q, k) in qk_by_layer.items()}
                        selected_global = _select_transfer_positions_proxy(
                            row_conf=confidence[j],
                            row_mask=row_mask,
                            base_k=base_k,
                            green_conf_thresh=float(green_conf_thresh),
                            yellow_conf_thresh=float(yellow_conf_thresh),
                            probabilistic=probabilistic,
                            prob_tau=float(prob_gyr_tau),
                            row_layers=row_layers,
                            proxy_cfg=proxy_cfg,
                            committed_global=committed_global,
                        )
                    else:
                        selected_global = _select_transfer_positions_standard(
                            row_conf=confidence[j],
                            row_mask=row_mask,
                            base_k=base_k,
                            gr_enabled=gr_enabled,
                            yellow_enabled=yellow_enabled,
                            green_conf_thresh=float(green_conf_thresh),
                            yellow_conf_thresh=float(yellow_conf_thresh),
                            probabilistic=probabilistic,
                            prob_tau=float(prob_gyr_tau),
                        )

                    if selected_global.numel() > 0:
                        transfer_index[j, selected_global] = True

                x[transfer_index] = x0[transfer_index]

                if early_stop_enabled:
                    block_remain = (x[:, block_start:block_end] == mask_id).sum().item()
                    if block_remain == 0:
                        if (not early_stop_only_when_gr) or gr_enabled:
                            break
    finally:
        _remove_qk_hooks(qk_state)

    if return_nfe:
        return x, int(nfe)
    return x


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    # The LLaDA architecture theoretically supports both left-padding and right-padding.
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336

    prompts = [
        "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
        "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
        "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?"
    ]

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    messages = [{"role": "user", "content": prompt} for prompt in prompts]
    prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    out = generate(model, input_ids, attention_mask, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for o in output:
        print(o)
        print('-' * 50)


if __name__ == '__main__':
    main()
