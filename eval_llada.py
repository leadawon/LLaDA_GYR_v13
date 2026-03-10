'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import accelerate
import torch
import re
import json
import time
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from generate import generate


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        cfg=0.,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        temperature=0.0,
        top_p=1.0,
        alg='entropy',
        outp_path=None,
        fp_stats_path=None,
        show_speed=False,
        apply_chat_template=False,
        enable_green_red_policy=False,
        enable_yellow_policy=False,
        green_conf_thresh=0.0,
        yellow_conf_thresh=-0.1,
        enable_probabilistic_gyr_policy=False,
        prob_gyr_tau=0.03,
        yellow_dependency_method='forward',
        yellow_proxy_cfg=None,
        enable_early_stop_when_no_mask=True,
        early_stop_only_when_gr_enabled=False,
        device="cuda",
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer 
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which 
                             returns a True/False judgment used for accuracy calculation. 
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function. 
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality, 
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False 
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
        '''
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, **model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        self._rank = 0
        self._world_size = 1
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(device)

        self.mask_id = int(mask_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.mc_num = int(mc_num)
        self.batch_size = int(batch_size)
        assert self.mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = int(max_length)
        self.is_check_greedy = _as_bool(is_check_greedy, default=True)

        self.cfg = float(cfg)
        self.steps = int(steps)
        self.gen_length = int(gen_length)
        self.block_length = int(block_length)
        self.remasking = remasking
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.alg = alg
        self.outp_path = outp_path
        self.fp_stats_path = fp_stats_path
        self.show_speed = show_speed
        self.use_chat_template = apply_chat_template

        # Dream-style GYR controls
        self.enable_green_red_policy = _as_bool(enable_green_red_policy, default=False)
        self.enable_yellow_policy = _as_bool(enable_yellow_policy, default=False)
        self.green_conf_thresh = float(green_conf_thresh)
        self.yellow_conf_thresh = float(yellow_conf_thresh)
        self.enable_probabilistic_gyr_policy = _as_bool(enable_probabilistic_gyr_policy, default=False)
        self.prob_gyr_tau = float(prob_gyr_tau)
        self.yellow_dependency_method = str(yellow_dependency_method)
        self.yellow_proxy_cfg = yellow_proxy_cfg
        self.enable_early_stop_when_no_mask = _as_bool(enable_early_stop_when_no_mask, default=True)
        self.early_stop_only_when_gr_enabled = _as_bool(early_stop_only_when_gr_enabled, default=False)
    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer_name(self):
        return getattr(self.tokenizer, "name_or_path", "")

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            chat_history, add_generation_prompt=add_generation_prompt, tokenize=False
        )

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
        ds = [{"question": req.args[0], "until": req.args[1]["until"]} for req in requests]
        out = []
        total_nfe = 0
        total_generated_tokens = 0
        start_time = time.time()

        for elem in tqdm(ds, desc="Generating..."):
            prompt_text = elem["question"]
            if self.use_chat_template and "<|im_start|>" not in prompt_text and "<|start_header_id|>" not in prompt_text:
                messages = [{"role": "user", "content": prompt_text}]
                prompt_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            encoded_prompt = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
            prompt = encoded_prompt["input_ids"].to(self.device)
            attention_mask = encoded_prompt.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            stop_tokens = elem["until"]
 
            generated_answer, nfe = generate(
                self.model,
                prompt,
                attention_mask=attention_mask,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=self.temperature,
                top_p=self.top_p,
                alg=self.alg,
                cfg_scale=self.cfg,
                remasking=self.remasking,
                mask_id=self.mask_id,
                enable_green_red_policy=self.enable_green_red_policy,
                enable_yellow_policy=self.enable_yellow_policy,
                green_conf_thresh=self.green_conf_thresh,
                yellow_conf_thresh=self.yellow_conf_thresh,
                enable_probabilistic_gyr_policy=self.enable_probabilistic_gyr_policy,
                prob_gyr_tau=self.prob_gyr_tau,
                yellow_dependency_method=self.yellow_dependency_method,
                yellow_proxy_cfg=self.yellow_proxy_cfg,
                enable_early_stop_when_no_mask=self.enable_early_stop_when_no_mask,
                early_stop_only_when_gr_enabled=self.early_stop_only_when_gr_enabled,
                return_nfe=True,
            )
            total_nfe += nfe
            total_generated_tokens += int(generated_answer.shape[1] - prompt.shape[1])
            
            generated_answer = self.tokenizer.decode(generated_answer[0][prompt.shape[1]:], skip_special_tokens=False)
            for stop_seq in stop_tokens:
                    if stop_seq in generated_answer:
                        generated_answer = generated_answer.split(stop_seq)[0]

            # remove special tokens
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            out.append(generated_answer)

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        total_time = time.time() - start_time
        returned_examples = len(out)
        avg_nfe = (total_nfe / returned_examples) if returned_examples > 0 else 0.0

        if self.show_speed and returned_examples > 0:
            tokens_per_second = (total_generated_tokens / total_time) if total_time > 0 else 0.0
            token_per_nfe = (total_generated_tokens / total_nfe) if total_nfe > 0 else 0.0
            print(f"Time taken: {total_time} seconds")
            print(f"Generated token num: {total_generated_tokens}")
            print(f"Generated token num per second: {tokens_per_second}")
            print(f"Total NFE: {total_nfe}")
            print(f"Token per NFE: {token_per_nfe}")
            print(f"Average NFE: {avg_nfe}")

        if self.outp_path:
            out_path = Path(self.outp_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": datetime.now().isoformat(),
                "total_examples": len(ds),
                "returned_examples": returned_examples,
                "total_time_seconds": total_time,
                "tokens_generated": total_generated_tokens,
                "total_forward_passes": total_nfe,
                "avg_forward_passes": avg_nfe,
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if self.fp_stats_path:
            fp_path = Path(self.fp_stats_path)
            fp_path.parent.mkdir(parents=True, exist_ok=True)
            fp_stats = {
                "timestamp": datetime.now().isoformat(),
                "total_examples": len(ds),
                "returned_examples": returned_examples,
                "total_forward_passes": total_nfe,
                "avg_forward_passes": avg_nfe,
                "total_time_seconds": total_time,
                "tokens_generated": total_generated_tokens,
                "tokens_per_second": (total_generated_tokens / total_time) if total_time > 0 else 0.0,
                "avg_nfe": avg_nfe,
            }
            fp_path.write_text(json.dumps(fp_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        return out


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
    
