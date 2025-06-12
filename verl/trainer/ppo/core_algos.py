# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict
from typing import Optional, Tuple
import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num -
                                                        1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, eos_mask: torch.Tensor,
                                                  gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++. 
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * eos_mask[:, t]

        advantages = verl_F.masked_whiten(returns, eos_mask)
        advantages = advantages * eos_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor,
                                    eos_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward 
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        returns = (token_level_rewards * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def _random_select(idxs: torch.Tensor, num: int) -> torch.Tensor:
    """
    Uniformly pick `num` elements from 1-D index tensor `idxs`.
    Returns an empty tensor when either `num` or `idxs.numel()` is 0.
    """
    if num <= 0 or idxs.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=idxs.device)
    num = min(num, idxs.numel())
    perm = torch.randperm(idxs.numel(), device=idxs.device)
    return idxs[perm[:num]]


def build_top_p_mask(
    logits: torch.Tensor,
    labels: torch.LongTensor,
    top_p: float
) -> torch.BoolTensor:
    """
    Constructs a [B, T, V] Boolean mask that is True for those tokens which:
      • belong to the “nucleus” (top‐p) set of new‐policy probabilities at each (b,t),
      • except the true label at (b,t), which is forced to False.

    Args:
      logits: [B, T, V]  — the logits from the new policy at each position.
      labels: [B, T]     — the true token‐IDs (0 <= labels[b,t] < V).
      top_p: float in (0, 1)  — the cumulative‐probability threshold.

    Returns:
      mask: a [B, T, V] Boolean tensor where mask[b,t,i] == True iff
            token i is in the smallest set of tokens whose sorted‐probabilities sum ≥ top_p,
            and i != labels[b,t].
    """
    B, T, V = logits.shape

    # 1) Compute new‐policy probabilities
    new_log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
    new_probs     = torch.exp(new_log_probs)       # [B, T, V]

    # 2) Sort each [B, T] slice of new_probs descending, keep indices
    sorted_probs, sorted_indices = torch.sort(new_probs, dim=-1, descending=True)  # [B, T, V]
    # 3) Cumulative sum over the V‐axis
    cum_probs = torch.cumsum(sorted_probs, dim=-1)  # [B, T, V]
    # 4) Mask all slots whose cumulative sum ≤ top_p
    mask_cum = cum_probs <= top_p                   # [B, T, V] (bool)

    # 5) We still need to include the *first* slot k* where cum_probs ≥ top_p
    k_star = mask_cum.sum(dim=-1, keepdim=True)           # [B, T, 1], int64
    k_star_clamped = torch.clamp(k_star, max=V-1)          # just in case top_p≥1
    one_more_mask = torch.zeros_like(mask_cum, dtype=torch.bool)  # [B, T, V]

    # 6) Scatter a True at (b, t, k_star_clamped[b,t])
    b_idx = torch.arange(B, device=logits.device).view(B, 1).expand(B, T)  # [B, T]
    t_idx = torch.arange(T, device=logits.device).view(1, T).expand(B, T)  # [B, T]
    v_idx = k_star_clamped.view(B, T)                                       # [B, T]
    one_more_mask[b_idx, t_idx, v_idx] = True

    # 7) Final nucleus mask = everything with cum_probs ≤ top_p OR that one_more slot
    top_p_mask = mask_cum | one_more_mask  # [B, T, V], bool

    # 8) Build a one‐hot mask for the true labels so we can exclude them
    true_label_mask = F.one_hot(labels, num_classes=V).bool()  # [B, T, V]

    # 9) “Nucleus minus the true label”
    final_mask = top_p_mask & (~true_label_mask)  # [B, T, V], bool

    return final_mask


def compute_policy_loss(
    config,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    cliprange: Optional[float] = None,
    cliprange_low: Optional[float] = None,
    cliprange_high: Optional[float] = None,
    use_token_level_loss: bool = False,
    ref_log_prob = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PPO/GRPO policy-gradient loss with optional covariance-based refinements.

    Parameters
    ----------
    old_log_prob, log_prob, advantages, eos_mask : torch.Tensor
        Same as before.
    method : {"clip_cov", "kl_cov", None}
        - "clip_cov": randomly DETACH a subset of tokens whose covariance lies
          between `cov_lb` and `cov_ub`.  
        - "kl_cov"  : add the KL penalty *only* to the `select_ratio` tokens
          that have the largest covariance.  
        - None      : behave exactly like vanilla PPO.
    select_ratio : float
        Fraction ∈ (0,1].  How many tokens to act on (relative to total #tokens).
    cov_lb, cov_ub : float
        Bounds used when `method=="clip_cov"`.
    kl_coef : float
        Multiplier for the KL penalty when `method=="kl_cov"`.

    Returns
    -------
    pg_loss : torch.Tensor               # scalar
    pg_clipfrac : torch.Tensor           # scalar
    ppo_kl : torch.Tensor                # scalar
    """

    contrastive_coeff = config.get('contrastive_kl', {}).get('coef', 0.0)
    contrastive_kl = torch.tensor(0.0, device=log_prob.device)
    method = config.get('cov_reg', {}).get('method', None) if config.get('cov_reg', {}).get('enable', False) else None
    if (method not in {"clip", "kl"}) and (contrastive_coeff == 0.):
        ret = compute_policy_loss_no_cov(
            old_log_prob,
            log_prob,
            advantages,
            eos_mask,
            cliprange=cliprange,
            cliprange_low=cliprange_low,
            cliprange_high=cliprange_high,
            use_token_level_loss=use_token_level_loss,
        )
        assert len(ret) == 3
        return ret[0], ret[1], ret[2], contrastive_kl

    # ----------------------------------------------------------------------- #
    # basic PPO machinery (unchanged)                                         #
    # ----------------------------------------------------------------------- #
    seq_len_per_sample = torch.clamp(torch.sum(eos_mask, dim=1), min=1.0)
    negative_approx_kl = log_prob - old_log_prob          # (bs, T)
    ratio = torch.exp(negative_approx_kl)                 # (bs, T)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses1 = -advantages * ratio                      # (bs, T)

    # choose clip ranges
    cliprange_low = cliprange if cliprange_low is None else cliprange_low
    cliprange_high = cliprange if cliprange_high is None else cliprange_high

    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )

    # ----------------------------------------------------------------------- #
    # NEW: covariance between log-probs and advantages                        #
    # ----------------------------------------------------------------------- #
    if method in {"clip", "kl"}:
        # (bs, T) – center each series independently to stay true to diff
        covs = (log_prob - log_prob.mean()) * (advantages - advantages.mean())
        total_tokens = len(pg_losses1)
        select_num = max(int(config.cov_reg.select_ratio * total_tokens), 0)

    # ----------------------------------------------------------------------- #
    # Method-specific tweaks                                                  #
    # ----------------------------------------------------------------------- #
    if method == "clip":
        # RANDOMLY DETACH tokens whose covariance magnitude lies in (cov_lb, cov_ub)
        mask = (covs > config.cov_reg.clip_lb) & (covs < config.cov_reg.clip_ub) & eos_mask.bool()
        candidate_idxs = mask.flatten().nonzero(as_tuple=False).squeeze(-1)
        clip_idxs = _random_select(candidate_idxs, select_num)

        if clip_idxs.numel() > 0:  # perform the in-place detaching
            # work on flattened views so the same indices align
            pg1_flat, pg2_flat = pg_losses1.flatten(), pg_losses2.flatten()
            pg1_flat[clip_idxs] = pg1_flat[clip_idxs].detach()
            pg2_flat[clip_idxs] = pg2_flat[clip_idxs].detach()
            pg_losses1 = pg1_flat.view_as(pg_losses1)
            pg_losses2 = pg2_flat.view_as(pg_losses2)

        # standard PPO clipping after partial detaching
        pg_losses = torch.maximum(pg_losses1, pg_losses2)

    elif method == "kl":
        # APPEND KL penalty only to the top-covariance tokens
        kl_penalty = negative_approx_kl.abs()
        # pick global top-k (flattened) for simplicity
        k = min(select_num, covs.numel())
        if k > 0:
            topk_idx = torch.topk(covs.flatten(), k=k, largest=True).indices
            pg1_flat = pg_losses1.flatten()
            pg1_flat[topk_idx] += kl_penalty.flatten()[topk_idx]
            pg_losses1 = pg1_flat.view_as(pg_losses1)

        pg_losses = pg_losses1                             # no clipping here
        pg_losses2 = None                                  # unused—but keep for clipfrac calc

    else:  # vanilla PPO
        assert method is None, method
        pg_losses = torch.maximum(pg_losses1, pg_losses2)

    if contrastive_coeff > 0:
        assert ref_log_prob is not None, "ref_log_prob is required for contrastive loss"
        kl = log_prob - ref_log_prob

        pos_kl_mask = kl > 0
        neg_kl_mask = kl < 0
        correct_kl_mask = torch.logical_and(pos_kl_mask, advantages > 0)
        incorrect_kl_mask = torch.logical_and(neg_kl_mask, advantages < 0)
        pos_kl_penalty = torch.zeros_like(kl)
        neg_kl_penalty = torch.zeros_like(kl)
        pos_kl_penalty[correct_kl_mask] = kl[correct_kl_mask]
        neg_kl_penalty[incorrect_kl_mask] = -kl[incorrect_kl_mask]
        kl_penalty = pos_kl_penalty # + neg_kl_penalty
        #contrastive_kl = verl_F.masked_mean(pos_kl_penalty.abs() + neg_kl_penalty.abs(), eos_mask)
        contrastive_kl = verl_F.masked_mean(neg_kl_penalty.abs(), eos_mask)
        pg_losses = pg_losses + (contrastive_coeff * kl_penalty)

    # ----------------------------------------------------------------------- #
    # aggregate loss & bookkeeping                                            #
    # ----------------------------------------------------------------------- #
    if use_token_level_loss:
        pg_loss = verl_F.masked_mean(pg_losses, eos_mask)
    else:
        # sample-level average, identical to your original implementation
        pg_loss = torch.sum(pg_losses * eos_mask, dim=1) / seq_len_per_sample
        pg_loss = torch.mean(pg_loss)

    # clipfrac (meaningful only when we *have* a clipped version)
    if pg_losses2 is not None:
        pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), eos_mask)
    else:
        pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)

    return pg_loss, pg_clipfrac, ppo_kl, contrastive_kl


def compute_policy_loss_no_cov(
                        old_log_prob,
                        log_prob,
                        advantages,
                        eos_mask,
                        cliprange=None,
                        cliprange_low=None,
                        cliprange_high=None,
                        use_token_level_loss=False,
                        ):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        use_token_level_loss: (bool)
            Whether to use token level loss
    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
    """
    seq_len_per_sample = torch.clamp(torch.sum(eos_mask, dim=1), min=1.0)
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low,
                                           1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    pg_losses = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)

    if use_token_level_loss:
        pg_loss = verl_F.masked_mean(pg_losses, eos_mask)
    else:
        pg_loss = torch.sum(pg_losses * eos_mask, dim=1) / seq_len_per_sample
        pg_loss = torch.mean(pg_loss)

    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
