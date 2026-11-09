"""HORA-specific shared features, normalization and PPO losses on RSL-RL.

HORA formulas follow the MIT release, commit 410d958. RSL-RL provides
rollout storage, timeout handling, distributions and checkpoint integration.
"""

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import reduce_gradients_in_buckets

from .normalization import HoraRunningMeanStd


def hora_bounds_loss(mean):
  # Preserve the published code's clamp_max, including its nonzero value
  # inside the interval. This intentionally follows code, not an inferred loss.
  high = torch.clamp_max(mean - 1.1, 0).square()
  low = torch.clamp_max(-mean + 1.1, 0).square()
  return (high + low).sum(-1).mean()


class HoraPPO(PPO):
  def __init__(self, actor, critic, storage, bounds_loss_coef=1e-4, **kwargs):
    critic.env_encoder = actor.env_encoder
    critic.obs_normalizer = actor.obs_normalizer
    for index in range(len(actor.mlp) - 1):
      critic.mlp[index] = actor.mlp[index]
    super().__init__(actor, critic, storage, **kwargs)
    self.bounds_loss_coef = bounds_loss_coef
    self.value_normalizer = HoraRunningMeanStd(1).to(self.device).eval()

  def act(self, obs):
    self.actor.obs_normalizer.eval()
    actions = super().act(obs)
    self.transition.values = self.value_normalizer(
      self.transition.values, unnorm=True
    ).detach()
    return actions

  def compute_returns(self, obs):
    self.actor.obs_normalizer.eval()
    st = self.storage
    last_values = self.value_normalizer(self.critic(obs).detach(), unnorm=True)
    advantage = 0
    for step in reversed(range(st.num_transitions_per_env)):
      next_values = (
        last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
      )
      nonterminal = 1 - st.dones[step].float()
      delta = (
        st.rewards[step] + nonterminal * self.gamma * next_values - st.values[step]
      )
      advantage = delta + nonterminal * self.gamma * self.lam * advantage
      st.returns[step] = advantage + st.values[step]
    st.advantages = st.returns - st.values
    st.advantages = (st.advantages - st.advantages.mean()) / (
      st.advantages.std() + 1e-8
    )
    # Original code updates the same statistics on values, then returns.
    self.value_normalizer.train()
    st.values.copy_(
      self.value_normalizer(st.values.flatten(0, 1)).reshape_as(st.values)
    )
    st.returns.copy_(
      self.value_normalizer(st.returns.flatten(0, 1)).reshape_as(st.returns)
    )
    self.value_normalizer.eval()

  def process_env_step(self, obs, rewards, dones, extras):
    # Match HORA's rollout shaping; environment rewards/logs retain raw units.
    return super().process_env_step(obs, rewards * 0.01, dones, extras)

  def update(self):
    losses = dict(value=0.0, surrogate=0.0, entropy=0.0, bounds=0.0)
    generator = self.hora_batches()
    epoch_kl = []
    unique_parameters = list(
      dict.fromkeys(list(self.actor.parameters()) + list(self.critic.parameters()))
    )
    for number, batch in enumerate(generator):
      self.actor.obs_normalizer.train()
      self.actor(batch.observations, stochastic_output=True)
      log_prob = self.actor.get_output_log_prob(batch.actions)
      self.actor.obs_normalizer.eval()
      values = self.critic(batch.observations)
      ratio = torch.exp(log_prob - batch.old_actions_log_prob.squeeze(-1))
      advantages = batch.advantages.squeeze(-1)
      surrogate = torch.maximum(
        -advantages * ratio,
        -advantages * ratio.clamp(1 - self.clip_param, 1 + self.clip_param),
      ).mean()
      clipped = batch.values + (values - batch.values).clamp(
        -self.clip_param, self.clip_param
      )
      value_loss = torch.maximum(
        (values - batch.returns).square(), (clipped - batch.returns).square()
      ).mean()
      entropy = self.actor.output_entropy.mean()
      bounds = hora_bounds_loss(self.actor.output_mean)
      total = (
        surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy
      )
      total += self.bounds_loss_coef * bounds
      self.optimizer.zero_grad()
      total.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      torch.nn.utils.clip_grad_norm_(unique_parameters, self.max_grad_norm)
      self.optimizer.step()
      for name, value in (
        ("value", value_loss),
        ("surrogate", surrogate),
        ("entropy", entropy),
        ("bounds", bounds),
      ):
        losses[name] += float(value.detach())
      with torch.no_grad():
        # Same orientation and epsilon as HORA policy_kl.
        mean, std = self.actor.output_mean.detach(), self.actor.output_std.detach()
        old_mean, old_std = batch.old_distribution_params
        kl = (
          (
            torch.log(old_std / std + 1e-5)
            + (std.square() + (old_mean - mean).square())
            / (2 * (old_std.square() + 1e-5))
            - 0.5
          )
          .sum(-1)
          .mean()
        )
        epoch_kl.append(kl)
        old_mean.copy_(mean)
        old_std.copy_(std)
      if (number + 1) % self.num_mini_batches == 0:
        kl_mean = torch.stack(epoch_kl).mean()
        if self.is_multi_gpu:
          torch.distributed.all_reduce(kl_mean)
          kl_mean /= self.gpu_world_size
        if kl_mean > self.desired_kl * 2:
          self.learning_rate = max(1e-6, self.learning_rate / 1.5)
        elif kl_mean < self.desired_kl / 2:
          self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        for group in self.optimizer.param_groups:
          group["lr"] = self.learning_rate
        epoch_kl.clear()
    self.actor.obs_normalizer.eval()
    self.storage.clear()
    updates = self.num_mini_batches * self.num_learning_epochs
    return {name: value / updates for name, value in losses.items()}

  def hora_batches(self):
    # HORA uses env-major, contiguous minibatches and refreshes old mu/sigma
    # after each minibatch; its action log probabilities remain unchanged.
    st = self.storage

    def flatten(value):
      return value.transpose(0, 1).flatten(0, 1)

    fields = {
      name: flatten(getattr(st, name))
      for name in ("observations", "actions", "values", "advantages", "returns")
    }
    log_prob = flatten(st.actions_log_prob)
    assert st.distribution_params is not None
    params = tuple(flatten(p) for p in st.distribution_params)
    size = st.num_envs * st.num_transitions_per_env
    for _ in range(self.num_learning_epochs):
      for index in range(self.num_mini_batches):
        section = slice(
          index * size // self.num_mini_batches,
          (index + 1) * size // self.num_mini_batches,
        )
        yield RolloutStorage.Batch(
          **{name: value[section] for name, value in fields.items()},
          old_actions_log_prob=log_prob[section],
          old_distribution_params=tuple(p[section] for p in params),
        )

  def save(self):
    state = super().save()
    state["hora_value_normalizer"] = self.value_normalizer.state_dict()
    state["hora_shared_features"] = True
    return state

  def reduce_parameters(self):
    parameters = dict.fromkeys(
      list(self.actor.parameters()) + list(self.critic.parameters())
    )
    reduce_gradients_in_buckets(
      parameters, self.gpu_world_size, self.grad_reduce_bucket_mb
    )

  def load(self, loaded_dict, load_cfg, strict=True):
    state = loaded_dict
    if state.get("hora_shared_features") is not True:
      raise ValueError("This checkpoint predates the shared HORA architecture")
    result = super().load(state, load_cfg, strict)
    if load_cfg is None or load_cfg.get("optimizer", False):
      self.learning_rate = self.optimizer.param_groups[0]["lr"]
    self.value_normalizer.load_state_dict(state["hora_value_normalizer"], strict=strict)
    self.value_normalizer.eval()
    return result
