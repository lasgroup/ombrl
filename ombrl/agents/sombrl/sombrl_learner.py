"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxrl.agents.sac import temperature
from jaxrl.agents.sac.actor import update as update_actor
from jaxrl.agents.sac.critic import target_update
from jaxrl.agents.sac.critic import update as update_critic
import collections
import copy
from dataclasses import dataclass, asdict
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey
from maxinforl_jax.models.ensemble_model import EnsembleState, DeterministicEnsemble, ProbabilisticEnsemble
from ombrl.utils.pertubation import PerturbationModule

ModelBasedBatch = collections.namedtuple(
    'ModelBasedBatch',
    ['observations', 'actions', 'rewards', 'intrinsic_rewards', 'masks', 'next_observations'])


@dataclass
class SOMBRLConfig:
    reward_model: Optional[Callable] = None
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    ens_lr: float = 3e-4
    ens_wd: float = 0.0
    hidden_dims: Sequence[int] = (256, 256)
    model_hidden_dims: Sequence[int] = (256, 256)
    num_heads: int = 5
    predict_reward: bool = True
    predict_diff: bool = True
    use_log_transform: bool = False
    learn_std: bool = False
    discount: float = 0.99
    tau: float = 0.005
    target_update_period: int = 1
    target_entropy: Optional[float] = None
    backup_entropy: bool = True
    init_temperature: float = 1.0
    init_mean: Optional[np.ndarray] = None
    policy_final_fc_init_scale: float = 1.0
    sample_model: bool = True
    policy_update_period: int = 1
    expl_agent_update_period: int = -1
    agent_update_period: int = 1
    max_gradient_norm: Optional[float] = None
    use_bronet: bool = False
    explore_until: int = 0
    int_rew_weight_start: float = 1.0
    int_rew_weight_end: float = 0.0
    int_rew_weight_decrease_steps: int = -1
    num_imagined_steps: int | optax.Schedule = 1
    actor_critic_updates_per_model_update: int | optax.Schedule = -1
    reset_period: int = 2_500_000
    reset_models: bool = False
    perturb_rate: float = 0.2
    perturb_policy: bool = True
    perturb_model: bool = False


@dataclass
class AgentState:
    config: SOMBRLConfig
    actor: Model
    critic: Model
    target_critic: Model
    temp: Model
    ens_state: EnsembleState
    expl_actor: Model | None
    expl_critic: Model | None
    expl_target_critic: Model | None
    expl_temp: Model
    perturbation_module: PerturbationModule
    steps: int
    rng: PRNGKey


@jax.jit
def jax_symexp(x):
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)


@jax.jit
def get_action(observation: jnp.ndarray, actor: Model, rng: PRNGKey):
    dist = actor(observation)
    act = dist.sample(seed=rng)
    return act


@jax.jit
def concatenate_batches(bs: ModelBasedBatch, concatenated_bs: ModelBasedBatch):
    # Add a new axis and concatenate along that
    return jax.tree_util.tree_map(lambda x, y: jnp.vstack([x[jnp.newaxis], y]), bs, concatenated_bs)


@jax.jit
def mask_batch(old_batch: ModelBasedBatch, new_batch: ModelBasedBatch):
    # Use previous batch if mask == 0.0 else take new batch.
    mask = old_batch.masks
    obs = (1 - mask[..., jnp.newaxis]) * old_batch.observations + mask[..., jnp.newaxis] * new_batch.observations
    actions = (1 - mask[..., jnp.newaxis]) * old_batch.actions + mask[..., jnp.newaxis] * new_batch.actions
    rewards = (1 - mask) * old_batch.rewards + mask * new_batch.rewards
    int_rews = (1 - mask) * old_batch.intrinsic_rewards + mask * new_batch.intrinsic_rewards
    new_masks = (1 - mask) * old_batch.masks + mask * new_batch.masks
    next_obs = (1 - mask[..., jnp.newaxis]) * old_batch.next_observations \
               + mask[..., jnp.newaxis] * new_batch.next_observations
    return ModelBasedBatch(
        observations=obs,
        actions=actions,
        rewards=rewards,
        intrinsic_rewards=int_rews,
        masks=new_masks,
        next_observations=next_obs,
    )


def rollout_learned_model(
        batch: ModelBasedBatch,
        ens: DeterministicEnsemble,
        ens_state: EnsembleState,
        actor: Model,
        predict_rewards: bool,
        predict_diff: bool,
        sample_model: bool,
        key: PRNGKey,
        num_imagined_steps: int = 1,
        reward_model: Optional[Callable] = None,
):
    key, model_sample_key = jax.random.split(key)

    imagined_batch = batch
    # expand dims
    full_batch = jax.tree_util.tree_map(lambda x: x[jnp.newaxis], batch)

    for step in range(num_imagined_steps):
        key, actor_rng, model_sample_key = jax.random.split(key, 3)
        next_actions = get_action(imagined_batch.next_observations, actor, actor_rng)
        # Slide batch by a window of 1
        new_batch = imagined_batch._replace(
            observations=batch.next_observations,
            actions=next_actions,
        )
        # Calculates next observation, reward, and intrinsic reward
        new_batch = get_imagined_batch(
            batch=new_batch,
            ens_state=ens_state,
            ens=ens,
            predict_diff=predict_diff,
            predict_rewards=predict_rewards,
            sample_model=sample_model,
            key=model_sample_key,
            reward_model=reward_model,
        )
        # For the states that have not yet terminated, add the imagined state else keep the last stored state.
        # If mask == 0, we keep the original batch, else we store the imagined batch.
        imagined_batch = mask_batch(imagined_batch, new_batch)
        full_batch = concatenate_batches(imagined_batch, full_batch)
    return full_batch


@functools.partial(jax.jit,
                   static_argnames=('ens',
                                    'predict_rewards',
                                    'predict_diff',
                                    'sample_model',
                                    'reward_model',
                                    ))
def get_imagined_batch(
        batch: ModelBasedBatch,
        ens: DeterministicEnsemble,
        ens_state: EnsembleState,
        predict_rewards: bool,
        predict_diff: bool,
        sample_model: bool,
        key: PRNGKey,
        reward_model: Optional[Callable] = None,
) -> ModelBasedBatch:
    intrinsic_rewards, _ = ens.get_info_gain(input=jnp.concatenate([batch.observations, batch.actions], axis=-1),
                                             state=ens_state,
                                             update_normalizer=False)
    input = jnp.concatenate([batch.observations, batch.actions], axis=-1)
    ens_mean, ens_std = ens(input=input, state=ens_state, denormalize_output=True)
    noise_key, key = jax.random.split(key, 2)
    if sample_model:
        ens_mean = jax.random.choice(key=key, a=ens_mean, axis=0)
        ens_std = jax.random.choice(key=key, a=ens_std, axis=0)
    else:
        ens_mean = jnp.mean(ens_mean, axis=0)
        ens_std = jnp.mean(ens_std, axis=0)
    next_state = ens_mean + jax.random.normal(noise_key, shape=ens_std.shape) * ens_std
    if predict_rewards:
        rew = next_state[..., -1]
        next_state = next_state[..., :-1]
        if predict_diff:
            next_state = next_state + batch.observations
    else:
        if predict_diff:
            next_state = next_state + batch.observations
        rew = jnp.zeros_like(batch.rewards)
        if reward_model is not None:
            rew = jax.vmap(reward_model, in_axes=(0, 0, 0))(
                batch.observations, batch.actions, next_state
            )

    imagined_batch = batch._replace(next_observations=next_state, rewards=rew, intrinsic_rewards=intrinsic_rewards)
    return imagined_batch


@functools.partial(jax.jit, static_argnames=('ens', 'predict_diff', 'predict_rewards'))
def update_ensemble(batch: Batch,
                    ens: DeterministicEnsemble,
                    ens_state: EnsembleState,
                    predict_rewards: bool,
                    predict_diff: bool,
                    ) -> Tuple[EnsembleState, InfoDict, jnp.ndarray]:
    intrinsic_rewards, ens_state = ens.get_info_gain(input=jnp.concatenate([batch.observations,
                                                                            batch.actions], axis=-1),
                                                     state=ens_state,
                                                     update_normalizer=True)

    if predict_diff:
        outputs = batch.next_observations - batch.observations
    else:
        outputs = batch.next_observations
    if predict_rewards:
        outputs = jnp.concatenate([outputs, batch.rewards.reshape(-1, 1)], axis=-1)
    new_ens_state, (loss, mse) = ens.update(
        input=jnp.concatenate([batch.observations, batch.actions], axis=-1),
        output=outputs,
        state=ens_state,
    )
    ens_info = {'ens_nll': loss,
                'ens_mse': mse,
                'ens_inp_mean': ens_state.ensemble_normalizer_state.input_normalizer_state.mean.mean(),
                'ens_inp_std': ens_state.ensemble_normalizer_state.input_normalizer_state.std.mean(),
                'ens_out_mean': ens_state.ensemble_normalizer_state.output_normalizer_state.mean.mean(),
                'ens_out_std': ens_state.ensemble_normalizer_state.output_normalizer_state.std.mean(),
                'ens_info_gain_mean': ens_state.ensemble_normalizer_state.info_gain_normalizer_state.mean.mean(),
                'ens_info_gain_std': ens_state.ensemble_normalizer_state.info_gain_normalizer_state.std.mean(),
                }
    # returns exploration rewards for the exploration critic and policy
    return new_ens_state, ens_info, intrinsic_rewards


@functools.partial(jax.jit,
                   static_argnames=('backup_entropy',
                                    'update_target',
                                    'use_log_transform',
                                    'update_policy',
                                    ))
def _update_jit(
        rng: PRNGKey, actor: Model, critic: Model, target_critic: Model, temp: Model,
        batch: Batch, discount: float, tau: float,
        target_entropy: float, backup_entropy: bool, update_target: bool,
        use_log_transform: bool, update_policy: bool,
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:
    # [x]: Keep update critic with real data?

    rng, critic_key = jax.random.split(rng)

    new_critic, critic_info = update_critic(
        key=critic_key,
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        temp=temp,
        batch=batch,
        discount=discount,
        backup_entropy=backup_entropy,
    )
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    if update_policy:
        rng, key = jax.random.split(rng)
        new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)
        new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                                  target_entropy, use_log_transform=use_log_transform)
    else:
        new_actor, new_temp = actor, temp
        actor_info, alpha_info = {}, {}

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }


class SOMBRLExplorerLearner(object):
    """Initializes the SombrlExplorerLearner.
        
    Has different exploration methods:
            - Mean (greedy) exploration if:
                - int_rew_weight_start = int_rew_weight_end = 0
                - expl_agent_update_period < 0
                - sample_model = False
            - PETS-like exploration if:
                - int_rew_weight_start = int_rew_weight_end = 0
                - expl_agent_update_period < 0
                - sample_model = True
            - SOMBRL exploration if:
                - int_rew_weight_start > 0
                - expl_agent_update_period < 0
            - Unsupervised exploration if:
                - explore_until > 0
    """

    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 *args,
                 **kwargs
                 ):
        self._config = SOMBRLConfig(*args, **kwargs)
        self._action_dim = actions.shape[-1]
        self._ens_output_dim = observations.shape[-1]
        assert self._config.reward_model is not None or self._config.predict_reward, \
            "No reward model specified and neither the reward function is being learned."
        if self._config.predict_reward:
            self._ens_output_dim += 1

        if self._config.target_entropy is None:
            self.target_entropy = -self._action_dim
        else:
            self.target_entropy = self._config.target_entropy
        if isinstance(self._config.num_imagined_steps, int):
            self._num_imagined_steps = optax.constant_schedule(self._config.num_imagined_steps)
        else:
            self._num_imagined_steps = self._config.num_imagined_steps
        if isinstance(self._config.actor_critic_updates_per_model_update, int):
            self._actor_critic_updates_per_model_update = optax.constant_schedule(
                self._config.actor_critic_updates_per_model_update)
        else:
            self._actor_critic_updates_per_model_update = self._config.actor_critic_updates_per_model_update

        rng = jax.random.PRNGKey(seed)
        actor_critic_init_key, rng = jax.random.split(rng)
        self.actor, self.critic, self.target_critic, self.temp = self.init_actor_critic(
            rng=actor_critic_init_key,
            observations=observations,
            actions=actions)
        if self.train_unsupervised_agent:
            unsupervised_actor_critic_init_key, rng = jax.random.split(rng)
            self.expl_actor, self.expl_critic, self.expl_target_critic, self.expl_temp = self.init_actor_critic(
                rng=unsupervised_actor_critic_init_key,
                observations=observations,
                actions=actions)
        else:
            self.expl_actor, self.expl_critic, self.expl_target_critic, self.expl_temp = None, None, None, None

        model_key, rng = jax.random.split(rng, 2)
        self.ens, self.ens_state = self.init_dynamics_model(rng=model_key, observations=observations, actions=actions)
        self.perturbation_module = self.init_perturbation_module()

        self.rng = rng
        self.steps = 0

        if self._config.int_rew_weight_decrease_steps > 0:
            assert self._config.int_rew_weight_start >= 0
            assert self._config.int_rew_weight_end >= 0
            self.int_rew_weight_schedule = optax.linear_schedule(
                init_value=self._config.int_rew_weight_start,
                end_value=self._config.int_rew_weight_end,
                transition_steps=self._config.int_rew_weight_decrease_steps,
            )
        else:
            assert self._config.int_rew_weight_start >= 0
            self.int_rew_weight_schedule = optax.constant_schedule(
                value=self._config.int_rew_weight_start
            )

    def agent_state(self):
        return AgentState(
            config=self._config,
            actor=self.actor,
            critic=self.critic,
            target_critic=self.target_critic,
            temp=self.temp,
            ens_state=self.ens_state,
            expl_actor=self.expl_actor,
            expl_critic=self.expl_critic,
            expl_target_critic=self.expl_target_critic,
            expl_temp=self.expl_temp,
            perturbation_module=self.perturbation_module,
            steps=self.steps,
            rng=self.rng,
        )

    @classmethod
    def load_from_agent_state(cls,
                              seed: int,
                              observations: jnp.ndarray,
                              actions: jnp.ndarray,
                              agent_state: AgentState,
                              load_params: bool = False):
        config = asdict(agent_state.config)
        agent = cls(seed, observations, actions, **config)
        if load_params:
            agent.actor = agent_state.actor
            agent.critic = agent_state.critic
            agent.target_critic = agent_state.target_critic
            agent.temp = agent.temp
            agent.ens_state = agent_state.ens_state
            agent.expl_actor = agent_state.expl_actor
            agent.expl_critic = agent_state.expl_critic
            agent.expl_target_critic = agent_state.expl_target_critic
            agent.expl_temp = agent_state.expl_temp
            agent.perturbation_module = agent_state.perturbation_module
            agent.rng = agent_state.rng
        return agent

    def init_dynamics_model(self, rng: PRNGKey, observations: jnp.ndarray, actions: jnp.ndarray):
        model_optimizer = optax.adamw(learning_rate=self._config.ens_lr, weight_decay=self._config.ens_wd)
        if self._config.max_gradient_norm:
            max_gradient_norm = self._config.max_gradient_norm
            assert max_gradient_norm > 0
            model_optimizer = optax.chain(
                optax.clip_by_global_norm(max_gradient_norm),
                model_optimizer,
            )

        if self._config.learn_std:
            model_type = ProbabilisticEnsemble
        else:
            model_type = DeterministicEnsemble
        ensemble = model_type(
            model_kwargs={'hidden_dims': self._config.model_hidden_dims + (self._ens_output_dim,)},
            optimizer=model_optimizer,
            num_heads=self._config.num_heads,
            # use_entropy_for_int_rew=False,
        )

        ens_state = ensemble.init(key=rng, input=jnp.concatenate([observations, actions], axis=-1))
        return ensemble, ens_state

    def init_actor_critic(self, rng: PRNGKey, observations: jnp.ndarray, actions: jnp.ndarray):
        actor_optimizer = optax.adam(learning_rate=self._config.actor_lr)
        critic_optimizer = optax.adam(learning_rate=self._config.critic_lr)
        temp_optimizer = optax.adam(learning_rate=self._config.temp_lr)
        if self._config.max_gradient_norm:
            max_gradient_norm = self._config.max_gradient_norm
            assert max_gradient_norm > 0
            actor_optimizer = optax.chain(
                optax.clip_by_global_norm(max_gradient_norm),  # Apply gradient clipping
                actor_optimizer  # Apply Adam optimizer
            )
            critic_optimizer = optax.chain(
                optax.clip_by_global_norm(max_gradient_norm),
                critic_optimizer,
            )
            temp_optimizer = optax.chain(
                optax.clip_by_global_norm(max_gradient_norm),
                temp_optimizer,
            )

        actor_def = policies.NormalTanhPolicy(
            self._config.hidden_dims,
            self._action_dim,
            init_mean=self._config.init_mean,
            final_fc_init_scale=self._config.policy_final_fc_init_scale)

        critic_def = critic_net.DoubleCritic(self._config.hidden_dims, use_bronet=self._config.use_bronet)
        actor_key, critic_key, temp_key = jax.random.split(rng, 3)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=actor_optimizer)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=critic_optimizer)
        target_critic = Model.create(critic_def,
                                     inputs=[critic_key, observations, actions])
        temp = Model.create(temperature.Temperature(self._config.init_temperature),
                            inputs=[temp_key],
                            tx=temp_optimizer)
        return actor, critic, target_critic, temp

    def init_perturbation_module(self):
        actor_def = policies.NormalTanhPolicy(
            self._config.hidden_dims,
            self._action_dim,
            init_mean=self._config.init_mean,
            final_fc_init_scale=self._config.policy_final_fc_init_scale)

        critic_def = critic_net.DoubleCritic(self._config.hidden_dims, use_bronet=self._config.use_bronet)

        return PerturbationModule(
            actor_init_fn=actor_def.init,
            critic_init_fn=critic_def.init,
            model_init_fn=self.ens.init,
            actor_init_opt_state=copy.deepcopy(self.actor.opt_state),
            critic_init_opt_state=copy.deepcopy(self.critic.opt_state),
            perturb_rate=self._config.perturb_rate,
            perturbation_freq=self._config.reset_period,
            perturb_policy=self._config.perturb_policy,
            perturb_model=self._config.perturb_model,
        )

    @property
    def train_unsupervised_agent(self):
        return self._config.expl_agent_update_period > 0

    def get_num_imagined_steps_and_update_frequency(self, steps: int) -> Tuple[int, int]:
        assert isinstance(self._num_imagined_steps, Callable)
        num_imagined_steps = int(self._num_imagined_steps(steps))

        assert isinstance(self._actor_critic_updates_per_model_update, Callable)
        actor_critic_updates_per_model_update = int(self._actor_critic_updates_per_model_update(steps))

        if actor_critic_updates_per_model_update < 1:
            actor_critic_updates_per_model_update = num_imagined_steps + 1
        return num_imagined_steps, actor_critic_updates_per_model_update

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> np.ndarray:
        if temperature == 0:
            rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                                   self.actor.params, observations,
                                                   temperature)
        else:
            if self.steps <= self._config.explore_until and self.train_unsupervised_agent:
                rng, actions = policies.sample_actions(self.rng, self.expl_actor.apply_fn,
                                                       self.expl_actor.params, observations,
                                                       temperature)
            else:
                rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                                       self.actor.params, observations,
                                                       temperature)
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def _update_base_agent(self, batch: ModelBasedBatch):
        int_rew_weight = self.int_rew_weight_schedule(self.steps)
        # If doing unsupervised exploration, the base agent acts greedily wrt extrinsic reward.
        if self.train_unsupervised_agent:
            int_rew_weight = jnp.zeros_like(int_rew_weight)
        weight = jnp.maximum(int_rew_weight, 0.0)
        # Define r' = (r + \lambda r_int) / (1 + \lambda) so that its normalized wrt lambda scales
        total_rew = (batch.rewards + weight * batch.intrinsic_rewards) / (1 + weight)
        train_batch = Batch(
            observations=batch.observations,
            actions=batch.actions,
            rewards=total_rew,
            next_observations=batch.next_observations,
            masks=batch.masks,
        )
        out = _update_jit(
            rng=self.rng,
            actor=self.actor,
            critic=self.critic,
            target_critic=self.target_critic,
            temp=self.temp,
            batch=train_batch,
            discount=self._config.discount,
            tau=self._config.tau,
            target_entropy=self.target_entropy,
            backup_entropy=self._config.backup_entropy,
            update_target=self.steps % self._config.target_update_period == 0,
            use_log_transform=self._config.use_log_transform,
            update_policy=self.steps % self._config.policy_update_period == 0)
        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = out
        info = {f'base_agent_{k}': v for k, v in info.items()}
        info = info | {'int_reward_weight': float(weight)}
        return new_rng, new_actor, new_critic, new_target_critic, new_temp, info

    def _update_expl_agent(self, batch: ModelBasedBatch):
        # Override extrinsic with intrinsic reward
        train_batch = Batch(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.intrinsic_rewards,
            next_observations=batch.next_observations,
            masks=batch.masks,
        )
        out = _update_jit(
            rng=self.rng,
            actor=self.expl_actor,
            critic=self.expl_critic,
            target_critic=self.expl_target_critic,
            temp=self.expl_temp,
            batch=train_batch,
            discount=self._config.discount,
            tau=self._config.tau,
            target_entropy=self.target_entropy,
            backup_entropy=self._config.backup_entropy,
            update_target=self.steps % self._config.target_update_period == 0,
            use_log_transform=self._config.use_log_transform,
            update_policy=self.steps % self._config.policy_update_period == 0)
        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = out
        info = {f'expl_agent_{k}': v for k, v in info.items()}
        return new_rng, new_actor, new_critic, new_target_critic, new_temp, info

    def update(self, batch: Batch) -> InfoDict:
        if self._config.reset_models:
            rng, self.rng = jax.random.split(self.rng)
            new_actor, new_critic, new_target_actor, new_target_critic, new_ens_state = self.perturbation_module.perturb(
                actor=self.actor,
                critic=self.critic,
                target_actor=self.actor,
                target_critic=self.target_critic,
                ens_state=self.ens_state,
                observation=batch.observations,
                action=batch.actions,
                rng=rng,
                step=self.steps,
            )
            self.actor = new_actor
            self.critic = new_critic
            self.target_critic = new_target_critic
            self.ens_state = new_ens_state
            if self.train_unsupervised_agent:
                rng, self.rng = jax.random.split(self.rng)
                new_actor, new_critic, new_target_actor, new_target_critic, _ = self.perturbation_module.perturb(
                    actor=self.expl_actor,
                    critic=self.expl_critic,
                    target_actor=self.expl_actor,
                    target_critic=self.expl_target_critic,
                    ens_state=self.ens_state,
                    observation=batch.observations,
                    action=batch.actions,
                    rng=rng,
                    step=self.steps,
                )
                self.expl_actor = new_actor
                self.expl_critic = new_critic
                self.expl_target_critic = new_target_critic
        self.steps += 1
        new_ens_state, ens_info, intrinsic_rewards = update_ensemble(batch=batch,
                                                                     ens=self.ens,
                                                                     ens_state=self.ens_state,
                                                                     predict_diff=self._config.predict_diff,
                                                                     predict_rewards=self._config.predict_reward,
                                                                     )
        self.ens_state = new_ens_state
        # Add intrinsic reward to the batch
        model_batch = ModelBasedBatch(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            intrinsic_rewards=intrinsic_rewards,
            next_observations=batch.next_observations,
            masks=batch.masks,
        )
        rollout_key, self.rng = jax.random.split(self.rng)
        num_imagined_steps, \
            actor_critic_updates_per_model_update = self.get_num_imagined_steps_and_update_frequency(self.steps)
        # TODO: Unclear which actor to use to generated imagined rollouts.
        if self.steps <= self._config.explore_until and self.train_unsupervised_agent:
            actor = self.expl_actor
        else:
            actor = self.actor
        full_batch = rollout_learned_model(batch=model_batch,
                                           ens=self.ens,
                                           ens_state=self.ens_state,
                                           actor=actor,
                                           predict_rewards=self._config.predict_reward,
                                           predict_diff=self._config.predict_diff,
                                           sample_model=self._config.sample_model,
                                           key=rollout_key,
                                           num_imagined_steps=num_imagined_steps,
                                           reward_model=self._config.reward_model,
                                           )  # [Num imagined steps + 1, B, Z]
        info = ens_info

        sample_key, self.rng = jax.random.split(self.rng)
        desired_sample_shape = (actor_critic_updates_per_model_update,)
        sample_indices = jax.random.randint(sample_key, desired_sample_shape,
                                            0, num_imagined_steps + 1)
        summarized_info = collections.defaultdict(list)
        for i in range(actor_critic_updates_per_model_update):
            index = sample_indices[i]
            # extract the batch from imagined batch
            current_batch = jax.tree_util.tree_map(lambda x: x[index], full_batch)
            agent_info = {}
            if self.steps % self._config.agent_update_period == 0:
                out = self._update_base_agent(batch=current_batch)
                new_rng, new_actor, new_critic, new_target_critic, new_temp, agent_info = out
                self.rng = new_rng
                self.actor = new_actor
                self.critic = new_critic
                self.target_critic = new_target_critic
                self.temp = new_temp

            if self.train_unsupervised_agent:
                if self.steps % self._config.expl_agent_update_period == 0:
                    out = self._update_expl_agent(batch=current_batch)
                    new_rng, new_actor, new_critic, new_target_critic, new_temp, expl_agent_info = out

                    agent_info = agent_info | expl_agent_info
                    self.rng = new_rng
                    self.expl_actor = new_actor
                    self.expl_critic = new_critic
                    self.expl_target_critic = new_target_critic
                    self.expl_temp = new_temp
            for key, value in agent_info.items():
                summarized_info[key].append(value)
        summarized_info = {key: np.stack(value).mean() for key, value in summarized_info.items()}
        info = info | summarized_info

        return info | {'policy_phase': int(self.steps <= self._config.explore_until
                                           and self.train_unsupervised_agent)}
