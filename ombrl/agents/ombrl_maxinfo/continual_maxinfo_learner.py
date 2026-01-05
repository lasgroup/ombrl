import functools
import collections
import copy
from typing import Optional, Sequence, Tuple, Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.sac import temperature
from maxinforl_jax.agents.maxinfosac.actor import update as update_actor
from maxinforl_jax.agents.maxinfosac.critic import target_update
from maxinforl_jax.agents.maxinfosac.critic import update as update_critic
from jaxrl.agents.sac.temperature import update as update_temp

from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey

from maxinforl_jax.models import EnsembleState, DeterministicEnsemble, ProbabilisticEnsemble
from ombrl.utils.pertubation import PolicyPerturbationModule, CriticPerturbationModule, \
    ModelPerturbationModule


@dataclass
class ContinualMaxInfoConfig:
    reward_model: Optional[Callable] = None
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    dyn_ent_lr: float = 3e-4
    dyn_wd: float = 0.0
    ens_lr: float = 3e-4
    ens_wd: float = 0.0
    hidden_dims: Sequence[int] = (256, 256)
    model_hidden_dims: Sequence[int] = (256, 256)
    num_heads: int = 5
    predict_reward: bool = True
    predict_diff: bool = True
    use_log_transform: bool = True
    learn_std: bool = False
    discount: float = 0.99
    tau: float = 0.005
    target_update_period: int = 1
    target_entropy: Optional[float] = None
    backup_entropy: bool = True
    init_temperature: float = 1.0
    init_temperature_dyn_entropy: float = 1.0
    init_mean: Optional[np.ndarray] = None
    policy_final_fc_init_scale: float = 1.0
    sample_model: bool = True
    policy_update_period: int = 1
    num_imagined_steps: int | optax.Schedule = 1
    actor_critic_updates_per_model_update: int | optax.Schedule = -1
    max_gradient_norm: Optional[float] = None
    use_bronet: bool = False
    reset_models: bool = False
    perturb_policy: bool = True
    perturb_critic: bool = True
    perturb_model: bool = True
    policy_perturb_rate: float = 0.2
    critic_perturb_rate: float = 0.2
    model_perturb_rate: float = 0.2
    policy_reset_period: int = 5
    critic_reset_period: int = 5
    model_reset_period: int = 5
    pseudo_ct: bool = False
    dt: float = None
    action_repeat: int = None


@dataclass
class AgentState:
    config: ContinualMaxInfoConfig
    actor: Model
    target_actor: Model
    critic: Model
    target_critic: Model
    temp: Model
    dyn_ent_temp: Model
    ens_state: EnsembleState
    policy_perturb_module: PolicyPerturbationModule
    critic_perturb_module: CriticPerturbationModule
    model_perturb_module: ModelPerturbationModule
    rng: PRNGKey
    step: int


@jax.jit
def concatenate_batches(bs: Batch, concatenated_bs: Batch):
    # Add a new axis and concatenate along that
    return jax.tree_util.tree_map(lambda x, y: jnp.vstack([x[jnp.newaxis], y]), bs, concatenated_bs)


@jax.jit
def mask_batch(old_batch: Batch, new_batch: Batch):
    # Use previous batch if mask == 0.0 else take new batch.
    mask = old_batch.masks
    obs = (1 - mask[..., jnp.newaxis]) * old_batch.observations + mask[..., jnp.newaxis] * new_batch.observations
    actions = (1 - mask[..., jnp.newaxis]) * old_batch.actions + mask[..., jnp.newaxis] * new_batch.actions
    rewards = (1 - mask) * old_batch.rewards + mask * new_batch.rewards
    new_masks = (1 - mask) * old_batch.masks + mask * new_batch.masks
    next_obs = (1 - mask[..., jnp.newaxis]) * old_batch.next_observations \
               + mask[..., jnp.newaxis] * new_batch.next_observations
    return Batch(
        observations=obs,
        actions=actions,
        rewards=rewards,
        masks=new_masks,
        next_observations=next_obs,
    )

@functools.partial(jax.jit, static_argnames=('ens', 'predict_rewards', 'predict_diff', 'sample_model', 'num_imagined_steps', 'dt', 'action_repeat'))
def rollout_learned_model(
        batch: Batch,
        ens: DeterministicEnsemble,
        ens_state: EnsembleState,
        actor: Model,
        predict_rewards: bool,
        predict_diff: bool,
        sample_model: bool,
        key: PRNGKey,
        num_imagined_steps: int = 1,
        dt: float = None,
        action_repeat: int = 1,
        reward_model: Optional[Callable] = None,
) -> Batch:
    
    imagined_batch = batch
    # expand dims
    full_batch = jax.tree_util.tree_map(lambda x: x[jnp.newaxis], batch)

    for _ in range(num_imagined_steps):
        key, actor_rng, model_rng = jax.random.split(key, 3)
        
        # get actions for the current predicted state
        obs = imagined_batch.next_observations
        _, actions = policies.sample_actions(actor_rng, actor.apply_fn, actor.params, obs)
        
        # predict dynamics
        input_data = jnp.concatenate([obs, actions], axis=-1)
        ens_mean, ens_std = ens(input=input_data, state=ens_state, denormalize_output=True)
        
        if sample_model:
            model_key, noise_key = jax.random.split(model_rng)
            ens_mean = jax.random.choice(key=model_key, a=ens_mean, axis=0)
            ens_std = jax.random.choice(key=model_key, a=ens_std, axis=0)
        else:
            noise_key = model_rng
            ens_mean = jnp.mean(ens_mean, axis=0)
            ens_std = jnp.mean(ens_std, axis=0)

        # handle reward prediction
        if predict_rewards:
            ens_rew_pred = ens_mean[..., -1]
            next_state_delta = ens_mean[..., :-1]
            ens_std_state = ens_std[..., :-1]
        else:
            ens_rew_pred = jnp.zeros_like(batch.rewards)
            next_state_delta = ens_mean
            ens_std_state = ens_std

        # apply state transition logic
        delta = next_state_delta + jax.random.normal(noise_key, shape=ens_std_state.shape) * ens_std_state
        if predict_diff:
            if dt is not None:
                delta = delta * dt * action_repeat
            next_state = delta + obs
        else:
            next_state = delta

        # finalize reward prediction
        if predict_rewards:
            rew = ens_rew_pred
        elif reward_model is not None:
            # vmap across the batch dimension
            rew = jax.vmap(reward_model, in_axes=(0, 0, 0))(obs, actions, next_state)
        else:
            rew = jnp.zeros_like(batch.rewards)

        # Create the raw new transition
        new_batch_step = Batch(
            observations=obs, 
            actions=actions, 
            rewards=rew, 
            masks=imagined_batch.masks, 
            next_observations=next_state
        )
        
        # For the states that have not yet terminated, add the imagined state else keep the last stored state.
        # If mask == 0, we keep the original batch, else we store the imagined batch.
        imagined_batch = mask_batch(imagined_batch, new_batch_step)
        full_batch = concatenate_batches(imagined_batch, full_batch)
        
    return full_batch

@functools.partial(jax.jit, static_argnames=('ens', 'predict_diff', 'predict_rewards', 'dt', 'action_repeat'))
def update_ensemble_jit(batch: Batch, 
                        ens: DeterministicEnsemble, 
                        ens_state: EnsembleState, 
                        predict_rewards: bool,
                        predict_diff: bool,
                        dt: float = None,
                        action_repeat: int = 1,
                        ):
    """Ensemble update logic extracted to ensure it only runs on REAL data."""
    if predict_diff:
        outputs = batch.next_observations - batch.observations
        if dt is not None:
            outputs = outputs / (dt * action_repeat)
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
    return new_ens_state, ens_info

@functools.partial(jax.jit, static_argnames=('ens', 'backup_entropy', 'update_target', 'use_log_transform', 'update_policy'))
def update_ac_jit(
        rng: PRNGKey, actor: Model, critic: Model, target_actor: Model, target_critic: Model, temp: Model, # type: ignore
        dyn_entropy_temp: Model, ens: DeterministicEnsemble, ens_state: EnsembleState,
        batch: Batch, discount: float, tau: float,
        target_entropy: float, backup_entropy: bool, update_target: bool,
        use_log_transform: bool, update_policy: bool,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, Model, EnsembleState, InfoDict]: 
    """Actor-Critic update logic for use with IMAGINED data."""
    rng, key = jax.random.split(rng)
    
    # Update critic
    new_critic, ens_state, critic_info = update_critic(
        key=key, actor=actor, critic=critic, target_critic=target_critic,
        temp=temp, dyn_entropy_temp=dyn_entropy_temp, ens=ens, ens_state=ens_state,
        batch=batch, discount=discount, backup_entropy=backup_entropy,
    )

    new_target_critic = target_update(new_critic, target_critic, tau) if update_target else target_critic

    # Update policy and temperature
    if update_policy:
        rng, key = jax.random.split(rng)
        new_actor, new_ens_state, actor_info = update_actor(
            key=key, actor=actor, target_actor=target_actor, critic=new_critic,
            temp=temp, dyn_entropy_temp=dyn_entropy_temp, ens=ens, ens_state=ens_state, batch=batch
        )
        new_target_actor = target_update(new_actor, target_actor, tau) if update_target else target_actor
        new_temp, alpha_info = update_temp(temp, actor_info['entropy'], target_entropy, use_log_transform=use_log_transform)
        new_dyn_entropy_temp, dyn_ent_info = update_temp(dyn_entropy_temp, actor_info['info_gain'], actor_info['target_info_gain'], use_log_transform=use_log_transform)
    else:
        new_actor, new_target_actor, new_temp, new_dyn_entropy_temp = actor, target_actor, temp, dyn_entropy_temp
        actor_info, alpha_info, dyn_ent_info = {}, {}, {}

    return rng, \
        new_actor, \
        new_critic, \
        new_target_actor, \
        new_target_critic, \
        new_temp, \
        new_dyn_entropy_temp, \
        new_ens_state, {
            **critic_info,
            **actor_info,
            **alpha_info,
            **dyn_ent_info
    }

class ContinualMaxInfoLearner(object):
    def __init__(self, seed: int, observations: jnp.ndarray, actions: jnp.ndarray, **kwargs):
        self.config: ContinualMaxInfoConfig = ContinualMaxInfoConfig(**kwargs)
        if self.config.target_entropy is None: self.config.target_entropy = -actions.shape[-1]

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key, dyn_key, model_key = jax.random.split(rng, 6)
        
        # Networks and Optimizers
        actor_def = policies.NormalTanhPolicy(self.config.hidden_dims, actions.shape[-1], init_mean=self.config.init_mean, final_fc_init_scale=self.config.policy_final_fc_init_scale)
        critic_def = critic_net.DoubleCritic(self.config.hidden_dims, use_bronet=self.config.use_bronet)
        
        def make_opt(lr, wd=0.0):
            opt = optax.adamw(lr, weight_decay=wd) if wd > 0 else optax.adam(lr)
            return optax.chain(optax.clip_by_global_norm(self.config.max_gradient_norm), opt) if self.config.max_gradient_norm else opt

        self.actor = Model.create(actor_def, inputs=[actor_key, observations], tx=make_opt(self.config.actor_lr))
        self.target_actor = Model.create(actor_def, inputs=[actor_key, observations])
        self.critic = Model.create(critic_def, inputs=[critic_key, observations, actions], tx=make_opt(self.config.critic_lr))
        self.target_critic = Model.create(critic_def, inputs=[critic_key, observations, actions])
        self.temp = Model.create(temperature.Temperature(self.config.init_temperature), inputs=[temp_key], tx=make_opt(self.config.temp_lr))
        self.dyn_ent_temp = Model.create(temperature.Temperature(self.config.init_temperature_dyn_entropy), inputs=[dyn_key], tx=make_opt(self.config.dyn_ent_lr, self.config.dyn_wd))
        
        # Dynamics
        out_dim = observations.shape[-1]
        assert self.config.reward_model is not None or self.config.predict_reward, \
            "No reward model specified and neither the reward function is being learned."
        if self.config.predict_reward:
            out_dim += 1
        if self.config.dt is not None and self.config.predict_diff:
            assert self.config.pseudo_ct, "dt specified but pseudo continuous time flag not set."
        model_type = ProbabilisticEnsemble if self.config.learn_std else DeterministicEnsemble
        self.ensemble = model_type(model_kwargs={'hidden_dims': self.config.model_hidden_dims + (out_dim,)}, 
                                   optimizer=make_opt(self.config.ens_lr, self.config.ens_wd), 
                                   num_heads=self.config.num_heads
                                   )
        self.ens_state = self.ensemble.init(key=model_key, input=jnp.concatenate([observations, actions], axis=-1))

        # Perturbation Modules
        self.policy_perturb_module = PolicyPerturbationModule(
            actor_init_fn=actor_def.init,
            actor_init_opt_state=copy.deepcopy(self.actor.opt_state),
            perturb_rate=self.config.policy_perturb_rate,
            perturbation_freq=self.config.policy_reset_period,
            perturb_policy=self.config.perturb_policy
        )
        self.critic_perturb_module = CriticPerturbationModule(
            critic_init_fn=critic_def.init,
            critic_init_opt_state=copy.deepcopy(self.critic.opt_state),
            perturb_rate=self.config.critic_perturb_rate,
            perturbation_freq=self.config.critic_reset_period,
            perturb_critic=self.config.perturb_critic
        )
        self.model_perturb_module = ModelPerturbationModule(
            model_init_fn=self.ensemble.init,
            perturb_rate=self.config.model_perturb_rate,
            perturbation_freq=self.config.model_reset_period,
            perturb_model=self.config.perturb_model,
        )
        self.rng = rng
        self.step = 0

    @property
    def agent_state(self) -> AgentState:
        return AgentState(
            config=self.config,
            actor=self.actor,
            target_actor=self.target_actor,
            critic=self.critic,
            target_critic=self.target_critic,
            temp=self.temp,
            dyn_ent_temp=self.dyn_ent_temp,
            ens_state=self.ens_state,
            policy_perturb_module=self.policy_perturb_module,
            critic_perturb_module=self.critic_perturb_module,
            model_perturb_module=self.model_perturb_module,
            rng=self.rng,
            step=self.step,
        )


    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> np.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch, episode_idx: int = None) -> InfoDict:
        # Pertubation
        if self.config.reset_models and episode_idx != getattr(self, '_last_perturbed_episode', -1):
            if episode_idx is None:
                raise ValueError(
                    "episode_idx must be provided when reset_models=True"
                )
            self._last_perturbed_episode = episode_idx
            policy_rng, critic_rng, model_rng, self.rng = jax.random.split(self.rng, 4)
            new_actor, new_target_actor = self.policy_perturb_module.perturb(self.actor, self.target_actor, batch.observations, policy_rng, episode_idx)
            new_critic, new_target_critic = self.critic_perturb_module.perturb(self.critic, self.target_critic, batch.observations, batch.actions, critic_rng, episode_idx)
            new_ens_state = self.model_perturb_module.perturb(self.ens_state, batch.observations, batch.actions, model_rng, episode_idx)
            self.actor, self.target_actor = new_actor, new_target_actor
            self.critic, self.target_critic = new_critic, new_target_critic
            self.ens_state = new_ens_state

        self.step += 1
        
        # Update Ensemble
        new_ens_state, ens_info = update_ensemble_jit(
            batch, 
            self.ensemble, 
            self.ens_state, 
            self.config.predict_reward, 
            self.config.predict_diff, 
            self.config.dt, 
            self.config.action_repeat
        )
        self.ens_state = new_ens_state

        # Generate rollouts
        rollout_key, self.rng = jax.random.split(self.rng)
        full_rollout = rollout_learned_model(
            batch, self.ensemble, self.ens_state, self.actor, self.config.predict_reward, 
            self.config.predict_diff, self.config.sample_model, rollout_key, self.config.num_imagined_steps, 
            self.config.dt, self.config.action_repeat, self.config.reward_model
        )

        # Update actor/critic
        sample_key, self.rng = jax.random.split(self.rng)
        indices = jax.random.randint(sample_key, (self.config.actor_critic_updates_per_model_update,), 0, self.config.num_imagined_steps + 1)
        
        aggregated_info = collections.defaultdict(list)
        for idx in indices:
            current_batch = jax.tree_util.tree_map(lambda x: x[idx], full_rollout)
            train_batch = Batch(current_batch.observations, current_batch.actions, current_batch.rewards, current_batch.masks, current_batch.next_observations)
            
            self.rng, self.actor, self.critic, self.target_actor, self.target_critic, \
            self.temp, self.dyn_ent_temp, self.ens_state, ac_info = update_ac_jit(
                rng=self.rng,
                actor=self.actor,
                critic=self.critic,
                target_actor=self.target_actor,
                target_critic=self.target_critic,
                temp=self.temp,
                dyn_entropy_temp=self.dyn_ent_temp,
                ens=self.ensemble,
                ens_state=self.ens_state,
                batch=train_batch,
                discount=self.config.discount,
                tau=self.config.tau,
                target_entropy=self.config.target_entropy,
                backup_entropy=self.config.backup_entropy,
                update_target=self.step % self.config.target_update_period == 0,
                use_log_transform=self.config.use_log_transform,
                update_policy=self.step % self.config.policy_update_period == 0
            )
            for k, v in ac_info.items(): aggregated_info[k].append(v)

        return {**ens_info, **{k: np.mean(v) for k, v in aggregated_info.items()}}