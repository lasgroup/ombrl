"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple, Dict, List, Callable

import jax  
import jax.numpy as jnp
import numpy as np
import optax
import copy
from jaxrl.agents.sac import temperature
from ombrl.agents.sombrl.actor import update as update_actor
from ombrl.agents.sombrl.critic import target_update
from ombrl.agents.sombrl.critic import update as update_critic
from jaxrl.agents.sac.temperature import update as update_temp

from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey
from maxinforl_jax.models.ensemble_model import EnsembleState, DeterministicEnsemble, ProbabilisticEnsemble


@jax.jit
def jax_symexp(x):
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)

def predict_batch_from_batch(
        reward_model: Callable,
        batch: Batch,
        action_repeat: float = 1.0,
        use_symlog: bool = False,
        predict_rewards: bool = False,
        ) -> Batch:
    """
    Populate batch.rewards for a (possibly imagined) batch.

    Behavior:
      - If predict_rewards is True and next_observations has an extra dimension
        (state_dim + 1), treat its last component as the predicted reward and
        trim next_observations back to state_dim.
      - Otherwise, compute rewards via `reward_model(observation, action, next_observation)`,
        optionally undoing symlog on observations.
    """
    if predict_rewards:
        next_obs = batch.next_observations
        pred_rewards = next_obs[..., -1]
        trimmed_next_obs = next_obs[..., :-1]
        return batch._replace(next_observations=trimmed_next_obs,
                              rewards=pred_rewards)

    def per_sample(o, a, no):
        if use_symlog:
            o  = jax_symexp(o)
            no = jax_symexp(no)
        # Support both .predict(...) and direct call
        if hasattr(reward_model, "predict"):
            return reward_model.predict(o, a, no)
        else:
            return reward_model(o, a, no)

    new_rewards = jax.vmap(per_sample, in_axes=0)(
        batch.observations, batch.actions, batch.next_observations
    )
    new_rewards = action_repeat * new_rewards # If we repeat action we scale the reward with action repeat in order to have same reward scale
    return batch._replace(rewards=new_rewards)


def get_imagined_batch(
        batch: Batch,
        ens: DeterministicEnsemble,
        ens_state: EnsembleState,
        predict_rewards: bool,
        predict_diff: bool,
        sample_model: bool,
        key: PRNGKey,
        ) -> Batch:
    """
    Generate imagined next states using the ensemble model.
    """
    input = jnp.concatenate([batch.observations, batch.actions], axis=-1)
    ens_mean, ens_std = ens(input=input, state=ens_state, denormalize_output=True)
    noise_key, key = jax.random.split(key, 2)
    if sample_model:
        ens_mean = jax.random.choice(key=key, a=ens_mean, axis=0)
        ens_std = jax.random.choice(key=key, a=ens_std, axis=0)
    else:
        ens_mean = jnp.mean(ens_mean, axis=0)
        ens_std = jnp.mean(ens_std, axis=0)

    if predict_rewards:
        ens_mean = ens_mean[..., :-1]
        ens_std = ens_std[..., :-1]
    next_state = ens_mean + jax.random.normal(noise_key, shape=ens_std.shape) * ens_std

    if predict_diff:
        next_state = next_state + batch.observations
    imagined_batch = batch._replace(next_observations=next_state)
    return imagined_batch



@functools.partial(jax.jit, static_argnames=('ens', 'predict_diff', 'predict_rewards'))
def update_ensemble(batch: Batch,
                    ens: DeterministicEnsemble,
                    ens_state: EnsembleState,
                    predict_rewards: bool,
                    predict_diff: bool,
                    ) -> Tuple[EnsembleState, InfoDict, Batch]:
    expl_rew, ens_state = ens.get_info_gain(input=jnp.concatenate([batch.observations, batch.actions], axis=-1),
                                            state=ens_state,
                                            update_normalizer=True)

    outputs = batch.next_observations
    if predict_diff:
        outputs -= batch.observations
        
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
    return new_ens_state, ens_info, batch._replace(rewards=expl_rew)


def update_policy_actions(imagined_batch: Batch, actor: Model, rng: PRNGKey) -> Tuple[PRNGKey, Batch]:
    rng, actions = policies.sample_actions(rng, actor.apply_fn,
                                       actor.params, imagined_batch.observations,
                                       temperature = 1)
    return rng, imagined_batch._replace(actions=actions)
    

@functools.partial(jax.jit,
                   static_argnames=('ens',
                                    'backup_entropy',
                                    'update_target',
                                    'use_log_transform',
                                    'predict_rewards',
                                    'predict_diff',
                                    'sample_model',
                                    'update_critic_with_real_data',
                                    'update_policy',
                                    'reward_model',
                                    'num_imagined_steps',
                                    'use_symlog'
                                    ))
def _update_jit(
        rng: PRNGKey, actor: Model, critic: Model, target_critic: Model, temp: Model,
        ens: DeterministicEnsemble, ens_state: EnsembleState,
        batch: Batch, discount: float, tau: float,
        target_entropy: float, backup_entropy: bool, update_target: bool,
        use_log_transform: bool, predict_rewards: bool, predict_diff: bool,
        sample_model: bool, update_critic_with_real_data: bool, update_policy: bool,
        action_repeat: int, num_imagined_steps: int = 1, reward_model: Optional[Callable] = None,
        use_symlog: bool = True,
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:
    # [x]: Keep update critic with real data?
    rng, key = jax.random.split(rng)
    if update_critic_with_real_data:
        new_critic, critic_info = update_critic(
            key=key,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            batch=batch,
            discount=discount,
            backup_entropy=backup_entropy,
        )
    else:
        new_critic = critic
        critic_info = {}

    rng, model_sample_key, critic_key = jax.random.split(rng, 3)
    imagined_batch = get_imagined_batch(
            batch=batch,
            ens_state=ens_state,
            ens=ens,
            predict_diff=predict_diff,
            predict_rewards=predict_rewards,
            sample_model=sample_model,
            key=model_sample_key,
            )

    new_critic, imagined_critic_info = update_critic(
        key=critic_key,
        actor=actor,
        critic=new_critic,
        target_critic=target_critic,
        temp=temp,
        batch=imagined_batch,
        discount=discount,
        backup_entropy=backup_entropy,
    )
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    for _ in range(num_imagined_steps - 1):
        # Re-sample actions using the current policy for the next imagined step, replace current state with the next imagined state
        rng, model_sample_key, critic_key = jax.random.split(rng, 3)

        imagined_batch = imagined_batch._replace(
            observations=imagined_batch.next_observations)

        rng, imagined_batch = update_policy_actions(imagined_batch, actor, rng)

        imagined_batch = get_imagined_batch(
            batch=imagined_batch,
            ens_state=ens_state,
            ens=ens,
            predict_diff=predict_diff,
            predict_rewards=predict_rewards,
            sample_model=sample_model,
            key=model_sample_key,
            )
        
        if reward_model is not None:
            imagined_batch = predict_batch_from_batch(reward_model, imagined_batch, action_repeat, use_symlog, predict_rewards)
        else:
            expl_rew, _ = ens.get_info_gain(input=jnp.concatenate(
                                            [imagined_batch.observations, imagined_batch.actions], axis=-1),
                                            state=ens_state,
                                            update_normalizer=False)
            imagined_batch = imagined_batch._replace(rewards=expl_rew)

        new_critic, imagined_critic_info = update_critic(
            key=critic_key,
            actor=actor,
            critic=new_critic,
            target_critic=target_critic,
            temp=temp,
            batch=imagined_batch,
            discount=discount,
            backup_entropy=backup_entropy,
        )

        if update_target:
            new_target_critic = target_update(new_critic, target_critic, tau)
        else:
            new_target_critic = target_critic

    imagined_critic_info = {f'imagined_critic_{key}': val for key, val in imagined_critic_info.items()}

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
        **imagined_critic_info,
        **actor_info,
        **alpha_info
    }


class SombrlExplorerLearner(object):
    """Initializes the SombrlExplorerLearner.
        
    Has different exploration methods:
            - Mean (greedy) exploration if:
                - int_rew_weight_start = int_rew_weight_end = -1
                - int_rew_weight_decrease_steps = -1
                - sample_model = False
                - explore_until = 0
            - PETS-like exploration if:
                - int_rew_weight_start = int_rew_weight_end = -1
                - int_rew_weight_decrease_steps = -1
                - sample_model = True
                - explore_until = 0
            - SOMBRL exploration if:
                - int_rew_weight_start >= 0
                - int_rew_weight_start >= int_rew_weight_end >= 0
                - int_rew_weight_decrease_steps >= 0
                - explore_until > 0
            - Unsupervised exploration if:
                - int_rew_weight_start = int_rew_weight_end = -1
                - int_rew_weight_decrease_steps = -1
                - explore_until > 0
    """

    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 reward_model: Optional[Callable] = None,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 ens_lr: float = 3e-4,
                 ens_wd: float = 0.0,
                 hidden_dims: Sequence[int] = (256, 256),
                 model_hidden_dims: Sequence[int] = (256, 256),
                 num_heads: int = 5,
                 predict_reward: bool = False,
                 predict_diff: bool = True,
                 use_symlog: bool = True,
                 use_log_transform: bool = False,
                 learn_std: bool = False,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 backup_entropy: bool = True,
                 init_temperature: float = 1.0,
                 init_mean: Optional[np.ndarray] = None,
                 policy_final_fc_init_scale: float = 1.0,
                 sample_model: bool = True,
                 critic_real_data_update_period: int = 2,
                 policy_update_period: Optional[int] = None,
                 expl_agent_update_period: int = 1,
                 agent_update_period: int = 1,
                 max_gradient_norm: Optional[float] = None,
                 use_bronet: bool = False,
                 reset_period: Optional[int] = None,
                 reset_models: bool = False,
                 explore_until: int = 1_000_000,
                 action_repeat: int = None,
                 int_rew_weight_start: float = -1.0,
                 int_rew_weight_end: float = 0.0,
                 int_rew_weight_decrease_steps: int = -1,
                 num_imagined_steps: int = 1,
                 ):

        self.predict_diff = predict_diff
        self.predict_reward = predict_reward
        self.expl_agent_update_period = expl_agent_update_period
        self.agent_update_period = agent_update_period

        self.num_heads = num_heads
        self.sample_model = sample_model
        self.critic_real_data_update_period = critic_real_data_update_period
        if policy_update_period:
            self.policy_update_period = policy_update_period
        else:
            self.policy_update_period = critic_real_data_update_period

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        if reset_period is None:
            self.reset_period = 2_500_000
        else:
            self.reset_period = reset_period

        self._reset_models = reset_models

        actor_optimizer = optax.adam(learning_rate=actor_lr)
        critic_optimizer = optax.adam(learning_rate=critic_lr)
        temp_optimizer = optax.adam(learning_rate=temp_lr)
        model_optimizer = optax.adamw(learning_rate=ens_lr, weight_decay=ens_wd)
        if max_gradient_norm:
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
            model_optimizer = optax.chain(
                optax.clip_by_global_norm(max_gradient_norm),
                model_optimizer,
            )

        actor_def = policies.NormalTanhPolicy(
            hidden_dims,
            action_dim,
            init_mean=init_mean,
            final_fc_init_scale=policy_final_fc_init_scale)
        
        critic_def = critic_net.DoubleCritic(hidden_dims, use_bronet=use_bronet)

        # Key splitting
        rng = jax.random.PRNGKey(seed)
        rng, actor_keys, critic_keys, temp_keys = jax.random.split(rng, 4)
        actor_keys = jax.random.split(actor_keys, 2)
        critic_keys = jax.random.split(critic_keys, 2)
        temp_keys = jax.random.split(temp_keys, 2)

        actor = Model.create(actor_def,
                                inputs=[actor_keys[0], observations],
                                tx=actor_optimizer)
        critic = Model.create(critic_def,
                                inputs=[critic_keys[0], observations, actions],
                                tx=critic_optimizer)
        target_critic = Model.create(critic_def,
                                        inputs=[critic_keys[0], observations, actions])
        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_keys[0]],
                            tx=temp_optimizer)

        expl_actor = Model.create(actor_def,
                                  inputs=[actor_keys[-1], observations],
                                  tx=actor_optimizer)

        expl_critic = Model.create(critic_def,
                                   inputs=[critic_keys[-1], observations, actions],
                                   tx=critic_optimizer)
        
        expl_target_critic = Model.create(critic_def, 
                                          inputs=[critic_keys[-1], observations, actions])
        
        expl_temp = Model.create(temperature.Temperature(init_temperature),
                                 inputs=[temp_keys[-1]],
                                 tx=temp_optimizer)

        model_key, rng = jax.random.split(rng, 2)

        output_dim = observations.shape[-1]
        if predict_reward:
            output_dim += 1

        if learn_std:
            model_type = ProbabilisticEnsemble
        else:
            model_type = DeterministicEnsemble
        ensemble = model_type(
            model_kwargs={'hidden_dims': model_hidden_dims + (output_dim,)},
            optimizer=model_optimizer,
            num_heads=self.num_heads,
            use_entropy_for_int_rew=False,
        )

        ens_state = ensemble.init(key=model_key, input=jnp.concatenate([observations, actions], axis=-1))

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.expl_actor = expl_actor
        self.expl_critic = expl_critic
        self.expl_target_critic = expl_target_critic
        self.expl_temp = expl_temp
        self.rng = rng
        self.step = 0
        self.use_log_transform = use_log_transform
        self.explore_until = explore_until

        self.ens = ensemble
        self.ens_state = ens_state

        self.step = 1

        self.action_repeat = action_repeat

        if int_rew_weight_decrease_steps >= 0:
            assert int_rew_weight_start >= 0
            self.int_rew_weight_schedule = optax.linear_schedule(
                init_value=int_rew_weight_start,
                end_value=int_rew_weight_end,
                transition_steps=int_rew_weight_decrease_steps,
            )
        else:
            self.int_rew_weight_schedule = optax.constant_schedule(
                value=int_rew_weight_start
            )

        self.reward_model = reward_model
        self.use_symlog = use_symlog
        self.num_imagined_steps = num_imagined_steps


    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0,
                       reward_index: int = 0) -> np.ndarray:
        if temperature == 0:
            rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                                   self.actor.params, observations,
                                                   temperature)
        else:
            if self.step <= self.explore_until:
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

    def update(self, batch: Batch) -> InfoDict:
        if self._reset_models:
            rng, self.rng = jax.random.split(self.rng)
            rng, agent_rng = jax.random.split(rng)
            raise NotImplementedError("Pertubation model not implemented")
            self.actor = new_actor
            self.critic = new_critic
            self.target_critic = new_target_critic
            self.ens_state = new_ens_state
        
        self.step += 1
        new_ens_state, ens_info, expl_batch = update_ensemble(batch=batch,
                                                              ens=self.ens,
                                                              ens_state=self.ens_state,
                                                              predict_diff=self.predict_diff,
                                                              predict_rewards=self.predict_reward,
                                                              )
        
        int_rew_weight = self.int_rew_weight_schedule(self.step)

        if int_rew_weight >= 0:
            # SOMBRL
            external_reward = batch.rewards
            internal_rewards = expl_batch.rewards
            expl_batch = expl_batch._replace(
                rewards=(external_reward + int_rew_weight * internal_rewards) / (1+int_rew_weight))
        else:
            # OPAX or greedy
            pass

        self.ens_state = new_ens_state
        info = ens_info

        if self.step % self.agent_update_period == 0:
            new_rng, new_actor, new_critic, new_target_critic, new_temp, agent_info = _update_jit(
                self.rng, self.actor, self.critic, self.target_critic, self.temp,
                self.ens, self.ens_state, batch, self.discount, self.tau, self.target_entropy,
                self.backup_entropy, self.step % self.target_update_period == 0,
                self.use_log_transform, self.predict_reward, self.predict_diff,
                self.sample_model, self.step % self.critic_real_data_update_period == 0,
                self.step % self.policy_update_period == 0, self.action_repeat,
                self.num_imagined_steps, self.reward_model, self.use_symlog)

            self.rng = new_rng
            self.actor = new_actor
            self.critic = new_critic
            self.target_critic = new_target_critic
            self.temp = new_temp

            info = info | {f'agent_{k}': v for k, v in agent_info.items()}

        if self.step % self.expl_agent_update_period == 0:
            new_rng, new_actor, new_critic, new_target_critic, new_temp, agent_info = _update_jit(
                self.rng, self.expl_actor, self.expl_critic, self.expl_target_critic, self.expl_temp,
                self.ens, self.ens_state, expl_batch, self.discount, self.tau, self.target_entropy,
                self.backup_entropy, self.step % self.target_update_period == 0, 
                self.use_log_transform, self.predict_reward, self.predict_diff, 
                self.sample_model, self.step % self.critic_real_data_update_period == 0,
                self.step % self.policy_update_period == 0, self.action_repeat,
                self.num_imagined_steps, None, self.use_symlog)

            self.rng = new_rng
            self.expl_actor = new_actor
            self.expl_critic = new_critic
            self.expl_target_critic = new_target_critic
            self.expl_temp = new_temp
            agent_info = {f'expl_{k}': v for k, v in agent_info.items()}

            info = info | agent_info

        return info | {'policy_phase': int(self.step <= self.explore_until),
                       'int_reward_weight': float(int_rew_weight)}
