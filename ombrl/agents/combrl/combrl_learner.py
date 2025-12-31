"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import optax
import copy
from jaxrl.agents.sac import temperature
from ombrl.agents.combrl.actor import update as update_actor
from ombrl.agents.combrl.critic import target_update
from ombrl.agents.combrl.critic import update as update_critic
from ombrl.utils.pertubation import PerturbationModule
from jaxrl.agents.sac.temperature import update as update_temp

from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey
from maxinforl_jax.models.ensemble_model import EnsembleState, DeterministicEnsemble, ProbabilisticEnsemble
from ombrl.utils.multiple_reward_wrapper import RewardFunction


def get_imagined_batch(
        batch: Batch,
        ens: DeterministicEnsemble,
        ens_state: EnsembleState,
        predict_rewards: bool,
        predict_diff: bool,
        sample_model: bool,
        key: PRNGKey, # type: ignore
        dt: float = None,
        action_repeat: int = 1,
):
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
        if dt is not None:
            # CT case: The ensemble predicts the derivative of the next_state
            next_state = next_state * dt * action_repeat
        next_state = next_state + batch.observations
    imagined_batch = batch._replace(next_observations=next_state)
    return imagined_batch



@functools.partial(jax.jit, static_argnames=('ens', 'predict_diff', 'predict_rewards'))
def update_ensemble(batch: Batch,
                    ens: DeterministicEnsemble,
                    ens_state: EnsembleState,
                    predict_rewards: bool,
                    predict_diff: bool,
                    dt: float = None,
                    action_repeat: int = 1,
                    ) -> Tuple[EnsembleState, InfoDict, Batch]:
    expl_rew, ens_state = ens.get_info_gain(input=jnp.concatenate([batch.observations, batch.actions], axis=-1),
                                            state=ens_state,
                                            update_normalizer=True)

    if predict_diff:
        outputs = batch.next_observations - batch.observations
        if dt is not None:
            outputs = outputs / (dt * action_repeat)
    else:
        outputs = batch.next_observations
    assert predict_rewards==False, "predict rewards should be False for COMBRL exps"
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
                                    ))
def _update_jit(
        rng: PRNGKey, actor: Model, critic: Model, target_critic: Model, temp: Model, # type: ignore
        ens: DeterministicEnsemble, ens_state: EnsembleState,
        batch: Batch, discount: float, tau: float,
        target_entropy: float, backup_entropy: bool, update_target: bool,
        use_log_transform: bool, predict_rewards: bool, predict_diff: bool,
        sample_model: bool, update_critic_with_real_data: bool, update_policy: bool,
        dt: float, action_repeat: int,
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]: # type: ignore
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

    rng, model_sample_key = jax.random.split(rng)
    imagined_batch = get_imagined_batch(
        batch=batch,
        ens_state=ens_state,
        ens=ens,
        predict_diff=predict_diff,
        predict_rewards=predict_rewards,
        sample_model=sample_model,
        key=model_sample_key,
        dt=dt,
        action_repeat=action_repeat,
    )
    rng, key = jax.random.split(rng)
    new_critic, imagined_critic_info = update_critic(
        key=key,
        actor=actor,
        critic=new_critic,
        target_critic=target_critic,
        temp=temp,
        batch=imagined_batch,
        discount=discount,
        backup_entropy=backup_entropy,
    )

    imagined_critic_info = {f'imagined_critic_{key}': val for key, val in imagined_critic_info.items()}

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
        **imagined_critic_info,
        **actor_info,
        **alpha_info
    }


class COMBRLExplorerLearner(object):

    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 reward_list: List[RewardFunction],
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
                 perturb_rate: float = 0.2,
                 perturb_policy: bool = True,
                 perturb_model: bool = True,
                 explore_until: int = 1_000_000,
                 pseudo_ct: bool = False,
                 dt: float = None,
                 action_repeat: int = None,
                int_rew_weight_start: float = -1.0,
                int_rew_weight_end: float = 0.0,
                int_rew_weight_decrease_steps: int = -1,
                 ):
        self.predict_diff = predict_diff
        self.predict_reward = predict_reward
        self.expl_agent_update_period = expl_agent_update_period
        self.agent_update_period = agent_update_period

        self.num_heads = num_heads
        self.sample_model = sample_model
        self.critic_real_data_update_period = critic_real_data_update_period
        self.perturb_rate = perturb_rate
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

        num_rewards = len(reward_list)
        self.reward_list = reward_list


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
        actor_keys = jax.random.split(actor_keys, num_rewards+1)
        critic_keys = jax.random.split(critic_keys, num_rewards+1)
        temp_keys = jax.random.split(temp_keys, num_rewards+1)

        self.agents: List[Dict[str, RewardFunction | Model]] = []
        for i, reward_function in enumerate(reward_list):
            actor = Model.create(actor_def,
                                inputs=[actor_keys[i], observations],
                                tx=actor_optimizer)
            critic = Model.create(critic_def,
                                inputs=[critic_keys[i], observations, actions],
                                tx=critic_optimizer)
            target_critic = Model.create(critic_def, 
                                         inputs=[critic_keys[i], observations, actions])
            
            temp = Model.create(temperature.Temperature(init_temperature),
                                inputs=[temp_keys[i]],
                                tx=temp_optimizer)
            self.agents.append({
                'reward_fn': reward_function,
                'actor': actor,
                'critic': critic,
                'target_critic': target_critic,
                'temp': temp,
            })
        
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

        self.perturb_module = PerturbationModule(
            actor_init_fn=actor_def.init,
            critic_init_fn=critic_def.init,
            model_init_fn=ensemble.init,
            actor_init_opt_state=copy.deepcopy(actor.opt_state),
            critic_init_opt_state=copy.deepcopy(critic.opt_state),
            perturb_rate=perturb_rate,
            perturbation_freq=self.reset_period,
            perturb_policy=perturb_policy,
            perturb_model=perturb_model,
        )

        self.use_log_transform = use_log_transform

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
        if dt is not None:
            assert pseudo_ct == True, f"continuous-time must be enabled for given dt, got: {pseudo_ct}"
            assert predict_diff == True, \
            f"predict_diff should be True in the pseudo-ct case, got: {predict_diff} for dt={dt}"
        else:
            assert pseudo_ct == False, f"continuous-time must be disabled for given dt, got: {pseudo_ct}"
        self.dt = dt
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

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0,
                       reward_index: int = 0) -> np.ndarray:
        agent = self.agents[reward_index]
        if temperature == 0:
            rng, actions = policies.sample_actions(self.rng, agent['actor'].apply_fn,
                                                agent['actor'].params, observations,
                                                temperature)
        else:
            if self.step <= self.explore_until:
                rng, actions = policies.sample_actions(self.rng, self.expl_actor.apply_fn,
                                                    self.expl_actor.params, observations,
                                                    temperature)
            else:
                rng, actions = policies.sample_actions(self.rng, agent['actor'].apply_fn,
                                                    agent['actor'].params, observations,
                                                    temperature)
        self.rng = rng
        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        if self._reset_models:
            rng, self.rng = jax.random.split(self.rng)
            for agent in self.agents:
                rng, agent_rng = jax.random.split(rng)
                new_actor, new_critic, new_target_critic, new_ens_state = self.perturb_module.perturb(
                    actor=agent['actor'],
                    critic=agent['critic'],
                    target_critic=agent['target_critic'],
                    ens_state=self.ens_state,
                    observation=batch.observations,
                    action=batch.actions,
                    rng=agent_rng,
                    step=self.step
                )
                agent['actor'] = new_actor
                agent['critic'] = new_critic
                agent['target_critic'] = new_target_critic
                self.ens_state = new_ens_state
        
        self.step += 1
        new_ens_state, ens_info, expl_batch = update_ensemble(batch=batch,
                                                              ens=self.ens,
                                                              ens_state=self.ens_state,
                                                              predict_diff=self.predict_diff,
                                                              predict_rewards=self.predict_reward,
                                                              dt=self.dt,
                                                              action_repeat=self.action_repeat,
                                                              )
        
        int_rew_weight = self.int_rew_weight_schedule(self.step)

        if int_rew_weight >= 0:
            # COMBRL
            external_reward = batch.rewards
            internal_rewards = expl_batch.rewards
            expl_batch = expl_batch._replace(rewards=external_reward + int_rew_weight * internal_rewards)
        else:
            # Unsupervised COMBRL or greedy
            pass

        self.ens_state = new_ens_state

        info = ens_info

        if self.step % self.agent_update_period == 0:
            for idx, agent in enumerate(self.agents):
                new_rewards = self.action_repeat * agent['reward_fn'](
                    batch.observations, batch.actions, batch.next_observations, batch.rewards
                )
                # Create a new batch with the updated rewards.
                new_batch = batch._replace(rewards=new_rewards)
                
                new_rng, new_actor, new_critic, new_target_critic, new_temp, agent_info = _update_jit(
                    self.rng, agent['actor'], agent['critic'], agent['target_critic'], agent['temp'],
                    self.ens, self.ens_state, new_batch, self.discount, self.tau, self.target_entropy,
                    self.backup_entropy, self.step % self.target_update_period == 0, 
                    self.use_log_transform, self.predict_reward, self.predict_diff, 
                    self.sample_model, self.step % self.critic_real_data_update_period == 0,
                    self.step % self.policy_update_period == 0, self.dt, self.action_repeat)

                self.rng = new_rng
                agent['actor'] = new_actor
                agent['critic'] = new_critic
                agent['target_critic'] = new_target_critic
                agent['temp'] = new_temp

                # info = info | agent_info
                info = info | {f'agent_{idx}_{k}': v for k, v in agent_info.items()}

        if self.step % self.expl_agent_update_period == 0:
            new_rng, new_actor, new_critic, new_target_critic, new_temp, agent_info = _update_jit(
                self.rng, self.expl_actor, self.expl_critic, self.expl_target_critic, self.expl_temp,
                self.ens, self.ens_state, expl_batch, self.discount, self.tau, self.target_entropy,
                self.backup_entropy, self.step % self.target_update_period == 0, 
                self.use_log_transform, self.predict_reward, self.predict_diff, 
                self.sample_model, self.step % self.critic_real_data_update_period == 0,
                self.step % self.policy_update_period == 0, self.dt, self.action_repeat)

            self.rng = new_rng
            self.expl_actor = new_actor
            self.expl_critic = new_critic
            self.expl_target_critic = new_target_critic
            self.expl_temp = new_temp
            agent_info = {f'expl_{k}': v for k, v in agent_info.items()}

            info = info | agent_info

        return info | {'policy_phase': int(self.step <= self.explore_until),
                       'int_reward_weight': float(int_rew_weight)}
