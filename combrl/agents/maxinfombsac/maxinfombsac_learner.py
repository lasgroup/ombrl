import functools
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import copy
from jaxrl.agents.sac import temperature
from maxinforl_jax.agents.maxinfosac.actor import update as update_actor
from maxinforl_jax.agents.maxinfosac.critic import target_update
from maxinforl_jax.agents.maxinfosac.critic import update as update_critic
from combrl.agents.maxinfombsac.utils import PerturbationModule
from jaxrl.agents.sac.temperature import update as update_temp

from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey

from maxinforl_jax.models import EnsembleState, DeterministicEnsemble, ProbabilisticEnsemble


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
        rng: PRNGKey, actor: Model, critic: Model, target_actor: Model, target_critic: Model, temp: Model, # type: ignore
        dyn_entropy_temp: Model, ens: DeterministicEnsemble, ens_state: EnsembleState,
        batch: Batch, discount: float, tau: float,
        target_entropy: float, backup_entropy: bool, update_target: bool,
        use_log_transform: bool, predict_rewards: bool, predict_diff: bool,
        sample_model: bool, update_critic_with_real_data: bool, update_policy: bool,
        dt: float, action_repeat: int,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, Model, EnsembleState, InfoDict]: # type: ignore
    rng, key = jax.random.split(rng)
    if update_critic_with_real_data:
        new_critic, ens_state, critic_info = update_critic(
            key=key,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            dyn_entropy_temp=dyn_entropy_temp,
            ens=ens,
            ens_state=ens_state,
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
    new_critic, ens_state, imagined_critic_info = update_critic(
        key=key,
        actor=actor,
        critic=new_critic,
        target_critic=target_critic,
        temp=temp,
        dyn_entropy_temp=dyn_entropy_temp,
        ens=ens,
        ens_state=ens_state,
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
        new_actor, ens_state, actor_info = update_actor(key=key,
                                                        actor=actor,
                                                        target_actor=target_actor,
                                                        critic=new_critic,
                                                        temp=temp,
                                                        dyn_entropy_temp=dyn_entropy_temp,
                                                        ens=ens,
                                                        ens_state=ens_state,
                                                        batch=batch,
                                                        )
        if update_target:
            new_target_actor = target_update(new_actor, target_actor, tau)
        else:
            new_target_actor = target_actor

        new_temp, alpha_info = update_temp(temp, actor_info['entropy'],
                                           target_entropy, use_log_transform=use_log_transform)
        new_dyn_entropy_temp, dyn_ent_info = update_temp(dyn_entropy_temp, actor_info['info_gain'],
                                                         actor_info['target_info_gain'],
                                                         use_log_transform=use_log_transform)
        dyn_ent_info = {f'dyn_ent_{key}': val for key, val in dyn_ent_info.items()}
    else:
        new_actor, new_temp, new_dyn_entropy_temp, new_target_actor = actor, temp, dyn_entropy_temp, target_actor
        actor_info, alpha_info, dyn_ent_info = {}, {}, {}

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
                # 'ens_inp_num_points': ens_state.ensemble_normalizer_state.input_normalizer_state.num_points,
                'ens_out_mean': ens_state.ensemble_normalizer_state.output_normalizer_state.mean.mean(),
                'ens_out_std': ens_state.ensemble_normalizer_state.output_normalizer_state.std.mean(),
                # 'ens_out_num_points': ens_state.ensemble_normalizer_state.output_normalizer_state.num_points,
                'ens_info_gain_mean': ens_state.ensemble_normalizer_state.info_gain_normalizer_state.mean.mean(),
                'ens_info_gain_std': ens_state.ensemble_normalizer_state.info_gain_normalizer_state.std.mean(),
                # 'ens_info_gain_num_points': ens_state.ensemble_normalizer_state.info_gain_normalizer_state.num_points,
                }

    return rng, \
        new_actor, \
        new_critic, \
        new_target_actor, \
        new_target_critic, \
        new_temp, \
        new_dyn_entropy_temp, \
        new_ens_state, {
        **critic_info,
        **imagined_critic_info,
        **actor_info,
        **alpha_info,
        **dyn_ent_info,
        **ens_info,
    }


class MaxInfoMBSacLearner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 dyn_ent_lr: float = 3e-4,
                 dyn_wd: float = 0.0,
                 ens_lr: float = 3e-4,
                 ens_wd: float = 0.0,
                 hidden_dims: Sequence[int] = (256, 256),
                 model_hidden_dims: Sequence[int] = (256, 256),
                 num_heads: int = 5,
                 predict_reward: bool = True,
                 predict_diff: bool = True,
                 use_log_transform: bool = True,
                 learn_std: bool = False,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 backup_entropy: bool = True,
                 init_temperature: float = 1.0,
                 init_temperature_dyn_entropy: float = 1.0,
                 init_mean: Optional[np.ndarray] = None,
                 policy_final_fc_init_scale: float = 1.0,
                 sample_model: bool = True,
                 critic_real_data_update_period: int = 2,
                 policy_update_period: Optional[int] = None,
                 max_gradient_norm: Optional[float] = None,
                 use_bronet: bool = False,
                 reset_period: Optional[int] = None,
                 reset_models: bool = False,
                 perturb_rate: float = 0.2,
                 perturb_policy: bool = True,
                 perturb_model: bool = True,
                 pseudo_ct: bool = False,
                 dt: float = None,
                 action_repeat: int = None,
                 ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        self.predict_reward = predict_reward
        self.predict_diff = predict_diff
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

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
        actor_def = policies.NormalTanhPolicy(
            hidden_dims,
            action_dim,
            init_mean=init_mean,
            final_fc_init_scale=policy_final_fc_init_scale)
        actor_optimizer = optax.adam(learning_rate=actor_lr)
        critic_optimizer = optax.adam(learning_rate=critic_lr)
        temp_optimizer = optax.adam(learning_rate=temp_lr)
        dyn_ent_temp_optimizer = optax.adamw(learning_rate=dyn_ent_lr, weight_decay=dyn_wd)
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

            dyn_ent_temp_optimizer = optax.chain(
                optax.clip_by_global_norm(max_gradient_norm),
                dyn_ent_temp_optimizer,
            )
            model_optimizer = optax.chain(
                optax.clip_by_global_norm(max_gradient_norm),
                model_optimizer,
            )

        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=actor_optimizer)

        target_actor = Model.create(actor_def,
                                    inputs=[actor_key, observations])

        critic_def = critic_net.DoubleCritic(hidden_dims, use_bronet=use_bronet)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=critic_optimizer)
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])


        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=temp_optimizer)

        # information gain kwargs
        dyn_ent_temp_key, rng = jax.random.split(rng, 2)
        dyn_ent_temp = Model.create(temperature.Temperature(init_temperature_dyn_entropy),
                                    inputs=[dyn_ent_temp_key],
                                    tx=dyn_ent_temp_optimizer)

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
            use_entropy_for_int_rew=False,  # return model epistemic uncertainty as the intrinsic rew
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

        self.actor = actor
        self.target_actor = target_actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.dyn_ent_temp = dyn_ent_temp
        self.ens_state = ens_state
        self.ensemble = ensemble
        self.rng = rng

        self.step = 1
        if dt is not None:
            assert pseudo_ct == True, f"continuous-time must be enabled for given dt, got: {pseudo_ct}"
            assert predict_diff == True, \
            f"predict_diff should be True in the pseudo-ct case, got: {predict_diff} for dt={dt}"
        else:
            assert pseudo_ct == False, f"continuous-time must be disabled for given dt, got: {pseudo_ct}"
        self.dt = dt
        self.action_repeat = action_repeat

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> np.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        if self._reset_models:
            rng, self.rng = jax.random.split(self.rng)
            actor, critic, target_actor, target_critic, new_ens_state = self.perturb_module.perturb(
                actor=self.actor,
                critic=self.critic,
                target_actor=self.target_actor,
                target_critic=self.target_critic,
                ens_state=self.ens_state,
                observation=batch.observations,
                action=batch.actions,
                rng=rng,
                step=self.step
            )
            self.actor = actor
            self.target_actor = target_actor
            self.critic = critic
            self.target_critic = target_critic
            self.ens_state = new_ens_state

        self.step += 1
        new_rng, new_actor, new_critic, new_target_actor, new_target_critic, \
            new_temp, new_dyn_entropy_temp, new_ens_state, info = _update_jit(
            rng=self.rng,
            actor=self.actor,
            critic=self.critic,
            target_actor=self.target_actor,
            target_critic=self.target_critic,
            temp=self.temp,
            dyn_entropy_temp=self.dyn_ent_temp,
            ens=self.ensemble,
            ens_state=self.ens_state,
            batch=batch,
            discount=self.discount,
            tau=self.tau,
            target_entropy=self.target_entropy,
            backup_entropy=self.backup_entropy,
            update_target=self.step % self.target_update_period == 0,
            use_log_transform=self.use_log_transform,
            predict_rewards=self.predict_reward,
            predict_diff=self.predict_diff,
            sample_model=self.sample_model,
            update_critic_with_real_data=self.step % self.critic_real_data_update_period == 0,
            update_policy=self.step % self.policy_update_period == 0,
            dt=self.dt,
            action_repeat=self.action_repeat,
        )

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_actor = new_target_actor
        self.target_critic = new_target_critic
        self.temp = new_temp
        self.dyn_ent_temp = new_dyn_entropy_temp
        self.ens_state = new_ens_state

        return info
