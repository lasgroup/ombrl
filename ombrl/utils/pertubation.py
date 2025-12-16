import jax.numpy as jnp
from typing import Callable
from jaxtyping import PyTree
import chex
import jax
from jaxrl.networks.common import Model
from maxinforl_jax.models import EnsembleState


@jax.jit
def perturb_params(init_params: PyTree,
                   trained_params: PyTree,
                   perturb_factor: chex.Array,
                   ):
    # if z = 1.0, we perturb the params, else z = 0.0 and we keep the trained params
    perturbation_fn = lambda x, y: (1 - perturb_factor) * x + perturb_factor * y
    new_params = jax.tree_util.tree_map(perturbation_fn,
                                        trained_params,
                                        init_params,
                                        )
    return new_params


class PerturbationModule(object):
    def __init__(self,
                 actor_init_fn: Callable,
                 critic_init_fn: Callable,
                 model_init_fn: Callable,
                 actor_init_opt_state: PyTree,
                 critic_init_opt_state: PyTree,
                 perturb_rate: float = 0.2,
                 perturbation_freq: int = int(2.5e6),
                 perturb_policy: bool = True,
                 perturb_model: bool = True,
                 ):
        self._actor_init_fn = jax.jit(actor_init_fn)
        self._critic_init_fn = jax.jit(critic_init_fn)
        self._model_init_fn = jax.jit(model_init_fn)
        self._actor_init_opt_state = actor_init_opt_state
        self._critic_init_opt_state = critic_init_opt_state
        self.perturb_rate = perturb_rate
        self.perturbation_freq = perturbation_freq
        self.perturb_policy = perturb_policy
        self.perturb_model = perturb_model

    def get_actor_init_params(self, rng: chex.Array, observation: chex.Array):
        return self._actor_init_fn(rng, observation)

    def get_model_init_params(self, rng: chex.Array, observation: chex.Array, action: chex.Array) -> EnsembleState:
        return self._model_init_fn(key=rng, input=jnp.concatenate([observation, action], axis=-1))

    def get_critic_init_params(self, rng: chex.Array, observation: chex.Array, action: chex.Array):
        return self._critic_init_fn(rng, observation, action)

    def perturb(self, actor: Model, critic: Model, target_actor: Model, target_critic: Model,
                ens_state: EnsembleState, observation: chex.Array, action: chex.Array, rng: chex.Array,
                step: int,
                ):
        perturb_factor = self.perturb_rate
        if step >= 1 and step % self.perturbation_freq == 0:
            print(f'resetting model at step: {step}')
            actor_rng, target_actor_rng, critic_rng, target_critic_rng, model_rng = jax.random.split(rng, 5)
            if self.perturb_policy:
                init_actor_params = self.get_actor_init_params(rng=actor_rng, observation=observation)
                init_actor_params = init_actor_params.pop('params')
                init_target_actor_params = self.get_actor_init_params(rng=target_actor_rng, observation=observation)
                init_target_actor_params = init_target_actor_params.pop('params')

                new_actor_params = perturb_params(
                    init_params=init_actor_params,
                    trained_params=actor.params,
                    perturb_factor=perturb_factor,
                )

                new_target_actor_params = perturb_params(
                    init_params=init_target_actor_params,
                    trained_params=target_actor.params,
                    perturb_factor=perturb_factor,
                )

                new_actor = actor.replace(
                    params=new_actor_params,
                    opt_state=self.actor_init_opt_state
                )

                new_target_actor = target_actor.replace(params=new_target_actor_params)
            else:
                new_actor = actor
                new_target_actor = target_actor

            init_critic_params = self.get_critic_init_params(rng=critic_rng, observation=observation, action=action)
            init_critic_params = init_critic_params.pop('params')
            init_target_critic_params = self.get_critic_init_params(rng=target_critic_rng,
                                                                    observation=observation, action=action)
            init_target_critic_params = init_target_critic_params.pop('params')

            new_critic_params = perturb_params(
                init_params=init_critic_params,
                trained_params=critic.params,
                perturb_factor=perturb_factor,
            )

            new_target_critic_params = perturb_params(
                init_params=init_target_critic_params,
                trained_params=target_critic.params,
                perturb_factor=perturb_factor,
            )

            new_critic = critic.replace(
                params=new_critic_params,
                opt_state=self.critic_init_opt_state
            )
            new_target_critic = target_critic.replace(params=new_target_critic_params)

            if self.perturb_model:
                new_ens_state = self.get_model_init_params(rng=model_rng, observation=observation, action=action)
                new_params = new_ens_state.vmapped_params
                old_params = ens_state.vmapped_params
                new_params = perturb_params(
                    init_params=new_params,
                    trained_params=old_params,
                    perturb_factor=perturb_factor,
                )
                # change params for ens state
                new_ens_state = ens_state.replace(vmapped_params=new_params, opt_state=new_ens_state.opt_state)
            else:
                new_ens_state = ens_state

            return new_actor, new_critic, new_target_actor, new_target_critic, new_ens_state
        else:
            return actor, critic, target_actor, target_critic, ens_state

    @property
    def actor_init_opt_state(self):
        return self._actor_init_opt_state

    @property
    def critic_init_opt_state(self):
        return self._critic_init_opt_state


class PolicyPerturbationModule(object):
    def __init__(self,
                 actor_init_fn: Callable,
                 actor_init_opt_state: PyTree,
                 perturb_rate: float = 0.2,
                 perturbation_freq: int = 10,
                 perturb_policy: bool = True,
                 ):
        self._actor_init_fn = jax.jit(actor_init_fn)
        self._actor_init_opt_state = actor_init_opt_state
        self.perturb_rate = perturb_rate
        self.perturbation_freq = perturbation_freq
        self.perturb_policy = perturb_policy

    def get_actor_init_params(self, rng: chex.Array, observation: chex.Array):
        return self._actor_init_fn(rng, observation)

    def perturb(self, actor: Model, target_actor: Model,
                observation: chex.Array, rng: chex.Array,
                episode: int,
                ):
        perturb_factor = self.perturb_rate
        if episode >= 1 and episode % self.perturbation_freq == 0:
            print(f'resetting policy at episode: {episode}')
            actor_rng, target_actor_rng = jax.random.split(rng, 2)
            if self.perturb_policy:
                init_actor_params = self.get_actor_init_params(rng=actor_rng, observation=observation)
                init_actor_params = init_actor_params.pop('params')
                init_target_actor_params = self.get_actor_init_params(rng=target_actor_rng, observation=observation)
                init_target_actor_params = init_target_actor_params.pop('params')

                new_actor_params = perturb_params(
                    init_params=init_actor_params,
                    trained_params=actor.params,
                    perturb_factor=perturb_factor,
                )

                new_target_actor_params = perturb_params(
                    init_params=init_target_actor_params,
                    trained_params=target_actor.params,
                    perturb_factor=perturb_factor,
                )

                new_actor = actor.replace(
                    params=new_actor_params,
                    opt_state=self.actor_init_opt_state
                )

                new_target_actor = target_actor.replace(params=new_target_actor_params)
            else:
                new_actor = actor
                new_target_actor = target_actor

            return new_actor, new_target_actor
        else:
            return actor, target_actor

    @property
    def actor_init_opt_state(self):
        return self._actor_init_opt_state


class CriticPerturbationModule(object):
    def __init__(self,
                 critic_init_fn: Callable,
                 critic_init_opt_state: PyTree,
                 perturb_rate: float = 0.2,
                 perturbation_freq: int = 10,
                 perturb_critic: bool = True,
                 ):
        self._critic_init_fn = jax.jit(critic_init_fn)
        self._critic_init_opt_state = critic_init_opt_state
        self.perturb_rate = perturb_rate
        self.perturbation_freq = perturbation_freq
        self.perturb_critic = perturb_critic

    def get_critic_init_params(self, rng: chex.Array, observation: chex.Array, action: chex.Array):
        return self._critic_init_fn(rng, observation, action)

    def perturb(self, critic: Model, target_critic: Model,
                observation: chex.Array, action: chex.Array, rng: chex.Array,
                episode: int,
                ):
        perturb_factor = self.perturb_rate
        if episode >= 1 and episode % self.perturbation_freq == 0:
            print(f'resetting critic at episode: {episode}')
            critic_rng, target_critic_rng = jax.random.split(rng, 2)

            if self.perturb_critic:
                init_critic_params = self.get_critic_init_params(rng=critic_rng, observation=observation, action=action)
                init_critic_params = init_critic_params.pop('params')
                init_target_critic_params = self.get_critic_init_params(rng=target_critic_rng,
                                                                        observation=observation, action=action)
                init_target_critic_params = init_target_critic_params.pop('params')

                new_critic_params = perturb_params(
                    init_params=init_critic_params,
                    trained_params=critic.params,
                    perturb_factor=perturb_factor,
                )

                new_target_critic_params = perturb_params(
                    init_params=init_target_critic_params,
                    trained_params=target_critic.params,
                    perturb_factor=perturb_factor,
                )

                new_critic = critic.replace(
                    params=new_critic_params,
                    opt_state=self.critic_init_opt_state
                )
                new_target_critic = target_critic.replace(params=new_target_critic_params)
            else:
                new_critic = critic
                new_target_critic = target_critic

            return new_critic, new_target_critic
        else:
            return critic, target_critic

    @property
    def critic_init_opt_state(self):
        return self._critic_init_opt_state
    

class ModelPerturbationModule(object):
    def __init__(self,
                 model_init_fn: Callable,
                 perturb_rate: float = 0.2,
                 perturbation_freq: int = 10,
                 perturb_model: bool = True,
                 ):
        self._model_init_fn = jax.jit(model_init_fn)
        self.perturb_rate = perturb_rate
        self.perturbation_freq = perturbation_freq
        self.perturb_model = perturb_model

    def get_model_init_params(self, rng: chex.Array, observation: chex.Array, action: chex.Array) -> EnsembleState:
        return self._model_init_fn(key=rng, input=jnp.concatenate([observation, action], axis=-1))

    def perturb(self, ens_state: EnsembleState, observation: chex.Array, action: chex.Array, rng: chex.Array,
                episode: int,
                ):
        perturb_factor = self.perturb_rate
        if episode >= 1 and episode % self.perturbation_freq == 0:
            print(f'resetting model at episode: {episode}')

            if self.perturb_model:
                new_ens_state = self.get_model_init_params(rng=rng, observation=observation, action=action)
                new_params = new_ens_state.vmapped_params
                old_params = ens_state.vmapped_params
                new_params = perturb_params(
                    init_params=new_params,
                    trained_params=old_params,
                    perturb_factor=perturb_factor,
                )
                # change params for ens state
                new_ens_state = ens_state.replace(vmapped_params=new_params, opt_state=new_ens_state.opt_state)
            else:
                new_ens_state = ens_state

            return new_ens_state
        else:
            return ens_state
