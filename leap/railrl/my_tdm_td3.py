from collections import OrderedDict

import numpy as np
import torch
from torch import optim
from railrl.state_distance.tdm import TemporalDifferenceModel


from railrl.my_td3 import MY_TD3


class MyTdmTd3(TemporalDifferenceModel, MY_TD3):
    def __init__(
            self,
            actor,
            critic1,
            critic2,
            max_action,
            args,
            env,
            exploration_policy,
            td3_kwargs,
            tdm_kwargs,
            base_kwargs,
            eval_policy=None,
            replay_buffer=None,
            optimizer_class=optim.Adam,
            use_policy_saturation_cost=False,
            **kwargs
    ):
        MY_TD3.__init__(
            self,
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            max_action=max_action,
            args=args,
            env=env,
            exploration_policy=exploration_policy,
            replay_buffer=replay_buffer,
            eval_policy=eval_policy,
            optimizer_class=optimizer_class,
            **td3_kwargs,
            **base_kwargs
        )
        self.qf1 = self.critic1
        super().__init__(**tdm_kwargs)
        self.use_policy_saturation_cost = use_policy_saturation_cost

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goals = batch['goals']
        num_steps_left = batch['num_steps_left']
        new_batch = {'obs':obs,
                     'goal':goals,
                     'rem_steps':num_steps_left,
                     'action':actions,
                     'next_obs':next_obs,
                     'reward':rewards,
                     'done':terminals}
        super(MyTdmTd3, self).own_train(new_batch)