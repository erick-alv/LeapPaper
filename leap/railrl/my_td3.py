import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from railrl.torch.torch_rl_algorithm import TorchRLAlgorithm
from railrl.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from railrl.policies.base import SerializablePolicy
from railrl.torch.core import PyTorchModule
import numpy as np
from torch.autograd import Variable
from railrl.my_logger import CSV_Logger
# This implementation is based on https://github.com/sfujim/TD3/blob/master/TD3.py; which is based on the mentioned
# paper. Additional modifications, done to make it as leap paper and temporal difference models #TODO mention paper
# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(PyTorchModule, SerializablePolicy):#implicit nn.Module
    def __init__(self, state_dim, action_dim, goal_dim, rem_steps_dim, max_action, device, reward_scale, networks_hidden):
        PyTorchModule.quick_init(self, locals())
        super(Actor, self).__init__()
        self.reward_scale = reward_scale
        index = 0
        self.l_in = nn.Linear(state_dim + goal_dim + rem_steps_dim, networks_hidden[index]).float().cuda()
        self.hidden_layers = []
        for _ in range(len(networks_hidden)-1):
            self.hidden_layers.append(nn.Linear(networks_hidden[index], networks_hidden[index + 1]).float().cuda())
            index += 1
        self.l3 = nn.Linear(networks_hidden[index], action_dim).float().cuda()
        self.max_action = torch.FloatTensor(max_action).float().cuda()

    def forward(self, state, goal, rem_steps):
        if len(state.shape) > 1:
            a = torch.cat([state, goal, rem_steps], dim=1)
        else:
            a = torch.cat([state, goal, rem_steps])
        a = F.relu(self.l_in(a))
        for layer in self.hidden_layers:
            a = F.relu(layer(a))
        return Variable(self.max_action) * torch.tanh(self.l3(a))

    def get_action(self, state, goal, rem_steps):
        def transform_to_valid_tensor(value):
            if not isinstance(value, Variable):
                if isinstance(value, np.ndarray):
                    if value.dtype != np.float32:
                        value = value.astype(np.float32)
                    return Variable(torch.torch.from_numpy(value).cuda())
                else:
                    return Variable(torch.FloatTensor(value).cuda())
            else:
                return value
        state = transform_to_valid_tensor(state)
        goal = transform_to_valid_tensor(goal)
        rem_steps = transform_to_valid_tensor(rem_steps)
        actions = self.forward(state, goal, rem_steps)
        return actions.data.cpu().numpy(), {}


class Critic(PyTorchModule):#implicit nn.Module
    def __init__(self, state_dim, action_dim, goal_dim, rem_steps_dim, value_dim, device,networks_hidden):
        PyTorchModule.quick_init(self, locals())
        super(Critic, self).__init__()

        # Q1 architecture
        index = 0
        self.l1 = nn.Linear(state_dim + action_dim + goal_dim + rem_steps_dim, networks_hidden[index]).float().cuda()
        self.hidden_layers1 = []
        for _ in range(len(networks_hidden)-1):
            self.hidden_layers1.append(nn.Linear(networks_hidden[index], networks_hidden[index + 1]).float().cuda())
            index += 1
        self.l3 = nn.Linear(networks_hidden[index], value_dim).float().cuda()


    def forward(self, state, action, goal, rem_steps):
        sa = torch.cat([state, action, goal, rem_steps], 1)

        q = F.relu(self.l1(sa))
        for layer in self.hidden_layers1:
            q = F.relu(layer(q))
        q = self.l3(q)
        return q



class MY_TD3(TorchRLAlgorithm):
    def __init__(
            self,
            actor,
            critic1,
            critic2,
            max_action,
            args,
            env,#just to match
            reward_scale,
            exploration_policy,#just to match
            eval_policy = None,#just to match
            networks_hidden = [400, 300],
            discount=1.0,
            tau=0.005,
            policy_noise=0.2,#todo this better in wrapper and pass as arguiemnt
            noise_clip=0.5,
            policy_freq=2,
            lr = 0.001,#todo this better in wrapper
            **kwargs#just to match

    ):

        self.actor = actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic1 = critic1
        self.critic2 = critic2
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        self.reward_scale = reward_scale
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0
        self.args = args

        #just to match
        self.exploration_policy = exploration_policy
        super().__init__(
            env,
            exploration_policy,
            eval_policy=eval_policy or self.actor,
            **kwargs
        )
        self.my_log = CSV_Logger(fieldnames=['actor_loss', 'critic1_loss','critic2_loss'], args={},
                                 iteration_fieldnames=['epoch','step'])
        self.step = 0

    '''def get_action(self, state, goal, rem_steps):
        if not torch.is_tensor(state):
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float).to(self.args['device'])
        if not torch.is_tensor(goal):
            goal = torch.tensor(goal.reshape(1, -1), dtype=torch.float).to(self.args['device'])
        if not torch.is_tensor(rem_steps):
            if self.rem_steps_dim == 1:
                rem_steps = torch.tensor([[rem_steps]], dtype=torch.float).to(self.args['device'])
            else:
                rem_steps = torch.tensor(rem_steps.reshape(1, -1), dtype=torch.float).to(self.args['device'])
        return self.actor(state, goal, rem_steps).cpu().data.numpy()

    def get_Q_val(self, state, action, goal, rem_steps):
        if not torch.is_tensor(state):
            state = torch.tensor(state.reshape(1, -1), dtype=torch.float).to(self.args['device'])
        if not torch.is_tensor(action):
            action = torch.tensor(action.reshape(1, -1), dtype=torch.float).to(self.args['device'])
        if not torch.is_tensor(goal):
            goal = torch.tensor(goal.reshape(1, -1), dtype=torch.float).to(self.args['device'])
        if not torch.is_tensor(rem_steps):
            if self.rem_steps_dim == 1:
                rem_steps = torch.tensor([[rem_steps]], dtype=torch.float).to(self.args['device'])
            else:
                rem_steps = torch.tensor(rem_steps.reshape(1, -1), dtype=torch.float).to(self.args['device'])
        return self.critic.Q1(state, action, goal, rem_steps).cpu().data.numpy()'''

    '''def clip_actions(self, actions):
        #from https://stackoverflow.com/questions/54738045/column-dependent-bounds-in-torch-clamp
        l = torch.tensor([-self.max_action], dtype=torch.float).to(self.args['device'])
        u = torch.tensor([self.max_action], dtype=torch.float).to(self.args['device'])
        a = torch.max(actions.float(), l)
        return torch.min(a, u)'''


    def own_train(self, batch):
        self.actor.train()
        self.actor_target.train()
        self.critic1.train()
        self.critic1_target.train()
        self.critic2.train()
        self.critic2_target.train()
        self.total_it += 1

        # Sample replay buffer
        state, goal, rem_steps, action, next_state, reward, done = (batch[k] for k in ['obs', 'goal', 'rem_steps',
                                                                                       'action', 'next_obs', 'reward',
                                                                                       'done'])
        #with torch.no_grad():
        # Select action according to policy and add clipped noise
        mean = torch.FloatTensor([[0]*action.shape[1]] * action.shape[0]).cuda()
        std = torch.FloatTensor([[1]*action.shape[1]] * action.shape[0]).cuda()
        random = torch.normal(mean, std)
        noise = Variable((random * self.policy_noise).clamp(-self.noise_clip, self.noise_clip))

        next_action = self.actor_target(next_state, goal, rem_steps-1.)
        next_action = next_action + noise
        next_action = torch.clamp(next_action, min=float(-self.max_action[0]), max=float(self.max_action[0]))

        # Compute the target Q value
        target_Q1 = self.critic1_target(next_state, next_action, goal, rem_steps-1.)
        target_Q2 = self.critic2_target(next_state, next_action, goal, rem_steps-1.)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = self.reward_scale * reward + (1. - done) * self.discount * target_Q
        #detaching
        target_Q = target_Q.detach()



        # Get current Q estimates
        current_Q1 = self.critic1(state, action, goal, rem_steps)
        current_Q2 = self.critic2(state, action, goal, rem_steps)

        # Compute critic loss
        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losses
            #todo leap paper has option policy_saturation_cost = F.relu(torch.abs(pre_tanh_value) - 20.0),
            # but not used!!!!!!
            actor_loss = -self.critic1(state, self.actor(state, goal, rem_steps).float(), goal, rem_steps).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


            self.my_log.add_log('actor_loss', float(actor_loss))
            self.my_log.add_log('critic1_loss', float(critic1_loss))
            self.my_log.add_log('critic2_loss', float(critic2_loss))
            self.my_log.finish_step_log(self.step)
            self.my_log.finish_epoch_log(self.step)#just to save
            self.step += 1
            return float(critic1_loss), float(critic2_loss), float(actor_loss)

        else:
            self.my_log.add_log('actor_loss', None)
            self.my_log.add_log('critic1_loss', float(critic1_loss))
            self.my_log.add_log('critic2_loss', float(critic2_loss))
            self.my_log.finish_step_log(self.step)
            self.my_log.finish_epoch_log(self.step)  # just to save
            self.step += 1
            return float(critic1_loss), float(critic2_loss), None

    '''def evaluate(self, batch):
        self.actor.eval()
        self.actor_target.eval()
        self.critic1.eval()
        self.critic1_target.eval()
        # Sample replay buffer
        state, goal, rem_steps, action, next_state, reward, done = (batch[k] for k in ['obs', 'goal', 'rem_steps',
                                                                                       'action', 'next_obs', 'reward',
                                                                                       'done'])

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = self.clip_actions(
                self.actor_target(next_state, goal, rem_steps - 1.) + noise
            )

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action, goal, rem_steps - 1.)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1. - done) * self.discount * target_Q
            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action, goal, rem_steps)

            # Compute critic loss
            # todo LEAP paper optimizes them separately
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2,target_Q)
            # Compute actor losses
            # todo leap paper additionally uses policy_saturation_cost = F.relu(torch.abs(pre_tanh_value) - 20.0) but not used
            actor_loss = -self.critic.Q1(state, self.actor(state, goal, rem_steps).float(), goal, rem_steps).mean()

            return float(critic_loss), float(actor_loss)'''

    def save(self, filename):
        save_dict = {
            'critic1': self.critic1.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic1': self.critic2.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'actor': self.actor.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict()
        }
        path = self.args.dirpath + 'weights_dir/'+filename
        torch.save(save_dict, path)

    def save_train_checkpoint(self, filename, epoch):
        if not filename.endswith(str(epoch)):
            filename = filename + '_' + str(epoch)
        self.save(filename)

    def load(self, filename):
        path = self.args.dirpath + 'weights_dir/' + filename
        save_dict = torch.load(path)
        self.critic1.load_state_dict(save_dict['critic1'])
        self.critic1_optimizer.load_state_dict(save_dict['critic_optimizer1'])
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2.load_state_dict(save_dict['critic2'])
        self.critic2_optimizer.load_state_dict(save_dict['critic2_optimizer'])
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor.load_state_dict(save_dict['actor'])
        self.actor_optimizer.load_state_dict(save_dict['actor_optimizer'])
        self.actor_target = copy.deepcopy(self.actor)

    def load_train_checkpoint(self, filename, epoch):
        if not filename.endswith(str(epoch)):
            filename = filename + '_' + str(epoch)
        self.load(filename)

    #follwing methods are just to match
    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        raise NotImplementedError

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        self.update_epoch_snapshot(snapshot)
        return snapshot

    def update_epoch_snapshot(self, snapshot):
        snapshot.update(
            actor=self.actor,
            critic1=self.critic1,
            critic2=self.critic2,
            actor_target=self.actor_target,
            critic1_target=self.critic1_target,
            critic2_target=self.critic2_target,
            exploration_policy=self.exploration_policy,
        )

    @property
    def networks(self):
        return [
            self.actor,
            self.critic1,
            self.critic2,
            self.actor_target,
            self.critic1_target,
            self.critic2_target
        ]