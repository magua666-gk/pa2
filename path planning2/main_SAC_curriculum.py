# ==============================================================================
# 模块: 依赖导入
# ==============================================================================
import argparse
import numpy as np
import os
import sys
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import json
import uuid
import random
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import pickle as pkl
from typing import Dict, Any
# Import environment
from rl_env.path_env import RlGame
# Import network components
from masac_adapter.masac_adapter import MASACEntroy,set_log_level,  max_action, min_action, log, LOG_INFO, LOG_WARNING, LOG_DEBUG, LOG_ERROR, clear_log_history
from main_SAC import Ornstein_Uhlenbeck_Noise
from masac_adapter.smer_memory import SMERMemory
# Import new Actor and Critic networks
from masac_adapter.actor_networks import LeaderActorNet, FollowerActorNet, AttentionLeaderActorNet, AttentionFollowerActorNet
from masac_adapter.critic_networks import StructuredAttentionCriticNet
# Import curriculum learning manager
from curriculum import CurriculumManager, FixedTaskGenerator, CurriculumConfig, LinearTaskSequencer,PolicyTransfer
# ==============================================================================
# 模块: 可视化字体配置
# ==============================================================================
def configure_matplotlib_fonts():
    """Configure matplotlib to use a CJK-capable font when available."""
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "SimSun",
        "Arial Unicode MS"
    ]
    available_fonts = {font.name for font in fm.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False
configure_matplotlib_fonts()
# ==============================================================================
# 模块: 随机种子工具
# ==============================================================================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
# ==============================================================================
# 模块: MASAC 控制器核心
# ==============================================================================
class MASACController:
    """Role-based Multi-Agent SAC Controller"""
    '''这个类管理所有网络、经验池、训练逻辑'''

    def __init__(self, n_agents=1, state_dim=17, action_dim=4, memory_size=int(2e6),
                 batch_size=256, gamma=0.99, tau=0.01, value_lr=3e-4, policy_lr=1e-4,
                 hidden_dim=256, target_update_interval=2, reward_scale=0.1,
                 auto_entropy=True, entropy_lr=3e-4, target_entropy=-0.1, device=None,
                 memory_capacity=None, max_replay_ratio=10.0, share_follower_policy=False, use_attention=True,
                 use_gat=False):  # Add max_replay_ratio parameter
        """Initialize role-based MASAC controller
        
        Args:
            n_agents: Number of agents
            state_dim: State dimension per agent
            action_dim: Action dimension per agent
            memory_size: Replay buffer size (new parameter name)
            memory_capacity: Replay buffer capacity (old parameter name, backward compatible)
            batch_size: Training batch size
            gamma: Discount factor
            tau: Soft update coefficient
            value_lr: Critic learning rate
            policy_lr: Actor learning rate
            hidden_dim: Hidden layer dimension隐藏层维度
            target_update_interval: Target network update interval
            reward_scale: Reward scaling factor
            auto_entropy: Whether to auto-adjust entropy是否自动调整熵
            entropy_lr: Entropy adjustment learning rate
            target_entropy: Target entropy value
            device: Training device
            max_replay_ratio: Maximum replay ratio, limiting upper bound for set_replay_ratio
            share_follower_policy: Whether all followers share the same actor policy
            use_attention: Whether to enable attention-based actor/critic modules
        """
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.reward_scale = reward_scale
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.entropy_lr = entropy_lr # Save entropy_lr as instance attribute
        self.policy_lr = policy_lr
        self.share_follower_policy = bool(share_follower_policy)
        self.use_attention = bool(use_attention)
        self.use_gat = bool(use_gat)

        # Save actor architecture config for dynamic follower expansion.
        self.actor_embed_dim = 128
        self.actor_hidden_dims = [256, 128]
        self.actor_n_heads = 4
        self.actor_dropout = 0.1
        self.actor_use_shared_layer = True
        
        # Handle memory size parameter (backward compatible)
        if memory_capacity is not None:
            print(f"Warning: Deprecated parameter 'memory_capacity' used, please use 'memory_size' instead")
            self.memory_capacity = memory_capacity
            memory_size = memory_capacity
        else:
            self.memory_capacity = memory_size  # Save copy for reset_memory method
        
        attention_desc = "with attention" if self.use_attention else "without attention"
        print(f"Initializing MASAC controller ({attention_desc}): {n_agents} agents, state_dim={state_dim}/agent, action_dim={action_dim}/agent")
        print(f"Using device: {self.device}")
        
        # Initialize experience replay buffer
        obs_dims = {"leader": state_dim, "followers": state_dim}
        action_dims = {"leader": action_dim, "followers": action_dim}
        
        self.memory = SMERMemory(
            capacity=memory_size,
            obs_dims=obs_dims,
            action_dims=action_dims,
            device=self.device
        )
        
        # Record current agent count version for replay buffer
        self.memory_n_agents_version = n_agents
        
        # === Actor network hyperparameters ===
        embed_dim = self.actor_embed_dim  # Embedding dimension
        actor_hidden_dims = self.actor_hidden_dims  # Actor hidden layer dimensions
        actor_n_heads = self.actor_n_heads  # Number of attention heads
        actor_dropout = self.actor_dropout  # Dropout probability
        use_shared_layer = self.actor_use_shared_layer  # Whether to use shared layer for fusion features

        # === Create Leader/Follower Actor networks ===
        if self.use_attention:
            self.leader_actor = AttentionLeaderActorNet(
                state_dim=state_dim, action_dim=action_dim, embed_dim=embed_dim,
                hidden_dims=actor_hidden_dims, n_heads=actor_n_heads, dropout=actor_dropout,
                use_shared_layer=use_shared_layer, use_gat=self.use_gat
            )
            self.target_leader_actor = AttentionLeaderActorNet(
                state_dim=state_dim, action_dim=action_dim, embed_dim=embed_dim,
                hidden_dims=actor_hidden_dims, n_heads=actor_n_heads, dropout=actor_dropout,
                use_shared_layer=use_shared_layer, use_gat=self.use_gat
            )

            def _create_follower_actor_instance():
                return AttentionFollowerActorNet(
                    state_dim=state_dim, action_dim=action_dim, embed_dim=embed_dim,
                    hidden_dims=actor_hidden_dims, n_heads=actor_n_heads, dropout=actor_dropout,
                    use_shared_layer=use_shared_layer, use_gat=self.use_gat
                )
        else:
            self.leader_actor = LeaderActorNet(
                state_dim=state_dim,
                action_dim=action_dim,
                embed_dim=embed_dim,
                hidden_dims=actor_hidden_dims
            )
            self.target_leader_actor = LeaderActorNet(
                state_dim=state_dim,
                action_dim=action_dim,
                embed_dim=embed_dim,
                hidden_dims=actor_hidden_dims
            )

            def _create_follower_actor_instance():
                return FollowerActorNet(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    embed_dim=embed_dim,
                    hidden_dims=actor_hidden_dims
                )
        
        # Load initial target Leader Actor parameters
        self.target_leader_actor.load_state_dict(self.leader_actor.state_dict())

        # Determine maximum number of followers to create
        max_followers = max(n_agents - 1, 3)  # Support at least 3 followers, or based on initial n_agents
        
        # Create separate Actor network for each possible follower
        self.follower_actors = nn.ModuleList([
            _create_follower_actor_instance() for _ in range(max_followers)
        ])
        
        # Create separate target Actor network for each possible follower
        self.target_follower_actors = nn.ModuleList([
            _create_follower_actor_instance() for _ in range(max_followers)
        ])
        
        # Load initial target Follower Actor parameters
        for i in range(max_followers):
            self.target_follower_actors[i].load_state_dict(self.follower_actors[i].state_dict())

        if self.share_follower_policy:
            print("Follower policy sharing enabled: all followers will reuse follower actor #0.")
            self._sync_shared_follower_policy_weights()
            
        # Old LeaderActorNet and FollowerActorNet code (commented out)
        """
        # Leader Actor network
        self.leader_actor = LeaderActorNet(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            hidden_dims=actor_hidden_dims
        )
        
        # Follower Actor network
        self.follower_actor = FollowerActorNet(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            hidden_dims=actor_hidden_dims
        )
        
        # Target Actor networks
        self.target_leader_actor = LeaderActorNet(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            hidden_dims=actor_hidden_dims
        )
        
        self.target_follower_actor = FollowerActorNet(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            hidden_dims=actor_hidden_dims
        )
        
        # Load initial target Actor parameters
        self.target_leader_actor.load_state_dict(self.leader_actor.state_dict())
        self.target_follower_actor.load_state_dict(self.follower_actor.state_dict())
        """
        
        # === Create Critic network ===
        critic_hidden_dims = [256, 128]  # Critic hidden layer dimensions
        n_heads = 4  # Number of attention heads
        dropout = 0.1  # Dropout probability

        self.critic = StructuredAttentionCriticNet(
            state_dim=state_dim, action_dim=action_dim, embed_dim=embed_dim,
            n_heads=n_heads, hidden_dims=critic_hidden_dims, dropout=dropout,
            use_attention=self.use_attention, use_gat=self.use_gat
        )
        
        # === Initialize entropy adjustment ===
        # Leader and Follower each have an entropy parameter
        self.entroy_leader = MASACEntroy(action_dim=action_dim)
        self.entroy_follower = MASACEntroy(action_dim=action_dim)

        # Set target entropy
        if target_entropy < 0:
            self.entroy_leader.target_entropy = -0.1
            self.entroy_follower.target_entropy = -0.1
        else:
            self.entroy_leader.target_entropy = target_entropy
            self.entroy_follower.target_entropy = target_entropy
        
        # === Initialize optimizers ===
        # Leader Actor optimizer优化器使用 Adam 优化算法
        self.leader_actor_optimizer = optim.Adam(self.leader_actor.parameters(), lr=policy_lr)
        
        # Follower Actors optimizer
        self.follower_actor_optimizer = None
        self._rebuild_follower_actor_optimizer()
        
        # Critic optimizer (includes all Critic components)
        critic_params = []
        critic_params.extend(list(self.critic.leader_encoder.parameters()))
        critic_params.extend(list(self.critic.follower_encoder.parameters()))
        # Update: Use renamed leader_sees_followers_attention
        critic_params.extend(list(self.critic.leader_sees_followers_attention.parameters()))
        # Add: New follower_context_attention
        critic_params.extend(list(self.critic.follower_context_attention.parameters()))
        critic_params.extend(list(self.critic.leader_q_head.parameters()))
        critic_params.extend(list(self.critic.follower_q_head.parameters()))
        self.critic_optimizer = optim.Adam(critic_params, lr=value_lr)
        
        # Entropy parameter optimizers
        self.leader_alpha_optimizer = optim.Adam([self.entroy_leader.log_alpha], lr=entropy_lr)
        self.follower_alpha_optimizer = optim.Adam([self.entroy_follower.log_alpha], lr=entropy_lr)
        
        # Initialize noise generators
        self.noises = [Ornstein_Uhlenbeck_Noise(mu=np.zeros(action_dim)) for _ in range(n_agents)]
        
        # Move to specified device
        self.to(self.device)
        
        # Record training state
        self.train_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.last_train_status = {
            "updated": False,
            "reason": "not_started",
            "stage_tag": None,
            "stage_number": None,
            "batch_size": None,
            "train_step": 0
        }
        
        # Initialize replay ratio parameters
        self.min_replay_ratio = 0.1  # Minimum replay ratio
        self.max_replay_ratio = max_replay_ratio  # Maximum replay ratio
        self.replay_ratio = 1.0  # Current replay ratio
        
        # Initialize training step counter
        self.steps_done = 0
    
    def set_replay_ratio(self, ratio):
        """Set replay ratio
        
        Args:
            ratio: New replay ratio
        """
        if ratio < self.min_replay_ratio:
            ratio = self.min_replay_ratio
            print(f"Replay ratio too small, set to minimum value {self.min_replay_ratio}")
        elif ratio > self.max_replay_ratio:
            ratio = self.max_replay_ratio
            print(f"Replay ratio too large, set to maximum value {self.max_replay_ratio}")
            
        old_ratio = self.replay_ratio
        self.replay_ratio = ratio
        print(f"Replay ratio adjusted from {old_ratio} to {self.replay_ratio}")
    
    def reset_memory(self):
        """Reset replay buffer"""
        obs_dims = {"leader": self.state_dim, "followers": self.state_dim}
        action_dims = {"leader": self.action_dim, "followers": self.action_dim}
        
        self.memory = SMERMemory(
            capacity=self.memory_capacity,
            obs_dims=obs_dims,
            action_dims=action_dims,
            device=self.device
        )
        print(f"Reset SMERMemory buffer, capacity: {self.memory_capacity}")
        
        self._reset_noise()
    
    def _reset_noise(self):
        """Reset noise generators"""
        # Ensure sufficient noise generators
        if len(self.noises) < self.n_agents:
            # Add new noise generators
            while len(self.noises) < self.n_agents:
                self.noises.append(Ornstein_Uhlenbeck_Noise(mu=np.zeros(self.action_dim)))
        else:
            # Reduce noise generators
            self.noises = self.noises[:self.n_agents]
            
        # Reset all noise generator states
        for noise in self.noises:
            noise.reset()

    def _build_follower_actor(self):
        """Create a follower actor instance using current architecture settings."""
        if self.use_attention:
            return AttentionFollowerActorNet(
                state_dim=self.state_dim, action_dim=self.action_dim, embed_dim=self.actor_embed_dim,
                hidden_dims=self.actor_hidden_dims, n_heads=self.actor_n_heads, dropout=self.actor_dropout,
                use_shared_layer=self.actor_use_shared_layer, use_gat=self.use_gat
            )

        return FollowerActorNet(
            state_dim=self.state_dim, action_dim=self.action_dim,
            embed_dim=self.actor_embed_dim, hidden_dims=self.actor_hidden_dims
        )

    def _resolve_follower_actor_index(self, follower_idx):
        """Map follower slot to actor index, optionally sharing a single actor."""
        if self.share_follower_policy:
            return 0
        return follower_idx

    def _get_follower_actor(self, follower_idx, target=False):
        """Get follower actor module and resolved actor index for a follower slot."""
        actor_bank = self.target_follower_actors if target else self.follower_actors
        if len(actor_bank) == 0:
            raise RuntimeError("Follower actor bank is empty")

        actor_idx = self._resolve_follower_actor_index(follower_idx)
        actor_idx = min(max(actor_idx, 0), len(actor_bank) - 1)
        return actor_bank[actor_idx], actor_idx

    def _sync_shared_follower_policy_weights(self):
        """Keep follower actor banks synchronized when sharing one follower policy."""
        if not self.share_follower_policy:
            return
        if len(self.follower_actors) <= 1:
            return

        source_actor_state = self.follower_actors[0].state_dict()
        for i in range(1, len(self.follower_actors)):
            self.follower_actors[i].load_state_dict(source_actor_state, strict=False)

        if len(self.target_follower_actors) > 0:
            source_target_state = self.target_follower_actors[0].state_dict()
            for i in range(1, len(self.target_follower_actors)):
                self.target_follower_actors[i].load_state_dict(source_target_state, strict=False)

    def _rebuild_follower_actor_optimizer(self):
        """Rebuild follower optimizer so newly added follower networks can be trained."""
        follower_actor_parameters = []
        if self.share_follower_policy and len(self.follower_actors) > 0:
            follower_actor_parameters.extend(list(self.follower_actors[0].parameters()))
        else:
            for actor_instance in self.follower_actors:
                follower_actor_parameters.extend(list(actor_instance.parameters()))

        current_lr = self.policy_lr
        if hasattr(self, 'follower_actor_optimizer') and self.follower_actor_optimizer is not None:
            if self.follower_actor_optimizer.param_groups:
                current_lr = self.follower_actor_optimizer.param_groups[0].get('lr', self.policy_lr)

        self.follower_actor_optimizer = optim.Adam(follower_actor_parameters, lr=current_lr)

    def _ensure_follower_capacity(self, required_followers):
        """Expand follower actor banks when runtime follower count exceeds current capacity."""
        current_followers = len(self.follower_actors)
        if required_followers <= current_followers:
            return

        to_add = required_followers - current_followers
        print(f"Expanding follower actor capacity: {current_followers} -> {required_followers}")

        if current_followers > 0:
            source_idx = 0 if self.share_follower_policy else -1
            source_actor_state = self.follower_actors[source_idx].state_dict()
            source_target_state = self.target_follower_actors[source_idx].state_dict()
        else:
            source_actor_state = None
            source_target_state = None

        for _ in range(to_add):
            new_actor = self._build_follower_actor().to(self.device)
            new_target_actor = self._build_follower_actor().to(self.device)

            if source_actor_state is not None:
                new_actor.load_state_dict(source_actor_state, strict=False)
            if source_target_state is not None:
                new_target_actor.load_state_dict(source_target_state, strict=False)
            else:
                new_target_actor.load_state_dict(new_actor.state_dict())

            self.follower_actors.append(new_actor)
            self.target_follower_actors.append(new_target_actor)

        self._rebuild_follower_actor_optimizer()
        self._sync_shared_follower_policy_weights()
        print(f"Follower actor optimizer rebuilt for {len(self.follower_actors)} follower networks")

    def _safe_load_optimizer_state(self, optimizer, optimizer_state, optimizer_name):
        """Safely load optimizer states across architecture changes."""
        try:
            optimizer.load_state_dict(optimizer_state)
            return True
        except ValueError as e:
            print(f"Warning: Failed to load optimizer state for {optimizer_name}: {e}. Using fresh optimizer state.")
            return False
    
    def adapt_to_agent_count(self, n_agents):
        """Adapt to new agent count
        
        Args:
            n_agents: New agent count
        """
        if n_agents == self.n_agents:
            return
                
        print(f"Adapting to agent count change: {self.n_agents} -> {n_agents}")
        
        self.n_agents = n_agents
        num_followers = max(n_agents - 1, 0)
        self._ensure_follower_capacity(num_followers)
        self._sync_shared_follower_policy_weights()
        
        self._reset_noise()
        self.memory_n_agents_version = n_agents
            
        print(f"Successfully adjusted to {n_agents} agents (1 Leader + {num_followers} Followers)")
    
    def to(self, device):
        """Move all networks to specified device
        
        Args:
            device: Target device
            
        Returns:
            self: Support chaining
        """
        self.device = device
        
        self.leader_actor = self.leader_actor.to(device)
        self.follower_actors = self.follower_actors.to(device)
        
        self.target_leader_actor = self.target_leader_actor.to(device)
        self.target_follower_actors = self.target_follower_actors.to(device)
        
        self.critic = self.critic.to(device)
        if hasattr(self.entroy_leader, 'log_alpha') and torch.is_tensor(self.entroy_leader.log_alpha):
            self.entroy_leader.log_alpha = self.entroy_leader.log_alpha.to(device)
            self.entroy_leader.alpha = self.entroy_leader.alpha.to(device)
        if hasattr(self.entroy_follower, 'log_alpha') and torch.is_tensor(self.entroy_follower.log_alpha):
            self.entroy_follower.log_alpha = self.entroy_follower.log_alpha.to(device)
            self.entroy_follower.alpha = self.entroy_follower.alpha.to(device)
        
        # Move optimizer state
        optimizers_to_move = [
            self.leader_actor_optimizer,
            self.follower_actor_optimizer,
            self.critic_optimizer,
            self.leader_alpha_optimizer,
            self.follower_alpha_optimizer
        ]
        
        for opt in optimizers_to_move:
            if opt is not None:
                for state in opt.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
        
        print(f"MASAC controller (including optimizer state) moved to device: {device}")
        return self

    def set_eval_mode(self):
        """Switch all inference-related networks to eval mode."""
        self.leader_actor.eval()
        self.target_leader_actor.eval()
        self.critic.eval()

        for actor in self.follower_actors:
            actor.eval()

        for actor in self.target_follower_actors:
            actor.eval()

        return self
    
    def select_actions(self, observation, add_noise=False, noise_scale=0.1, evaluate=False):
        """Select actions for all agents
        
        Args:
            observation: Structured observation from environment {"leader": obs_leader, "followers": [obs_f1, obs_f2, ...]}
                or legacy flattened state list/array (n_agents, state_dim) or (state_dim * n_agents,)
            add_noise: Whether to add exploration noise
            noise_scale: Noise scaling factor
            evaluate: Whether in evaluation mode
            
        Returns:
            actions: Structured action dict {"leader": action_leader, "followers": [action_f1, action_f2, ...]}
        """
        # Check if input is structured format
        if isinstance(observation, dict) and "leader" in observation and "followers" in observation:
            # Handle structured input
            leader_obs = observation["leader"]
            follower_obs_list = observation["followers"]
            
            # Ensure they are numpy arrays
            leader_obs = np.nan_to_num(np.array(leader_obs, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            follower_obs_list = [
                np.nan_to_num(np.array(obs, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                for obs in follower_obs_list
            ]
            
            # Create numpy array and mask for follower observations
            num_followers = len(follower_obs_list)
            expected_n_agents = 1 + num_followers
            if expected_n_agents != self.n_agents:
                log(f"Detected agent count change: controller={self.n_agents}, input_state={expected_n_agents}", LOG_INFO)
                self.adapt_to_agent_count(expected_n_agents)
            else:
                self._ensure_follower_capacity(num_followers)

            if num_followers > 0:
                # Create follower observation array [1, num_followers, state_dim]
                followers_obs_array = np.stack(follower_obs_list).reshape(1, num_followers, -1)
                # Create follower mask [1, num_followers]
                followers_mask = np.ones((1, num_followers), dtype=bool)
            else:
                # If no followers, create empty array
                followers_obs_array = np.zeros((1, 0, self.state_dim), dtype=np.float32)
                followers_mask = np.zeros((1, 0), dtype=bool)
            
            try:
                if isinstance(self.leader_actor, AttentionLeaderActorNet):
                    leader_action = self.leader_actor.choose_action(
                        leader_obs, 
                        followers_obs_array, 
                        followers_mask,
                        evaluate=evaluate
                    )
                else:
                    leader_action = self.leader_actor.choose_action(leader_obs, evaluate=evaluate)
            except Exception as e:
                log(f"Error selecting Leader action: {e}", LOG_ERROR)
                leader_action = np.zeros(self.action_dim)
            
            if add_noise and not evaluate:
                noise = self.noises[0]() * noise_scale
                leader_action += noise
                leader_action = np.clip(leader_action, min_action, max_action)

            leader_action = np.nan_to_num(
                np.asarray(leader_action, dtype=np.float32),
                nan=0.0,
                posinf=max_action,
                neginf=min_action
            )
            leader_action = np.clip(leader_action, min_action, max_action)
            
            follower_actions = []
            for i, follower_obs in enumerate(follower_obs_list):
                if i >= len(self.noises):
                    self.noises.append(Ornstein_Uhlenbeck_Noise(mu=np.zeros(self.action_dim)))

                follower_actor, resolved_actor_idx = self._get_follower_actor(i, target=False)
                
                try:
                    # If AttentionFollowerActorNet, need to provide context information
                    if isinstance(follower_actor, AttentionFollowerActorNet):
                        # Prepare Leader context observations
                        leader_obs_for_context = leader_obs.reshape(1, -1)  # [1, state_dim]
                        leader_mask = np.ones((1, 1), dtype=bool)
                        
                        other_followers_indices = [j for j in range(num_followers) if j != i]
                        
                        if other_followers_indices:
                            other_followers_obs = np.stack([follower_obs_list[j] for j in other_followers_indices])
                            other_followers_obs = other_followers_obs.reshape(1, len(other_followers_indices), -1)
                            other_followers_mask = np.ones((1, len(other_followers_indices)), dtype=bool)
                        else:
                            other_followers_obs = np.zeros((1, 0, self.state_dim), dtype=np.float32)
                            other_followers_mask = np.zeros((1, 0), dtype=bool)
                        
                        follower_action = follower_actor.choose_action(
                            follower_obs,
                            leader_obs_for_context,
                            other_followers_obs,
                            leader_mask,
                            other_followers_mask,
                            evaluate=evaluate
                        )
                    else:
                        follower_action = follower_actor.choose_action(follower_obs, evaluate=evaluate)
                except Exception as e:
                    log(f"Error selecting Follower {i} action (actor #{resolved_actor_idx}): {e}", LOG_ERROR)
                    follower_action = np.zeros(self.action_dim)
                
                # Add exploration noise
                if add_noise and not evaluate:
                    noise = self.noises[i+1]() * noise_scale
                    follower_action += noise
                    follower_action = np.clip(follower_action, min_action, max_action)

                follower_action = np.nan_to_num(
                    np.asarray(follower_action, dtype=np.float32),
                    nan=0.0,
                    posinf=max_action,
                    neginf=min_action
                )
                follower_action = np.clip(follower_action, min_action, max_action)
                
                follower_actions.append(follower_action)
            
            # Return structured action dict
            return {
                "leader": leader_action,
                "followers": follower_actions
            }
            
        else:
            # Legacy flattened input, maintain compatibility
            # Ensure states is numpy array
            states = observation
            if not isinstance(states, np.ndarray):
                states = np.array(states)
            states = np.nan_to_num(states.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            
            # Handle different input shapes
            if len(states.shape) == 1:
                # Single vector contains all agent states
                # Infer agent count
                inferred_n_agents = states.shape[0] // self.state_dim
                if inferred_n_agents * self.state_dim != states.shape[0]:
                    log(f"Warning: State dimension {states.shape[0]} is not an integer multiple of state_dim {self.state_dim}", LOG_WARNING)
                    inferred_n_agents = max(1, states.shape[0] // self.state_dim)
                    
                if inferred_n_agents != self.n_agents:
                    log(f"Detected agent count change: controller={self.n_agents}, input state={inferred_n_agents}", LOG_INFO)
                    self.adapt_to_agent_count(inferred_n_agents)
                    
                # Split into n_agents individual agent states
                states_list = []
                for i in range(self.n_agents):
                    if i * self.state_dim < states.shape[0]:
                        agent_state = states[i*self.state_dim:(i+1)*self.state_dim]
                        # Ensure correct state dimension
                        if len(agent_state) < self.state_dim:
                            agent_state = np.pad(agent_state, (0, self.state_dim - len(agent_state)))
                        states_list.append(agent_state)
                    else:
                        # If states insufficient, use zero padding
                        states_list.append(np.zeros(self.state_dim))
                    
            elif len(states.shape) == 2:
                actual_n_agents = states.shape[0]
                
                if actual_n_agents != self.n_agents:
                    log(f"Detected agent count change: controller={self.n_agents}, input_state={actual_n_agents}", LOG_INFO)
                    self.adapt_to_agent_count(actual_n_agents)
                
                states_list = [states[i] if i < actual_n_agents else np.zeros(self.state_dim) 
                              for i in range(self.n_agents)]
            else:
                log(f"Error: State shape {states.shape} cannot be parsed as agent states", LOG_ERROR)
                # Return zero actions
                return {
                    "leader": np.zeros(self.action_dim),
                    "followers": [np.zeros(self.action_dim) for _ in range(self.n_agents-1)]
                }
                
            # Select actions for each agent
            leader_action = None
            follower_actions = []

            self._ensure_follower_capacity(max(len(states_list) - 1, 0))
            
            # Get Leader state and all follower states
            leader_state = states_list[0]
            follower_states = states_list[1:] if len(states_list) > 1 else []
            
            # Build follower state array and mask (for attention mechanism)
            if follower_states:
                follower_states_array = np.stack(follower_states).reshape(1, len(follower_states), -1)
                follower_mask = np.ones((1, len(follower_states)), dtype=bool)
            else:
                follower_states_array = np.zeros((1, 0, self.state_dim), dtype=np.float32)
                follower_mask = np.zeros((1, 0), dtype=bool)
            
            if len(states_list) > 0:
                try:
                    if isinstance(self.leader_actor, AttentionLeaderActorNet):
                        leader_action = self.leader_actor.choose_action(
                            leader_state, 
                            follower_states_array, 
                            follower_mask,
                            evaluate=evaluate
                        )
                    else:
                        # Compatible with old LeaderActorNet
                        leader_action = self.leader_actor.choose_action(leader_state, evaluate=evaluate)
                    
                    # Add exploration noise
                    if add_noise and not evaluate:
                        noise = self.noises[0]() * noise_scale
                        leader_action += noise
                        leader_action = np.clip(leader_action, min_action, max_action)

                    leader_action = np.nan_to_num(
                        np.asarray(leader_action, dtype=np.float32),
                        nan=0.0,
                        posinf=max_action,
                        neginf=min_action
                    )
                    leader_action = np.clip(leader_action, min_action, max_action)
                except Exception as e:
                    log(f"Error selecting Leader action: {e}", LOG_ERROR)
                    leader_action = np.zeros(self.action_dim)
            
            # Followers (remaining agents)
            for i in range(1, len(states_list)):
                follower_idx = i - 1  # Follower index (0-based)
                follower_actor, resolved_actor_idx = self._get_follower_actor(follower_idx, target=False)
                
                try:
                    # If AttentionFollowerActorNet, need to provide context information
                    if isinstance(follower_actor, AttentionFollowerActorNet):
                        # Prepare Leader context observation
                        leader_obs_for_context = leader_state.reshape(1, -1)  # [1, state_dim]
                        leader_mask = np.ones((1, 1), dtype=bool)  # [1, 1]
                        
                        # Prepare other follower context observation
                        other_followers_indices = [j-1 for j in range(1, len(states_list)) if j != i]
                        
                        if other_followers_indices:
                            other_followers_obs = np.stack([states_list[j+1] for j in other_followers_indices])
                            other_followers_obs = other_followers_obs.reshape(1, len(other_followers_indices), -1)
                            other_followers_mask = np.ones((1, len(other_followers_indices)), dtype=bool)
                        else:
                            other_followers_obs = np.zeros((1, 0, self.state_dim), dtype=np.float32)
                            other_followers_mask = np.zeros((1, 0), dtype=bool)
                        
                        # Call follower Actor network
                        follower_action = follower_actor.choose_action(
                            states_list[i],
                            leader_obs_for_context,
                            other_followers_obs,
                            leader_mask,
                            other_followers_mask,
                            evaluate=evaluate
                        )
                    else:
                        # Compatible with old FollowerActorNet
                        follower_action = follower_actor.choose_action(states_list[i], evaluate=evaluate)
                    
                    # Add exploration noise
                    if add_noise and not evaluate:
                        # Ensure sufficient noise generators
                        if i >= len(self.noises):
                            self.noises.append(Ornstein_Uhlenbeck_Noise(mu=np.zeros(self.action_dim)))
                        
                        noise = self.noises[i]() * noise_scale
                        follower_action += noise
                        follower_action = np.clip(follower_action, min_action, max_action)

                    follower_action = np.nan_to_num(
                        np.asarray(follower_action, dtype=np.float32),
                        nan=0.0,
                        posinf=max_action,
                        neginf=min_action
                    )
                    follower_action = np.clip(follower_action, min_action, max_action)
                    
                    follower_actions.append(follower_action)
                except Exception as e:
                    log(f"Error selecting Follower {i} action (actor #{resolved_actor_idx}): {e}", LOG_ERROR)
                    follower_actions.append(np.zeros(self.action_dim))

            return {
                "leader": leader_action,
                "followers": follower_actions
            }
    
    def store_transition(self, states, actions, rewards, next_states, done=False, current_stage_tag: str = "default_stage"):
        """Store transition to replay buffer
        
        Args:
            states: Current state array [n_agents, state_dim] or structured dict
            actions: Action array [n_agents, action_dim] or structured dict
            rewards: Reward array [n_agents] or structured dict
            next_states: Next state array [n_agents, state_dim] or structured dict
            done: Done flag
            current_stage_tag: Current curriculum stage tag for marking experience stage
        """
        if (isinstance(states, dict) and "leader" in states and "followers" in states and
            isinstance(actions, dict) and "leader" in actions and "followers" in actions and
            isinstance(rewards, dict) and "leader" in rewards and "followers" in rewards and
            isinstance(next_states, dict) and "leader" in next_states and "followers" in next_states):

            def _clean_state(x):
                return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

            def _clean_action(x):
                arr = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=max_action, neginf=min_action)
                return np.clip(arr, min_action, max_action)

            def _clean_reward(x):
                val = float(np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0))
                return float(np.clip(val, -1000.0, 1000.0))

            observation = {
                "leader": _clean_state(states["leader"]),
                "followers": [_clean_state(item) for item in states.get("followers", [])]
            }
            action = {
                "leader": _clean_action(actions["leader"]),
                "followers": [_clean_action(item) for item in actions.get("followers", [])]
            }
            reward = {
                "leader": _clean_reward(rewards.get("leader", 0.0)),
                "followers": [_clean_reward(item) for item in rewards.get("followers", [])]
            }
            next_observation = {
                "leader": _clean_state(next_states["leader"]),
                "followers": [_clean_state(item) for item in next_states.get("followers", [])]
            }

            self.memory.store_transition(observation, action, reward, next_observation, bool(done), stage_tag=current_stage_tag)
            return
            
        states = np.nan_to_num(np.asarray(states, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        actions = np.nan_to_num(np.asarray(actions, dtype=np.float32), nan=0.0, posinf=max_action, neginf=min_action)
        actions = np.clip(actions, min_action, max_action)
        rewards = np.nan_to_num(np.asarray(rewards, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        rewards = np.clip(rewards, -1000.0, 1000.0)
        next_states = np.nan_to_num(np.asarray(next_states, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        
        n_agents = states.shape[0]
        
        if n_agents == 0:
            log("store_transition: Received 0 agent states, skipping storage.", LOG_WARNING)
            return

        leader_state = states[0]
        follower_states = states[1:] if n_agents > 1 else []

        leader_action = actions[0]
        follower_actions = actions[1:] if n_agents > 1 else []

        leader_reward = rewards[0] 
        follower_rewards = rewards[1:] if n_agents > 1 else []
        
        leader_next_state = next_states[0]
        follower_next_states = next_states[1:] if n_agents > 1 else []

        observation = {"leader": leader_state, "followers": list(follower_states)}
        action = {"leader": leader_action, "followers": list(follower_actions)}
        reward = {"leader": leader_reward, "followers": list(follower_rewards)}
        next_observation = {"leader": leader_next_state, "followers": list(follower_next_states)}
        
        self.memory.store_transition(observation, action, reward, next_observation, done, stage_tag=current_stage_tag)
    
    def train(self, batch_size=None, current_stage_tag: str = "default_stage", current_stage_number: int = 0):
        """Train networks

        Args:
            batch_size: Batch size, use self.batch_size if None
            current_stage_tag: Current curriculum stage tag for distinguishing old/new experiences
            current_stage_number: Current curriculum stage number for calculating old/new experience sampling ratio
        """
        if batch_size is None:
            batch_size = self.batch_size

        train_status = {
            "updated": False,
            "reason": "running",
            "stage_tag": current_stage_tag,
            "stage_number": int(current_stage_number),
            "batch_size": int(batch_size),
            "train_step": int(self.train_step)
        }

        def _finalize_train_status(updated, reason, **extra):
            train_status["updated"] = bool(updated)
            train_status["reason"] = reason
            train_status["train_step"] = int(self.train_step)
            if extra:
                train_status.update(extra)
            self.last_train_status = train_status
            return train_status

        sampled_data = self.memory.sample(
            batch_size,
            current_stage_tag=current_stage_tag,
            current_stage_number=current_stage_number
        )
        
        if sampled_data is None:
            log(f"MASACController: Stage '{current_stage_tag}' (Number {current_stage_number}) skipped training step due to insufficient samples or sampling error.", LOG_DEBUG)
            return _finalize_train_status(False, "insufficient_samples")

        batch_data, batch_masks = sampled_data
        
        # Get leader and followers data
        obs_leader = batch_data["observation"]["leader"]
        obs_followers = batch_data["observation"]["followers"]
        mask_followers = batch_masks["followers"].bool()
        
        act_leader = batch_data["action"]["leader"]
        act_followers = batch_data["action"]["followers"]
        
        reward_leader = batch_data["reward"]["leader"]
        reward_followers = batch_data["reward"]["followers"]
        
        next_obs_leader = batch_data["next_observation"]["leader"]
        next_obs_followers = batch_data["next_observation"]["followers"]
        
        done = batch_data["done"]

        # Sanitize sampled tensors to avoid NaN/Inf poisoning training.
        obs_leader = torch.nan_to_num(obs_leader, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e4, 1e4)
        obs_followers = torch.nan_to_num(obs_followers, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e4, 1e4)
        next_obs_leader = torch.nan_to_num(next_obs_leader, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e4, 1e4)
        next_obs_followers = torch.nan_to_num(next_obs_followers, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e4, 1e4)
        act_leader = torch.nan_to_num(act_leader, nan=0.0, posinf=max_action, neginf=min_action).clamp(min_action, max_action)
        act_followers = torch.nan_to_num(act_followers, nan=0.0, posinf=max_action, neginf=min_action).clamp(min_action, max_action)
        reward_leader = torch.nan_to_num(reward_leader, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1000.0, 1000.0)
        reward_followers = torch.nan_to_num(reward_followers, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1000.0, 1000.0)
        done = torch.nan_to_num(done, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        
        # Get number of followers (batch may have different number of followers)
        B, max_F, _ = obs_followers.shape
        self._ensure_follower_capacity(max_F)
        num_active_followers = max_F
        
        # ===== Calculate Critic loss =====
        # 1. Calculate target Q value
        with torch.no_grad():
            # Evaluate leader's next state action
            if isinstance(self.target_leader_actor, AttentionLeaderActorNet):
                # Use attention-based Leader Actor network
                next_act_leader, next_log_prob_leader = self.target_leader_actor.evaluate(
                    next_obs_leader, next_obs_followers, mask_followers
                )
            else:
                # Use legacy Leader Actor network
                next_act_leader, next_log_prob_leader = self.target_leader_actor.evaluate(next_obs_leader)
            
            # Evaluate followers' next state actions
            next_act_followers_list = []
            next_log_prob_followers_list = []
            
            for k in range(num_active_followers):
                # Prepare context data for follower k
                follower_self_obs_k = next_obs_followers[:, k, :]  # [B, state_dim]
                leader_for_context_k = next_obs_leader  # [B, state_dim]
                valid_leader_mask_k = torch.ones(B, 1, dtype=torch.bool, device=self.device)  # [B, 1]
                target_follower_actor, _ = self._get_follower_actor(k, target=True)
                
                # Prepare other followers' context
                other_follower_indices = [j for j in range(max_F) if j != k]
                if other_follower_indices:
                    other_followers_context_k = next_obs_followers[:, other_follower_indices, :]  # [B, max_F-1, state_dim]
                    valid_other_followers_mask_k = mask_followers[:, other_follower_indices]  # [B, max_F-1]
                else:
                    other_followers_context_k = torch.zeros(B, 0, self.state_dim, device=self.device)
                    valid_other_followers_mask_k = torch.zeros(B, 0, dtype=torch.bool, device=self.device)
                
                if isinstance(target_follower_actor, AttentionFollowerActorNet):
                    # Use attention-based Follower Actor network
                    act_k, log_prob_k = target_follower_actor.evaluate(
                        follower_self_obs_k,
                        leader_for_context_k,
                        other_followers_context_k,
                        valid_leader_mask_k,
                        valid_other_followers_mask_k
                    )
                else:
                    # Use legacy Follower Actor network
                    act_k, log_prob_k = target_follower_actor.evaluate(follower_self_obs_k)
                
                next_act_followers_list.append(act_k.unsqueeze(1))  # Add follower dimension [B, 1, action_dim]
                next_log_prob_followers_list.append(log_prob_k.unsqueeze(1))  # [B, 1, 1]
            
            if num_active_followers < max_F:
                # Create padding actions
                action_padding = torch.zeros(
                    B, max_F - num_active_followers, self.action_dim, 
                    device=self.device
                )
                # Create padding log probabilities
                log_prob_padding = torch.zeros(
                    B, max_F - num_active_followers, 1, 
                    device=self.device
                )
                
                # Concatenate with padding if there are actual follower actions
                if next_act_followers_list:
                    next_act_followers = torch.cat(next_act_followers_list + [action_padding], dim=1)
                    next_log_prob_followers = torch.cat(next_log_prob_followers_list + [log_prob_padding], dim=1)
                else:
                    # If no actual follower actions, use padding directly
                    next_act_followers = action_padding
                    next_log_prob_followers = log_prob_padding
            else:
                # If there are enough follower Actor networks, concatenate all results directly
                if next_act_followers_list:
                    next_act_followers = torch.cat(next_act_followers_list, dim=1)  # [B, max_F, action_dim]
                    next_log_prob_followers = torch.cat(next_log_prob_followers_list, dim=1)  # [B, max_F, 1]
                else:
                    # Edge case: no followers
                    next_act_followers = torch.zeros(B, 0, self.action_dim, device=self.device)
                    next_log_prob_followers = torch.zeros(B, 0, 1, device=self.device)

            if (
                not torch.isfinite(next_act_leader).all()
                or not torch.isfinite(next_log_prob_leader).all()
                or not torch.isfinite(next_act_followers).all()
                or not torch.isfinite(next_log_prob_followers).all()
            ):
                log(
                    f"MASACController.train: detected non-finite target actions/log_probs at stage '{current_stage_tag}', skip this update.",
                    LOG_WARNING
                )
                return _finalize_train_status(False, "non_finite_target_policy")
            
            # Calculate target Q values
            target_q1_leader, target_q2_leader, target_q1_followers, target_q2_followers = self.critic.forward_target(
                next_obs_leader, next_obs_followers, 
                next_act_leader, next_act_followers,
                mask_followers
            )
            
            # Use min Q
            target_q_leader = torch.min(target_q1_leader, target_q2_leader)
            target_q_followers = torch.min(target_q1_followers, target_q2_followers)
            
            # Calculate target value (reward + gamma * (Q - alpha * log_prob))
            # Leader target
            target_leader = reward_leader + self.gamma * (1 - done) * (
                target_q_leader - self.entroy_leader.alpha * next_log_prob_leader
            )
            
            # Followers target (apply mask)
            target_followers = reward_followers + self.gamma * (1 - done).unsqueeze(1) * (
                target_q_followers - self.entroy_follower.alpha * next_log_prob_followers
            )
        
        # 2. Calculate current Q values
        current_q1_leader, current_q2_leader, current_q1_followers, current_q2_followers = self.critic(
            obs_leader, obs_followers,
            act_leader, act_followers,
            mask_followers
        )
        
        # 3. Calculate Critic loss (MSE)
        # Leader loss
        critic_loss_leader = F.mse_loss(current_q1_leader, target_leader) + F.mse_loss(current_q2_leader, target_leader)
        
        # Followers loss (apply mask)
        # First, calculate element-wise MSE loss
        critic_loss_followers_q1 = F.mse_loss(
            current_q1_followers, target_followers, reduction='none'
        )
        critic_loss_followers_q2 = F.mse_loss(
            current_q2_followers, target_followers, reduction='none'
        )
        
        # Apply mask and calculate average
        # Ensure mask shape is correct [B, max_F, 1]
        mask_3d = mask_followers.unsqueeze(-1)
        critic_loss_followers_q1 = (critic_loss_followers_q1 * mask_3d).sum() / (mask_3d.sum() + 1e-8)
        critic_loss_followers_q2 = (critic_loss_followers_q2 * mask_3d).sum() / (mask_3d.sum() + 1e-8)
        
        critic_loss_followers = critic_loss_followers_q1 + critic_loss_followers_q2
        
        # Total Critic loss
        critic_loss = critic_loss_leader + critic_loss_followers

        if not torch.isfinite(critic_loss):
            log(f"MASACController.train: critic_loss is non-finite at stage '{current_stage_tag}', skip this update.", LOG_WARNING)
            return _finalize_train_status(False, "non_finite_critic_loss")
        
        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()
        
        # ===== Calculate Actor loss =====
        # Leader Actor loss
        # 1. Generate new actions
        if isinstance(self.leader_actor, AttentionLeaderActorNet):
            # Use attention Leader Actor network
            new_act_leader, log_prob_leader = self.leader_actor.evaluate(
                obs_leader, obs_followers, mask_followers
            )
        else:
            # Use legacy Leader Actor network
            new_act_leader, log_prob_leader = self.leader_actor.evaluate(obs_leader)

        if not torch.isfinite(new_act_leader).all() or not torch.isfinite(log_prob_leader).all():
            log(f"MASACController.train: non-finite leader policy outputs at stage '{current_stage_tag}', skip this update.", LOG_WARNING)
            return _finalize_train_status(False, "non_finite_leader_policy")
        
        # 2. Calculate Q values
        q1_leader, q2_leader, _, _ = self.critic(
            obs_leader, obs_followers,
            new_act_leader, act_followers,  # Use new Leader action, keep Followers actions unchanged
            mask_followers
        )
        min_q_leader = torch.min(q1_leader, q2_leader)
        
        # 3. Calculate Leader Actor loss (policy gradient, maximize Q - alpha * log_prob)
        actor_loss_leader = (self.entroy_leader.alpha * log_prob_leader - min_q_leader).mean()

        if not torch.isfinite(actor_loss_leader):
            log(f"MASACController.train: actor_loss_leader is non-finite at stage '{current_stage_tag}', skip this update.", LOG_WARNING)
            return _finalize_train_status(False, "non_finite_leader_actor_loss")
        
        # Update Leader Actor
        self.leader_actor_optimizer.zero_grad()
        actor_loss_leader.backward()
        torch.nn.utils.clip_grad_norm_(self.leader_actor.parameters(), max_norm=10.0)
        self.leader_actor_optimizer.step()
        
        # Follower Actors loss
        # Generate new actions for each follower
        new_act_followers_list = []
        log_prob_followers_list = []
        
        for k in range(num_active_followers):
            # Prepare follower k's context data
            follower_self_obs_k = obs_followers[:, k, :]  # [B, state_dim]
            leader_for_context_k = obs_leader  # [B, state_dim]
            valid_leader_mask_k = torch.ones(B, 1, dtype=torch.bool, device=self.device)  # [B, 1]
            follower_actor, _ = self._get_follower_actor(k, target=False)
            
            # Prepare other followers context
            other_follower_indices = [j for j in range(max_F) if j != k]
            if other_follower_indices:
                other_followers_context_k = obs_followers[:, other_follower_indices, :]  # [B, max_F-1, state_dim]
                valid_other_followers_mask_k = mask_followers[:, other_follower_indices]  # [B, max_F-1]
            else:
                other_followers_context_k = torch.zeros(B, 0, self.state_dim, device=self.device)
                valid_other_followers_mask_k = torch.zeros(B, 0, dtype=torch.bool, device=self.device)
            
            if isinstance(follower_actor, AttentionFollowerActorNet):
                # Use attention Follower Actor network
                act_k, log_prob_k = follower_actor.evaluate(
                    follower_self_obs_k,
                    leader_for_context_k,
                    other_followers_context_k,
                    valid_leader_mask_k,
                    valid_other_followers_mask_k
                )
            else:
                # Use legacy Follower Actor network
                act_k, log_prob_k = follower_actor.evaluate(follower_self_obs_k)
            
            # Collect results
            new_act_followers_list.append(act_k.unsqueeze(1))  # Add follower dimension [B, 1, action_dim]
            log_prob_followers_list.append(log_prob_k.unsqueeze(1))  # [B, 1, 1]
        
        # Pad with zeros if not enough follower Actor networks
        if num_active_followers < max_F:
            # Create padding actions
            action_padding = torch.zeros(
                B, max_F - num_active_followers, self.action_dim, 
                device=self.device
            )
            # Create padding log probabilities
            log_prob_padding = torch.zeros(
                B, max_F - num_active_followers, 1, 
                device=self.device
            )
            
            # Concatenate with padding if there are actual follower actions
            if new_act_followers_list:
                new_act_followers = torch.cat(new_act_followers_list + [action_padding], dim=1)
                log_prob_followers = torch.cat(log_prob_followers_list + [log_prob_padding], dim=1)
            else:
                # If no actual follower actions, use padding directly
                new_act_followers = action_padding
                log_prob_followers = log_prob_padding
        else:
            # If there are enough follower Actor networks, concatenate all results directly
            if new_act_followers_list:
                new_act_followers = torch.cat(new_act_followers_list, dim=1)  # [B, max_F, action_dim]
                log_prob_followers = torch.cat(log_prob_followers_list, dim=1)  # [B, max_F, 1]
            else:
                # Edge case: no followers
                new_act_followers = torch.zeros(B, 0, self.action_dim, device=self.device)
                log_prob_followers = torch.zeros(B, 0, 1, device=self.device)

        if not torch.isfinite(new_act_followers).all() or not torch.isfinite(log_prob_followers).all():
            log(f"MASACController.train: non-finite follower policy outputs at stage '{current_stage_tag}', skip this update.", LOG_WARNING)
            return _finalize_train_status(False, "non_finite_follower_policy")
        
        # 2. Calculate Q values
        _, _, q1_followers, q2_followers = self.critic(
            obs_leader, obs_followers,
            act_leader, new_act_followers,  # Use new Followers actions, keep Leader action unchanged
            mask_followers
        )
        min_q_followers = torch.min(q1_followers, q2_followers)
        
        # 3. Calculate Follower Actor loss (policy gradient, maximize Q - alpha * log_prob)
        # Apply mask
        actor_loss_followers = (self.entroy_follower.alpha * log_prob_followers - min_q_followers) * mask_3d
        # Average over masked loss
        actor_loss_followers = actor_loss_followers.sum() / (mask_3d.sum() + 1e-8)

        if not torch.isfinite(actor_loss_followers):
            log(f"MASACController.train: actor_loss_followers is non-finite at stage '{current_stage_tag}', skip this update.", LOG_WARNING)
            return _finalize_train_status(False, "non_finite_follower_actor_loss")
        
        # Update Follower Actor
        self.follower_actor_optimizer.zero_grad()
        actor_loss_followers.backward()
        if self.share_follower_policy:
            torch.nn.utils.clip_grad_norm_(self.follower_actors[0].parameters(), max_norm=10.0)
        else:
            torch.nn.utils.clip_grad_norm_(self.follower_actors.parameters(), max_norm=10.0)
        self.follower_actor_optimizer.step()
        self._sync_shared_follower_policy_weights()
        
        # ===== Update entropy weight alpha =====
        # Leader alpha
        alpha_loss_leader = -(self.entroy_leader.log_alpha * (
            log_prob_leader.detach() + self.entroy_leader.target_entropy
        )).mean()

        if not torch.isfinite(alpha_loss_leader):
            log(f"MASACController.train: alpha_loss_leader is non-finite at stage '{current_stage_tag}', skip alpha update.", LOG_WARNING)
            return _finalize_train_status(False, "non_finite_leader_alpha_loss")
        
        self.leader_alpha_optimizer.zero_grad()
        alpha_loss_leader.backward()
        torch.nn.utils.clip_grad_norm_([self.entroy_leader.log_alpha], max_norm=5.0)
        self.leader_alpha_optimizer.step()
        
        # Update alpha value
        self.entroy_leader.alpha = self.entroy_leader.log_alpha.exp()
        
        # Follower alpha
        alpha_loss_followers = -(self.entroy_follower.log_alpha * (
            log_prob_followers.detach() + self.entroy_follower.target_entropy
        )) * mask_3d
        alpha_loss_followers = alpha_loss_followers.sum() / (mask_3d.sum() + 1e-8)

        if not torch.isfinite(alpha_loss_followers):
            log(f"MASACController.train: alpha_loss_followers is non-finite at stage '{current_stage_tag}', skip alpha update.", LOG_WARNING)
            return _finalize_train_status(False, "non_finite_follower_alpha_loss")
        
        self.follower_alpha_optimizer.zero_grad()
        alpha_loss_followers.backward()
        torch.nn.utils.clip_grad_norm_([self.entroy_follower.log_alpha], max_norm=5.0)
        self.follower_alpha_optimizer.step()
        
        # Update alpha value
        self.entroy_follower.alpha = self.entroy_follower.log_alpha.exp()

        with torch.no_grad():
            self.entroy_leader.log_alpha.data.clamp_(-20.0, 2.0)
            self.entroy_follower.log_alpha.data.clamp_(-20.0, 2.0)
        
        # ===== Soft update target networks =====
        # Update Critic target network
        self.critic.soft_update(self.tau)
        
        # Update Leader Actor target network
        for target_param, param in zip(self.target_leader_actor.parameters(), self.leader_actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        # Update Follower Actors target networks
        if self.share_follower_policy and len(self.follower_actors) > 0:
            for target_param, param in zip(self.target_follower_actors[0].parameters(), self.follower_actors[0].parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            self._sync_shared_follower_policy_weights()
        else:
            for i in range(len(self.follower_actors)):
                for target_param, param in zip(self.target_follower_actors[i].parameters(), self.follower_actors[i].parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
        # Update training step count
        self.train_step += 1
        return _finalize_train_status(
            True,
            "ok",
            critic_loss=float(critic_loss.detach().item()),
            actor_loss_leader=float(actor_loss_leader.detach().item()),
            actor_loss_followers=float(actor_loss_followers.detach().item()),
            alpha_loss_leader=float(alpha_loss_leader.detach().item()),
            alpha_loss_followers=float(alpha_loss_followers.detach().item())
        )

    def save_models(self, path):
        """Save model
        
        Args:
            path: Save path
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save network parameters
        torch.save({
            # Actor network parameters
            'leader_actor': self.leader_actor.state_dict(),
            'follower_actors': [actor.state_dict() for actor in self.follower_actors],
            'target_leader_actor': self.target_leader_actor.state_dict(),
            'target_follower_actors': [actor.state_dict() for actor in self.target_follower_actors],
            
            # Critic network parameters
            'critic': self.critic.state_dict(),
            
            # Entropy parameters
            'entroy_leader': self.entroy_leader.__dict__,
            'entroy_follower': self.entroy_follower.__dict__,
            
            # Optimizer parameters
            'leader_actor_optimizer': self.leader_actor_optimizer.state_dict(),
            'follower_actor_optimizer': self.follower_actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'leader_alpha_optimizer': self.leader_alpha_optimizer.state_dict(),
            'follower_alpha_optimizer': self.follower_alpha_optimizer.state_dict(),
            
            # Training statistics
            'train_step': self.train_step,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_successes': self.episode_successes,
            
            # Configuration
            'n_agents': self.n_agents,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'tau': self.tau,
            'batch_size': self.batch_size,
            'memory_capacity': self.memory_capacity,
            'replay_ratio': self.replay_ratio,
            'share_follower_policy': self.share_follower_policy,
            'use_attention': self.use_attention,
            'use_gat': self.use_gat
        }, path)
        
        print(f"Model saved to {path}")
    
    def load_models(self, path, strict=False):
        """Load model
        
        Args:
            path: Model path
            strict: Whether to strictly load (if False, allows partial parameter mismatch)
            
        Returns:
            bool: Whether successfully loaded
        """
        if not os.path.exists(path):
            print(f"Model file does not exist: {path}")
            return False
            
        try:
            # PyTorch 2.6+ defaults to weights_only=True, which can fail for full checkpoints
            # containing optimizer states and other Python objects.
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            except TypeError:
                # Backward compatibility for older PyTorch versions without weights_only argument.
                checkpoint = torch.load(path, map_location=self.device)

            checkpoint_share_policy = checkpoint.get('share_follower_policy', None)
            if checkpoint_share_policy is not None and bool(checkpoint_share_policy) != self.share_follower_policy:
                print(
                    f"Info: checkpoint share_follower_policy={bool(checkpoint_share_policy)}, "
                    f"runtime share_follower_policy={self.share_follower_policy}. Keeping runtime setting."
                )

            checkpoint_use_attention = checkpoint.get('use_attention', None)
            if checkpoint_use_attention is not None and bool(checkpoint_use_attention) != self.use_attention:
                print(f"Info: checkpoint use_attention={bool(checkpoint_use_attention)}, "
                      f"runtime use_attention={self.use_attention}. Keeping runtime setting.")

            checkpoint_use_gat = checkpoint.get('use_gat', None)
            if checkpoint_use_gat is not None and bool(checkpoint_use_gat) != getattr(self, 'use_gat', False):
                print(f"Info: checkpoint use_gat={bool(checkpoint_use_gat)}, "
                      f"runtime use_gat={getattr(self, 'use_gat', False)}. Keeping runtime setting.")
            
            # Load Actor network parameters
            self.leader_actor.load_state_dict(checkpoint['leader_actor'], strict=strict)
            saved_follower_actors = checkpoint.get('follower_actors', [])
            if not isinstance(saved_follower_actors, list) or len(saved_follower_actors) == 0:
                raise KeyError("Checkpoint missing valid 'follower_actors' list")

            saved_follower_count = len(saved_follower_actors)
            self._ensure_follower_capacity(saved_follower_count)

            if saved_follower_count > len(self.follower_actors):
                print(f"Warning: Checkpoint has {saved_follower_count} follower actors, but controller has {len(self.follower_actors)}. Extra checkpoint actors will be ignored.")
            elif saved_follower_count < len(self.follower_actors):
                print(f"Warning: Checkpoint has {saved_follower_count} follower actors, but controller has {len(self.follower_actors)}. Missing actors will reuse the last loaded follower actor.")

            for i, actor in enumerate(self.follower_actors):
                if i < saved_follower_count:
                    actor.load_state_dict(saved_follower_actors[i], strict=strict)
                else:
                    actor.load_state_dict(saved_follower_actors[-1], strict=strict)
            
            # Load target Actor network parameters
            if 'target_leader_actor' in checkpoint:
                self.target_leader_actor.load_state_dict(checkpoint['target_leader_actor'], strict=strict)
            else:
                self.target_leader_actor.load_state_dict(self.leader_actor.state_dict())
                
            if 'target_follower_actors' in checkpoint:
                saved_target_follower_actors = checkpoint.get('target_follower_actors', [])
                if not isinstance(saved_target_follower_actors, list):
                    saved_target_follower_actors = []

                target_count = len(saved_target_follower_actors)
                self._ensure_follower_capacity(max(saved_follower_count, target_count))

                if len(saved_target_follower_actors) > len(self.target_follower_actors):
                    print(f"Warning: Checkpoint has {len(saved_target_follower_actors)} target follower actors, but controller has {len(self.target_follower_actors)}. Extra checkpoint actors will be ignored.")
                elif len(saved_target_follower_actors) < len(self.target_follower_actors):
                    print(f"Warning: Checkpoint has {len(saved_target_follower_actors)} target follower actors, but controller has {len(self.target_follower_actors)}. Missing actors will mirror online follower actors.")

                for i, actor in enumerate(self.target_follower_actors):
                    if i < target_count:
                        actor.load_state_dict(saved_target_follower_actors[i], strict=strict)
                    else:
                        actor.load_state_dict(self.follower_actors[i].state_dict())
            else:
                for i, actor in enumerate(self.target_follower_actors):
                    actor.load_state_dict(self.follower_actors[i].state_dict())

            self._sync_shared_follower_policy_weights()
            
            # Load Critic network parameters
            self.critic.load_state_dict(checkpoint['critic'], strict=strict)
            
            # Load entropy parameters
            if 'entroy_leader' in checkpoint:
                # Handle tensors
                for key, value in checkpoint['entroy_leader'].items():
                    if isinstance(value, torch.Tensor):
                        setattr(self.entroy_leader, key, value.to(self.device))
                    else:
                        setattr(self.entroy_leader, key, value)
                        
            if 'entroy_follower' in checkpoint:
                # Handle tensors
                for key, value in checkpoint['entroy_follower'].items():
                    if isinstance(value, torch.Tensor):
                        setattr(self.entroy_follower, key, value.to(self.device))
                    else:
                        setattr(self.entroy_follower, key, value)
            
            # Load optimizer parameters (if exists)
            if 'leader_actor_optimizer' in checkpoint:
                self._safe_load_optimizer_state(
                    self.leader_actor_optimizer,
                    checkpoint['leader_actor_optimizer'],
                    'leader_actor_optimizer'
                )
                
            if 'follower_actor_optimizer' in checkpoint:
                self._safe_load_optimizer_state(
                    self.follower_actor_optimizer,
                    checkpoint['follower_actor_optimizer'],
                    'follower_actor_optimizer'
                )
                
            if 'critic_optimizer' in checkpoint:
                self._safe_load_optimizer_state(
                    self.critic_optimizer,
                    checkpoint['critic_optimizer'],
                    'critic_optimizer'
                )
                
            if 'leader_alpha_optimizer' in checkpoint:
                self._safe_load_optimizer_state(
                    self.leader_alpha_optimizer,
                    checkpoint['leader_alpha_optimizer'],
                    'leader_alpha_optimizer'
                )
                
            if 'follower_alpha_optimizer' in checkpoint:
                self._safe_load_optimizer_state(
                    self.follower_alpha_optimizer,
                    checkpoint['follower_alpha_optimizer'],
                    'follower_alpha_optimizer'
                )
            
            # Load training statistics (if exists)
            if 'train_step' in checkpoint:
                self.train_step = checkpoint['train_step']
                
            if 'episode_rewards' in checkpoint:
                self.episode_rewards = checkpoint['episode_rewards']
                
            if 'episode_lengths' in checkpoint:
                self.episode_lengths = checkpoint['episode_lengths']
                
            if 'episode_successes' in checkpoint:
                self.episode_successes = checkpoint['episode_successes']
            
            # Load configuration (if exists)
            if 'n_agents' in checkpoint:
                checkpoint_n_agents = checkpoint['n_agents']
                if checkpoint_n_agents != self.n_agents:
                    print(
                        f"Info: checkpoint n_agents={checkpoint_n_agents}, "
                        f"runtime n_agents={self.n_agents}. Keeping runtime setting for dynamic adaptation."
                    )
                
            if 'replay_ratio' in checkpoint:
                self.replay_ratio = checkpoint['replay_ratio']
            
            print(f"Successfully loaded model: {path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def track_episode_rewards(self, rewards):
        """Track episode rewards
        
        Args:
            rewards: List or array of rewards for each agent
        """
        # Calculate total reward (sum of all agent rewards)
        total_reward = sum(rewards) if isinstance(rewards, (list, np.ndarray)) else rewards
        self.episode_rewards.append(total_reward)
        
    def track_episode_length(self, length):
        """Track episode length
        
        Args:
            length: Number of steps in episode
        """
        self.episode_lengths.append(length)
        
    def track_episode_success(self, success):
        """Track episode success flag
        
        Args:
            success: Whether task completed successfully
        """
        self.episode_successes.append(1 if success else 0)
        
    def get_training_stats(self, window=100):
        """Get training statistics
        
        Args:
            window: Window size for calculating averages
            
        Returns:
            dict: Statistics dictionary
        """
        stats = {}
        
        # Calculate average reward
        if len(self.episode_rewards) > 0:
            stats['last_reward'] = self.episode_rewards[-1]
            stats['avg_reward'] = np.mean(self.episode_rewards[-window:])
            
        # Calculate average step length
        if len(self.episode_lengths) > 0:
            stats['last_length'] = self.episode_lengths[-1]
            stats['avg_length'] = np.mean(self.episode_lengths[-window:])
            
        # Calculate success rate
        if len(self.episode_successes) > 0:
            stats['last_success'] = self.episode_successes[-1]
            stats['success_rate'] = np.mean(self.episode_successes[-window:])
            
        # Training step
        stats['train_step'] = self.train_step
        
        return stats

    def get_policy_parameters_for_curriculum(self):
        """Export policy and Critic parameters for curriculum learning knowledge transfer"""
        # Create dictionary to save follower Actor parameters
        follower_actors_dict = {}
        for i, actor in enumerate(self.follower_actors):
            follower_actors_dict[f'actor_{i+1}'] = actor.state_dict()
        
        params = {
            'actors': {
                'actor_0': self.leader_actor.state_dict(),
                **follower_actors_dict  # Add all follower Actor parameters
            },
            'entropy': {
                'entropy_0': {
                    'log_alpha': self.entroy_leader.log_alpha.data.clone(),
                    'target_entropy': self.entroy_leader.target_entropy,
                    'alpha': self.entroy_leader.alpha.data.clone()
                },
                'entropy_1': {
                    'log_alpha': self.entroy_follower.log_alpha.data.clone(),
                    'target_entropy': self.entroy_follower.target_entropy,
                    'alpha': self.entroy_follower.alpha.data.clone()
                }
            },
            'critic': self.critic.state_dict(), # Export Critic parameters
            'num_follower_actors': len(self.follower_actors)  # Add follower Actor count information
        }
        log("Exported parameters for curriculum transfer from MASACController (includes Critic and multiple Follower Actors)", LOG_DEBUG)
        return params
        
    def update_components_from_transfer(self, transferred_params: Dict[str, Any]):
        """从课程学习迁移的参数更新控制器组件"""
        log("开始从迁移的参数更新MASACController组件...", LOG_INFO)
        updated_components = []
        try:
            # 检查是否有智能体数量信息
            if 'agent_counts' in transferred_params:
                agent_counts = transferred_params['agent_counts']
                log(f"检测到智能体数量变化信息:", LOG_INFO)
                log(f"  - 源任务: {agent_counts.get('source', 'N/A')} 个智能体", LOG_INFO)
                log(f"  - 目标任务: {agent_counts.get('target', 'N/A')} 个智能体", LOG_INFO)
                log(f"  - 源任务从机数: {agent_counts.get('source_followers', 'N/A')}", LOG_INFO)
                log(f"  - 目标任务从机数: {agent_counts.get('target_followers', 'N/A')}", LOG_INFO)
            
            # 更新 Actor 参数
            if 'actors' in transferred_params:
                actor_params = transferred_params['actors']
                
                # 更新Leader Actor
                if 'actor_0' in actor_params:
                    self.leader_actor.load_state_dict(actor_params['actor_0'])
                    self.target_leader_actor.load_state_dict(self.leader_actor.state_dict()) # 更新目标网络
                    updated_components.append("Leader Actor")
                
                # 更新Follower Actors
                for i in range(len(self.follower_actors)):
                    follower_key = f'actor_{i+1}'
                    if follower_key in actor_params:
                        self.follower_actors[i].load_state_dict(actor_params[follower_key])
                        self.target_follower_actors[i].load_state_dict(self.follower_actors[i].state_dict()) # 更新目标网络
                        
                        # 检查是否是复用的参数
                        if 'agent_counts' in transferred_params:
                            source_followers = transferred_params['agent_counts'].get('source_followers', 0)
                            if i >= source_followers:
                                log(f"  - Follower Actor {i+1} 使用了从最后一个已有从机复用的参数", LOG_INFO)
                        
                        updated_components.append(f"Follower Actor {i+1}")
                    else:
                        # 如果参数中没有对应的actor，说明这是新增的从机，应该由知识迁移系统处理了复用
                        log(f"  - Follower Actor {i+1} 在迁移参数中未找到（可能是新增的从机）", LOG_WARNING)

                self._sync_shared_follower_policy_weights()

            # 更新 Entropy 参数
            if 'entropy' in transferred_params:
                entropy_params = transferred_params['entropy']
                
                # 更新Leader Entropy
                if 'entropy_0' in entropy_params:
                    leader_entropy = entropy_params['entropy_0']
                    for key, value in leader_entropy.items():
                        if key == 'log_alpha' and isinstance(value, torch.Tensor):
                            # 确保log_alpha保留梯度信息
                            if not hasattr(self.entroy_leader, 'log_alpha') or self.entroy_leader.log_alpha is None:
                                self.entroy_leader.log_alpha = value.clone().to(self.device).requires_grad_(True)
                            else:
                                self.entroy_leader.log_alpha.data.copy_(value.to(self.device))
                        elif isinstance(value, torch.Tensor):
                            setattr(self.entroy_leader, key, value.to(self.device))
                        else:
                            setattr(self.entroy_leader, key, value)
                    
                    # 确保重新设置优化器
                    self.leader_alpha_optimizer = torch.optim.Adam([self.entroy_leader.log_alpha], lr=self.entropy_lr)
                    updated_components.append("Leader Entropy")
                
                # 更新Follower Entropy
                if 'entropy_1' in entropy_params:
                    follower_entropy = entropy_params['entropy_1']
                    for key, value in follower_entropy.items():
                        if key == 'log_alpha' and isinstance(value, torch.Tensor):
                            # 确保log_alpha保留梯度信息
                            if not hasattr(self.entroy_follower, 'log_alpha') or self.entroy_follower.log_alpha is None:
                                self.entroy_follower.log_alpha = value.clone().to(self.device).requires_grad_(True)
                            else:
                                self.entroy_follower.log_alpha.data.copy_(value.to(self.device))
                        elif isinstance(value, torch.Tensor):
                            setattr(self.entroy_follower, key, value.to(self.device))
                        else:
                            setattr(self.entroy_follower, key, value)
                    
                    # 确保重新设置优化器
                    self.follower_alpha_optimizer = torch.optim.Adam([self.entroy_follower.log_alpha], lr=self.entropy_lr)
                    updated_components.append("Follower Entropy (所有从机共享)")
            
            # 更新 Critic 参数
            if 'critic' in transferred_params:
                self.critic.load_state_dict(transferred_params['critic'])
                updated_components.append("Critic")
            
            log(f"已成功更新MASACController组件: {', '.join(updated_components)}", LOG_INFO)
            
        except Exception as e:
            log(f"更新组件时出错: {e}", LOG_ERROR)
            import traceback
            traceback.print_exc()


# ==============================================================================
# 模块: 文件与 JSON 序列化工具
# ==============================================================================

def ensure_dir_exists(dir_path):
    """确保目录存在，如果不存在则创建
    
    Args:
        dir_path: 目录路径
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")
    return dir_path

def get_timestamp():
    """获取格式化的时间戳
    
    Returns:
        格式化的时间戳字符串: YYYYMMDD_HHMMSS
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ==============================================================================
# 模块: 实验输出路径工具（标签与目录）
# ==============================================================================

def _sanitize_tag(tag: str, default_tag: str = "ablation_no_curriculum") -> str:
    """Sanitize user-provided tags for safe filesystem paths."""
    if not tag:
        return default_tag

    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in tag.strip())
    return safe or default_tag


def _default_no_curriculum_tag(use_attention: bool, use_gat: bool) -> str:
    """Get default no-curriculum experiment tag by architecture."""
    if not use_attention:
        return "no_attention_no_curriculum"
    if use_gat:
        return "gat_attention_no_curriculum"
    return "attention_no_curriculum"


def _default_curriculum_tag(use_attention: bool, use_gat: bool) -> str:
    """Get default curriculum experiment tag by architecture."""
    if not use_attention:
        return "ablation_curriculum_no_attention"
    if use_gat:
        return "curriculum_gat_attention"
    return "curriculum_attention"


def prepare_no_curriculum_output_paths(output_root: str, ablation_tag: str) -> Dict[str, str]:
    """Create isolated save paths for no-curriculum ablation runs."""
    run_stamp = get_timestamp()
    safe_tag = _sanitize_tag(ablation_tag)

    model_dir = os.path.join(output_root, "models", safe_tag, run_stamp)
    result_dir = os.path.join(output_root, "results", safe_tag)

    ensure_dir_exists(model_dir)
    ensure_dir_exists(result_dir)

    return {
        "run_stamp": run_stamp,
        "model_dir": model_dir,
        "result_dir": result_dir,
        "leader_model_path": os.path.join(model_dir, "Path_SAC_actor_L1.pth"),
        "follower_model_path": os.path.join(model_dir, "Path_SAC_actor_F1.pth"),
        "training_result_path": os.path.join(result_dir, f"MASAC_{safe_tag}_{run_stamp}.pkl")
    }


# ==============================================================================
# 模块: 无课程学习结果目录路由工具
# ==============================================================================

def prepare_no_curriculum_result_roots(output_root: str, ablation_tag: str) -> Dict[str, str]:
    """Create isolated result roots for no-curriculum experiments/tests."""
    safe_tag = _sanitize_tag(ablation_tag)
    result_dir = os.path.join(output_root, "results", safe_tag)
    test_results_base = os.path.join(result_dir, "test_results")

    ensure_dir_exists(result_dir)
    ensure_dir_exists(test_results_base)

    return {
        "result_dir": result_dir,
        "test_results_base": test_results_base,
        "training_results_file_prefix": os.path.join(result_dir, f"MASAC_{safe_tag}")
    }


def infer_ablation_tag_from_model_path(model_path: str) -> str:
    """Infer experiment tag from model path like models/<tag>/... when possible."""
    if not model_path:
        return ""

    normalized = os.path.normpath(model_path)
    path_parts = normalized.replace("\\", "/").split("/")

    if "models" in path_parts:
        models_idx = path_parts.index("models")
        if models_idx + 1 < len(path_parts):
            candidate_tag = path_parts[models_idx + 1]
            return _sanitize_tag(candidate_tag, default_tag="")

    return ""


# ==============================================================================
# 模块: 课程学习结果目录路由工具
# ==============================================================================

def prepare_curriculum_result_roots(output_root: str, use_attention: bool, use_gat: bool = False) -> Dict[str, str]:
    """Create isolated result roots for curriculum experiments."""
    safe_tag = _default_curriculum_tag(use_attention=use_attention, use_gat=use_gat)
    result_dir = os.path.join(output_root, "results", safe_tag)
    # GAT curriculum 测试结果直接放到结果根目录下，避免再额外创建一层 test_results 目录。
    test_results_base = result_dir if (use_attention and use_gat) else os.path.join(result_dir, "test_results")

    ensure_dir_exists(result_dir)
    ensure_dir_exists(test_results_base)

    return {
        "result_dir": result_dir,
        "test_results_base": test_results_base,
        "training_results_file_prefix": os.path.join(result_dir, f"MASAC_{safe_tag}")
    }

def convert_to_json_compatible(obj):
    """将对象转换为JSON兼容格式
    
    处理numpy数组、列表、字典等数据类型，使其可以被JSON序列化
    
    Args:
        obj: 需要转换的对象
        
    Returns:
        转换后的JSON兼容对象
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_compatible(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_compatible(value) for key, value in obj.items()}
    else:
        return obj


# ==============================================================================
# 模块: 测试结果索引生成
# ==============================================================================

def create_test_results_index():
    """创建测试结果索引文件
    
    生成一个HTML文件，列出所有测试结果，便于浏览
    """
    # 确保测试结果目录存在
    ensure_dir_exists(TEST_RESULTS_BASE)
    
    # 查找所有测试结果目录
    test_dirs = []
    for item in os.listdir(TEST_RESULTS_BASE):
        item_path = os.path.join(TEST_RESULTS_BASE, item)
        if os.path.isdir(item_path):
            # 获取目录信息
            try:
                # 检查是否有JSON结果文件
                json_files = [f for f in os.listdir(item_path) if f.endswith('.json')]
                info_file = os.path.join(item_path, "test_info.json")
                
                if os.path.exists(info_file):
                    with open(info_file, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                    
                    # 收集测试信息
                    test_dirs.append({
                        'dir_name': item,
                        'timestamp': info.get('timestamp', ''),
                        'date': info.get('date', ''),
                        'config': info.get('config', {}),
                        'images': [f for f in os.listdir(item_path) if f.endswith('.png')],
                        'success_rate': info.get('success_rate', 'N/A'),
                        'path': item_path
                    })
            except Exception as e:
                print(f"处理目录 {item_path} 时出错: {e}")
    
    # 按时间戳排序
    test_dirs.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # 生成HTML内容
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>测试结果索引</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .img-link {{ margin-right: 10px; }}
        </style>
    </head>
    <body>
        <h1>测试结果索引</h1>
        <p>生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>共找到 {len(test_dirs)} 个测试结果</p>
        
        <table>
            <tr>
                <th>测试日期</th>
                <th>配置</th>
                <th>成功率</th>
                <th>结果图片</th>
                <th>详细信息</th>
            </tr>
    """
    
    for test in test_dirs:
        config_str = ""
        config = test.get('config', {})
        if config:
            config_str = f"友方:{config.get('hero_count', 'N/A')}, " \
                        f"敌方:{config.get('enemy_count', 'N/A')}, " \
                        f"障碍:{config.get('obstacle_count', 'N/A')}"
            if config.get('uav_speed', 'N/A') != 'N/A':
                config_str += f", 速度:{config['uav_speed']}"
        
        # 图片链接
        img_links = ""
        for img in test.get('images', []):
            img_path = os.path.join(test['path'], img).replace('\\', '/')
            img_links += f'<a href="file:///{img_path}" class="img-link" target="_blank">{img}</a>'
        
        # 详细信息链接
        info_link = os.path.join(test['path'], "test_info.json").replace('\\', '/')
        result_link = os.path.join(test['path'], "test_results.json").replace('\\', '/')
        
        html_content += f"""
            <tr>
                <td>{test.get('date', test.get('timestamp', 'N/A'))}</td>
                <td>{config_str}</td>
                <td>{test.get('success_rate', 'N/A')}</td>
                <td>{img_links}</td>
                <td>
                    <a href="file:///{info_link}" target="_blank">测试信息</a> | 
                    <a href="file:///{result_link}" target="_blank">详细结果</a>
                </td>
            </tr>
        """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    # 保存HTML文件
    index_path = os.path.join(TEST_RESULTS_BASE, "index.html")
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"测试结果索引已生成: {index_path}")
    return index_path


# ==============================================================================
# 模块: 任务信息打印工具
# ==============================================================================

def print_task_details(task, title="任务详情"):
    """打印任务的详细信息
    
    增强了对固定任务的支持
    
    Args:
        task: 要打印的任务
        title: 显示标题
    """
    # 识别是否为固定任务
    is_fixed_task = 'task_' not in task.id
    task_type = "固定任务" if is_fixed_task else "动态任务"
    
    log(f"\n=== {title} ({task_type}) ===", LOG_INFO)
    log(f"任务ID: {task.id}", LOG_INFO)
    log(f"难度: {task.difficulty:.2f}", LOG_INFO)
    
    # 打印环境参数
    log("环境参数:", LOG_INFO)
    log(f"  - 友方无人机: {task.env_params.get('hero_count', 1)}", LOG_INFO)
    log(f"  - 敌方无人机: {task.env_params.get('enemy_count', 1)}", LOG_INFO)
    log(f"  - 障碍物数量: {task.env_params.get('obstacle_count', 1)}", LOG_INFO)
    uav_speed_param = task.env_params.get('uav_speed')
    if uav_speed_param is not None:
        log(f"  - 无人机速度 (任务指定): {uav_speed_param}", LOG_INFO)
    else:
        log(f"  - 无人机速度 (Agent默认): 由Agent自行随机初始化", LOG_INFO)
    # 如果有性能历史，打印最近的性能
    if task.performance_history:
        recent = task.performance_history[-1]
        log("最近性能:", LOG_INFO)
        for k, v in recent['metrics'].items():
            log(f"  - {k}: {v:.2f}", LOG_INFO)
    
    # 如果是固定任务，显示特殊提示
    if is_fixed_task:
        log("注意: 这是预定义的固定难度梯度任务，使用宽松的评估标准", LOG_INFO)
    
    log("=" * (len(title) + 10), LOG_INFO)
    log("", LOG_INFO)


# ==============================================================================
# 模块: 课程学习训练流程
# ==============================================================================

def run_with_curriculum(args, initial_n_agent, initial_m_enemy, seed=42, run_id=None):
    """使用课程学习训练MASAC
    
    Args:
        args: 命令行参数
        initial_n_agent: 初始友方数量
        initial_m_enemy: 初始敌方数量
        seed: 随机种子
        run_id: 运行ID，用于多次训练时区分不同的运行
    """
    # 设置随机种子
    set_seed(seed)
    use_attention = not bool(getattr(args, "disable_attention", False))
    use_gat = bool(getattr(args, "gat", False))
    print(f"设置随机种子: {seed}")
    if run_id is not None:
        print(f"训练运行ID: {run_id}")
    
    # 确保结果目录存在
    global RESULTS_DIR, TEST_RESULTS_BASE, TRAINING_RESULTS_FILE
    ensure_dir_exists(RESULTS_DIR)
    ensure_dir_exists(TEST_RESULTS_BASE)
    ensure_dir_exists(os.path.dirname(TRAINING_RESULTS_FILE))
    
    # 初始化历史记录列表
    alpha_history = []
    reward_history = []
    success_rate_history = []
    
    # 初始化训练记录数组
    all_ep_r = [[] for _ in range(TRAIN_NUM)]
    all_ep_r0 = [[] for _ in range(TRAIN_NUM)]
    all_ep_r1 = [[] for _ in range(TRAIN_NUM)]
    k = 0  # 使用索引0，因为TRAIN_NUM=1
    
    # 为每次课程训练创建独立模型目录，避免覆盖历史实验。
    if use_attention:
        arch_tag = "gat_attention" if use_gat else "attention"
        result_prefix = "MASAC_curriculum_gat_attention" if use_gat else "MASAC_curriculum"
    else:
        arch_tag = "no_attention"
        result_prefix = "MASAC_curriculum_no_attention"
    run_suffix = f"_run{run_id}" if run_id is not None else ""
    output_root = getattr(args, "ablation_output_root", None) or os.path.dirname(os.path.abspath(__file__))
    if use_attention:
        models_root_dir = os.path.join(output_root, "models")
    else:
        models_root_dir = os.path.join(output_root, "models", "ablation_curriculum_no_attention")
    models_base_dir = os.path.join(
        models_root_dir,
        f"curriculum_{arch_tag}_{get_timestamp()}{run_suffix}_seed{seed}"
    )

    # 创建模型基础目录
    os.makedirs(models_base_dir, exist_ok=True)
    log(f"课程学习模型目录: {models_base_dir}", LOG_INFO)
    
    # 设置日志级别
    set_log_level(LOG_DEBUG)  # 从INFO改为DEBUG级别，显示更详细的日志信息
    
    # 清除之前的日志历史
    if hasattr(globals(), 'clear_log_history'):
        clear_log_history()
    
    # 确保模型保存目录存在
    log(f"模型将保存在: {models_base_dir}", LOG_INFO)
    
    # 创建课程学习组件
    config = CurriculumConfig()
    
    # 修改变化范围配置
    config.set("task_generator.variation_ranges", {
        "hero_count": (1, 3),       # 友方无人机数量1-3
        "enemy_count": (1, 5),      # 敌方无人机数量1-5
        "obstacle_count": (0, 10),   # 障碍物数量0-10
        "map_size": (700, 1000),     # 地图尺寸
        "target_distance": (200, 600), # 目标距离
        "uav_speed": (10, 20)       # 无人机速度10-20
    })
    
    print("智能体奖励处理说明:")
    print("- 环境返回的奖励包含所有智能体 (友方+敌方) 的奖励")
    print("- 控制器会根据实际友方智能体数量自动调整，并从中提取所需的奖励")
    print("- 当智能体数量从1主机+1从机变为1主机+2从机时，控制器会自动适应")
    
    # 增加课程步骤和每个任务的训练轮数
    config.set("curriculum.max_curriculum_steps", 6)  # 从15减少到2
    config.set("curriculum_manager.max_episodes_per_task", 300)  # 设置为200
    
    # 设置评估窗口大小和稳定性阈值
    config.set("curriculum_manager.evaluation_window", 20)  # 从20减少到5
    config.set("curriculum_manager.min_training_rounds", 60)  # 从40减少到5
    config.set("curriculum_manager.reward_stability_threshold", 0.6)  # 从0.75降低到0.5
    config.set("curriculum_manager.success_rate_threshold", 0.8)  # 从0.8降低到0.5

    # 添加停滞检测相关配置
    config.set("curriculum_manager.progress_threshold", 0.05)  # 学习进度阈值(从0.01提高到0.05)
    config.set("curriculum_manager.stagnation_threshold", 3)  # 连续停滞检测阈值
    
    # 在配置中设置渲染选项，确保所有任务都使用同样的渲染设置
    config.set("render", RENDER)
    print(f"渲染设置: {'开启' if RENDER else '关闭'}")
    
    # 修改初始难度范围，使其更广泛
    config.set("curriculum_manager.initial_difficulty_range", (0.0, 0.3))  # 从(0.0, 0.1)扩展到(0.0, 0.3)
    
    # 设置固定任务的相关配置参数
    config.set("use_fixed_tasks", True) # 启用了固定任务
    print("固定任务集已启用")
    
    # 定义预设的固定任务难度级别
    # 这些值应该与FixedTaskGenerator中的SPECIFIC_TASKS_CONFIG任务数量匹配
    predefined_task_difficulties = [0.1, 0.2, 0.3, 0.4, 0.5]
    config.set("fixed_tasks_config.difficulty_levels", predefined_task_difficulties)
    log(f"已为FixedTaskGenerator配置预定义难度级别: {predefined_task_difficulties}", LOG_INFO)
    
    # 使用固定难度梯度任务生成器
    print("使用固定难度梯度任务生成器...")
    task_generator = FixedTaskGenerator(config)
    task_sequencer = LinearTaskSequencer(config)
    knowledge_transfer = PolicyTransfer(config)
    
    # 创建课程管理器
    curriculum_manager = CurriculumManager(
        config=config,
        task_generator=task_generator,
        task_sequencer=task_sequencer,
        knowledge_transfer=knowledge_transfer
    )
    
    # 初始化课程
    initial_task = curriculum_manager.initialize()
    print_task_details(initial_task, "初始任务详情")
    
    # 根据任务创建环境
    env = initial_task.create_env()
    
    # 训练模式下设置dt=1.0
    env.set_time_step(1.0)
    print(f"训练模式：时间步长dt设置为1.0")
    
    # 创建MASAC控制器
    n_agents = initial_n_agent + initial_m_enemy
    state_dim = state_number
    action_dim = action_number
    print("初始化MASAC控制器...")
    if use_attention:
        print(f"课程学习网络结构: {'GAT注意力' if use_gat else '注意力'}")
    else:
        print("课程学习网络结构: 无注意力")
    masac_controller = MASACController(
        n_agents=n_agents, 
        state_dim=state_dim, 
        action_dim=action_dim, 
        device=device,
        memory_capacity=MemoryCapacity,
        max_replay_ratio=20,  # 允许较高的重放比例以减轻过拟合
        use_attention=use_attention,
        use_gat = use_gat
    )
    
    # 添加奖励分配验证函数
    def verify_reward_allocation(rewards, n_friendly_agents):
        """验证奖励分配是否正确
        
        Args:
            rewards: 环境返回的奖励字典 {"leader": r_l, "followers": [r_f1, ...]} 或旧的扁平化数组
            n_friendly_agents: 友方智能体数量
            
        Returns:
            bool: 奖励分配是否有效
        """
        all_reward_values = []
        total_rewards_received = 0

        if isinstance(rewards, dict) and "leader" in rewards and "followers" in rewards:
            # 处理结构化奖励字典
            leader_reward = rewards.get("leader", 0.0)
            follower_rewards = rewards.get("followers", [])
            
            # 确保 leader_reward 是数值
            if isinstance(leader_reward, (int, float, np.number)):
                all_reward_values.append(float(leader_reward))
            
            # 确保 follower_rewards 是列表，且内部是数值
            if isinstance(follower_rewards, list):
                for r in follower_rewards:
                    if isinstance(r, (int, float, np.number)):
                        all_reward_values.append(float(r))
            
            # 结构化数据中，友方奖励数量 = 1 (leader) + len(followers)
            # 注意：这里我们不再严格检查 n_friendly_agents 和接收到的奖励数量是否完全匹配，
            # 因为结构化奖励明确区分了leader和followers，数量不匹配可能是环境设计问题
            total_rewards_received = 1 + len(follower_rewards)
            # 可以在这里添加更灵活的检查，例如 n_friendly_agents 是否等于 total_rewards_received
            if n_friendly_agents != total_rewards_received:
                log(f"[信息] 结构化奖励数量({total_rewards_received})与友方智能体数量({n_friendly_agents})不一致", LOG_INFO)

        elif isinstance(rewards, (np.ndarray, list)):
            # 尝试处理扁平化数组/列表
            try:
                rewards_array = np.asarray(rewards, dtype=float)
                all_reward_values = rewards_array.flatten().tolist()
                total_rewards_received = len(all_reward_values)
                # 扁平化数据下，检查长度
                if total_rewards_received < n_friendly_agents:
                    log(f"[信息] 扁平奖励数组长度({total_rewards_received})小于友方智能体数量({n_friendly_agents})", LOG_INFO)
            except (TypeError, ValueError):
                log(f"[错误] 无法将奖励转换为数值数组: {rewards}", LOG_ERROR)
                return False
        else:
            # 不支持的奖励格式
            log(f"[错误] 不支持的奖励格式: {type(rewards)}", LOG_ERROR)
            return False
            
        # 如果没有提取到任何有效的奖励值
        if not all_reward_values:
            log("[警告] 未能从奖励数据中提取任何有效数值", LOG_WARNING)
            # 根据情况决定是否返回 True 或 False，这里暂时返回 True，允许空奖励
            return True 
        
        # 将提取的奖励值转换为 NumPy 数组进行检查
        reward_values_np = np.array(all_reward_values)
        
        # 检查奖励是否包含NaN值
        if np.isnan(reward_values_np).any():
            log(f"[错误] 奖励包含NaN值: {reward_values_np}", LOG_ERROR)
            return False
            
        # 检查奖励值是否在合理范围
        if np.max(np.abs(reward_values_np)) > 1000:
            log(f"[警告] 奖励值超出正常范围: {reward_values_np}", LOG_WARNING)
            # 这只是警告，不是错误
                
        return True
    
    # 添加奖励记录功能
    reward_allocation_issues = []  # 记录奖励分配问题
    reward_distribution_stats = []  # 记录奖励分布统计信息
    
    # 添加奖励分布分析函数
    def analyze_reward_distribution(reward, n_friendly_agents, episode, timestep):
        """分析奖励分布情况，记录主机和从机奖励的比例和统计特性
        
        Args:
            reward: 环境返回的奖励字典 {"leader": r_l, "followers": [r_f1, ...]}
            n_friendly_agents: 友方智能体数量
            episode: 当前回合数
            timestep: 当前时间步
            
        Returns:
            dict or None: 奖励分布统计信息，如果输入无效则返回None
        """
        # 强制要求输入为结构化字典
        if not (isinstance(reward, dict) and "leader" in reward and "followers" in reward):
            log(f"[错误] analyze_reward_distribution 期望结构化奖励字典，收到: {type(reward)}", LOG_ERROR)
            return None
        
        leader_reward = reward.get("leader", 0.0)
        follower_rewards_raw = reward.get("followers", []) # 使用新变量名
        
        # 确保提取的值是数值
        all_reward_values = []
        valid_leader = False
        if isinstance(leader_reward, (int, float, np.number)):
            all_reward_values.append(float(leader_reward))
            valid_leader = True
        else:
             log(f"[警告] Leader 奖励不是有效数值: {leader_reward}", LOG_WARNING)
             leader_reward = 0.0 # 使用默认值
             
        valid_follower_rewards = []
        if isinstance(follower_rewards_raw, list):
            for r in follower_rewards_raw:
                if isinstance(r, (int, float, np.number)):
                    all_reward_values.append(float(r))
                    valid_follower_rewards.append(float(r))
                else:
                    log(f"[警告] Follower 奖励包含无效数值: {r}", LOG_WARNING)
        else:
             log(f"[警告] Followers 奖励不是列表: {follower_rewards_raw}", LOG_WARNING)
             # follower_rewards_raw = [] # 这里不需要重置，因为后面会检查 all_reward_values

        # 如果没有有效的奖励值
        if not all_reward_values:
            log("[警告] analyze_reward_distribution 未找到有效奖励值", LOG_WARNING)
            # 返回一个带有默认值的字典可能比返回None更好
            return {
                'episode': episode,
                'timestep': timestep,
                'n_friendly': n_friendly_agents,
                'total_agents_rewarded': 0,
                'reward_mean': 0.0, 'reward_std': 0.0, 'reward_min': 0.0, 'reward_max': 0.0,
                'friendly_mean': 0.0, 'friendly_std': 0.0
            }

        # 计算整体统计数据
        reward_values_np = np.array(all_reward_values)
        stats = {
            'episode': episode,
            'timestep': timestep,
            'n_friendly': n_friendly_agents,
            'total_agents_rewarded': len(all_reward_values), # 实际收到奖励的智能体数
            'reward_mean': float(np.mean(reward_values_np)),
            'reward_std': float(np.std(reward_values_np)),
            'reward_min': float(np.min(reward_values_np)),
            'reward_max': float(np.max(reward_values_np)),
        }
        
        # 计算友方（Leader + Followers）统计信息
        # 假设 all_reward_values 只包含友方奖励 (Leader + Followers)
        stats['friendly_mean'] = float(np.mean(reward_values_np))
        stats['friendly_std'] = float(np.std(reward_values_np))

        # 记录具体的 Leader 和 Follower 奖励
        if valid_leader:
            stats['leader_reward'] = float(leader_reward)
        if valid_follower_rewards: # 使用处理后的有效奖励列表
            stats['follower_rewards'] = valid_follower_rewards
            
        # 打印主机和从机奖励信息 (保持不变)
        if episode % 50 == 0 and timestep == 0: 
            log(f"奖励分布分析 (EP {episode}, 步 {timestep}):", LOG_INFO)
            if 'leader_reward' in stats:
                log(f"  领导者奖励: {stats['leader_reward']:.2f}", LOG_INFO)
            if 'follower_rewards' in stats:
                log(f"  跟随者奖励: {stats['follower_rewards']}", LOG_INFO)
            if 'friendly_mean' in stats:
                log(f"  友方平均/标准差: {stats['friendly_mean']:.2f}/{stats['friendly_std']:.2f}", LOG_INFO)
        
        return stats
    
    # 在外层循环检查是否需要跳出
    break_outer = False
    for curriculum_step in range(curriculum_manager.max_curriculum_steps):
        print(f"\n课程步骤 {curriculum_step+1}/{curriculum_manager.max_curriculum_steps}")
        
        # 添加这个检查，确保当前任务有效
        if curriculum_manager.get_current_task() is None:
            print("没有更多有效任务，课程学习已完成！")
            break
        
        # 获取当前任务的阶段标签和编号
        current_task = curriculum_manager.get_current_task()
        current_task_id = current_task.id if current_task and hasattr(current_task, 'id') else f"stage_{curriculum_step}"
        current_stage_tag_for_memory = current_task_id
        current_stage_number_for_memory = curriculum_step  # 使用循环索引作为阶段编号
        
        log(f"当前课程阶段: '{current_stage_tag_for_memory}' (编号 {current_stage_number_for_memory})", LOG_INFO)
        
        # 在每个任务上进行训练
        for episode in range(curriculum_manager.max_episodes_per_task):
            print(f"任务 {episode+1}/{curriculum_manager.max_episodes_per_task}")
            
            # 重置环境
            observation = env.reset()
            total_reward = 0
            reward_totle0 = 0 # 初始化领导者回合奖励累加器
            reward_totle1 = 0 # 初始化第一个跟随者回合奖励累加器
            done = False
            step_count = 0
            team_formation_time = 0
            last_distance = None

            # 在每个时间步上进行训练
            while not done and step_count < EP_LEN:

                # 1. 延长探索周期，前 80% 的回合都保持特定优先级的探索
                max_ep = curriculum_manager.max_episodes_per_task
                should_add_noise = episode < int(max_ep * 0.8)

                # 2. 动态衰减噪声比例：前期大步试错 (0.1)，后期精细微调 (逐步降至 0.02)
                current_noise_scale = 0.1 * max(0.2, 1.0 - (episode / max_ep))

                # 选择动作
                action = masac_controller.select_actions(
                    observation,
                    add_noise=should_add_noise,
                    noise_scale=current_noise_scale,
                    evaluate=False
                )
                
                # 执行动作
                observation_, reward, done, win, team_counter, dis = env.step(action)
                
                # 存储经验到回放缓冲区，传递阶段标签
                masac_controller.store_transition(
                    observation, action, reward, observation_, done,
                    current_stage_tag=current_stage_tag_for_memory
                )
                
                # 记录最后一步的距离
                last_distance = dis
                
                # 更新状态和统计
                observation = observation_
                # 累加当前时间步的总奖励 (Leader + Followers)
                current_step_reward = 0.0
                if isinstance(reward, dict):
                    leader_r = reward.get("leader", 0.0)
                    followers_r = reward.get("followers", [])
                    # 确保是数值
                    if isinstance(leader_r, (int, float, np.number)):
                         current_step_reward += float(leader_r)
                    if isinstance(followers_r, list):
                        for r in followers_r:
                             if isinstance(r, (int, float, np.number)):
                                 current_step_reward += float(r)
                total_reward += current_step_reward 
                step_count += 1
                
                # 计算编队时间
                if team_counter > 0:
                    team_formation_time += 1
                
                # 渲染环境
                if RENDER:
                    env.render()
                
                # 如果缓冲区足够大，开始学习
                # 使用10%的经验池大小作为开始学习的阈值
                training_start_size = int(MemoryCapacity * 0.3)
                if len(masac_controller.memory.buffer) > training_start_size:  # 使用实际的缓冲区长度
                    try:
                        masac_controller.train(
                            batch_size=BATCH,
                            current_stage_tag=current_stage_tag_for_memory,
                            current_stage_number=current_stage_number_for_memory
                        )
                    except Exception as e:
                        print(f"训练过程中出现错误: {e}")
                        import traceback
                        traceback.print_exc()
                
                # 更新状态和统计 - 这部分 observation = observation_ 已在上文处理
                # observation = observation_
                
                # 分别计算各个智能体的奖励 - 原 UnboundLocalError 行已删除
                # reward_totle += reward.mean() if isinstance(reward, np.ndarray) else reward
                
                # 确保安全地获取奖励值并累加到 reward_totle0 (Leader) 和 reward_totle1 (First Follower)
                if isinstance(reward, dict):
                    # Leader 奖励
                    leader_r_val = reward.get("leader")
                    if isinstance(leader_r_val, (int, float, np.number)):
                        reward_totle0 += float(leader_r_val)

                    # First Follower 奖励
                    followers_r_list = reward.get("followers")
                    if isinstance(followers_r_list, list) and len(followers_r_list) > 0:
                        first_follower_r_val = followers_r_list[0]
                        if isinstance(first_follower_r_val, (int, float, np.number)):
                            reward_totle1 += float(first_follower_r_val)
                
                # 渲染环境
                if RENDER:
                    env.render()
                
                # 结束判断
                if done:
                    # 显示回合结束状态
                    if win:
                        print(f"回合 {episode+1} 成功! 智能体到达目标!")
                    else:
                        print(f"回合 {episode+1} 失败")
                    break
            
            # 收集温度系数 Alpha 统计数据
            alpha_stats = []
            if hasattr(masac_controller, 'entroy_leader') and n_agents >= 1:
                alpha_stats.append(masac_controller.entroy_leader.get_alpha_stats())
            if hasattr(masac_controller, 'entroy_follower') and n_agents > 1:
                # 假设所有follower共享一个entroy对象
                # 如果需要每个follower独立，需要修改MASACController
                alpha_stats.append(masac_controller.entroy_follower.get_alpha_stats())
            
            avg_alpha = np.mean([stat["current"] for stat in alpha_stats]) if alpha_stats else 0.0
            alpha_history.append(avg_alpha)
            
            # 获取并打印各个飞机的速度和编队率
            leader_speeds = [f"{leader.speed:.1f}" for leader in env.entity_manager.leaders]
            follower_speeds = [f"{follower.speed:.1f}" for follower in env.entity_manager.followers]
            formation_rate = env.entity_manager.get_formation_rate()
            
            # 创建综合信息并使用log函数打印
            if episode % 10 == 0 or win:
                status = "成功" if win else "进行中"
                # 整合所有回合信息到一个日志消息中
                log_message = (
                    f"回合摘要 - 任务: {curriculum_step+1}/{curriculum_manager.max_curriculum_steps}, "
                    f"回合: {episode+1}/{curriculum_manager.max_episodes_per_task}, "
                    f"总回合: {curriculum_manager.total_episodes}, "
                    f"状态: {status}, 步数: {step_count}, 总奖励: {total_reward:.1f}, " # 在这里添加了步数
                    f"主机速度: [{', '.join(leader_speeds)}], "
                    f"从机速度: [{', '.join(follower_speeds)}], "
                    f"编队率: {formation_rate:.2f}, "
                    f"Alpha: {avg_alpha:.4f}, 重放比例: {masac_controller.replay_ratio}"
                )
                log(log_message, LOG_INFO)
            
            # 记录奖励
            all_ep_r[k].append(total_reward) # 使用 total_reward 记录总奖励
            all_ep_r0[k].append(reward_totle0)
            all_ep_r1[k].append(reward_totle1)
            
            # 记录历史数据
            reward_history.append(total_reward)
            success_rate_history.append(float(win))
            
            # 更新任务性能
            metrics = {
                'reward': total_reward,
                'success_rate': float(win),
                'team_coordination': team_counter  # 编队保持率
            }
            # 检查返回值，如果为True则说明达到最大总回合数，需终止训练
            if curriculum_manager.update_task_performance(metrics):
                print("达到总回合数限制，终止训练！")
                # 保存最终模型
                final_save_dir = f"{models_base_dir}/final"
                os.makedirs(final_save_dir, exist_ok=True)
                final_save_path = f"{final_save_dir}/final_model"
                try:
                    masac_controller.save_models(final_save_path)
                    print(f"最终模型已保存到: {final_save_path}")
                except Exception as e:
                    print(f"保存最终模型失败: {e}")
                    traceback.print_exc()
                # 直接跳出两层循环
                break_outer = True
                break
            
            # 检查是否需要切换任务
            if curriculum_manager.should_switch_task():
                log(f"任务切换，完成 {episode+1} 轮训练", LOG_INFO)
                
                # 保存任务完成时的模型
                task_complete_dir = f"{models_base_dir}/curriculum_step{curriculum_step}_complete"
                os.makedirs(task_complete_dir, exist_ok=True)
                task_complete_path = f"{task_complete_dir}/model_ep{episode}"
                try:
                    masac_controller.save_models(task_complete_path)
                    log(f"模型已保存到: {task_complete_path}", LOG_INFO)
                except Exception as e:
                    log(f"保存模型失败: {e}", LOG_ERROR)
                    traceback.print_exc()
                break
            
            # 定期保存模型 - 改为每100回合保存一次
            if episode % 100 == 0 and episode > 0:
                # 创建专门的保存目录
                save_dir = os.path.join(models_base_dir, f"curriculum_step{curriculum_step}")
                os.makedirs(save_dir, exist_ok=True)
                
                # 构建完整的保存路径
                save_path = f"{save_dir}/model_ep{episode}"
                try:
                    masac_controller.save_models(save_path)
                    print(f"模型已保存到: {save_path}")
                except Exception as e:
                    print(f"保存模型失败: {e}")
                    traceback.print_exc()
        
        # 在外层循环检查是否需要跳出
        if break_outer:
            print("由于达到最大总回合数限制，终止整个课程训练")
            break
                
        # 准备参数以进行知识迁移
        current_policy_and_critic_params = masac_controller.get_policy_parameters_for_curriculum()

        # 定义一个临时的包装器类，仅用于传递参数给 PolicyTransfer
        class TempPolicyWrapperForCurriculum:
            def __init__(self, params):
                self._params = params
            def get_parameters(self):
                return self._params

        policy_wrapper_for_transfer = TempPolicyWrapperForCurriculum(current_policy_and_critic_params)

        # 获取下一个任务和迁移后的参数字典
        # 假设 curriculum_manager.get_next_task 内部的 PolicyTransfer.transfer 返回参数字典
        next_task, transferred_params_dict = curriculum_manager.get_next_task(policy_wrapper_for_transfer)

        if next_task is None:
            print("没有更多任务，课程完成!")
            # ... (保存最终模型的逻辑不变) ...
            final_save_dir = f"{models_base_dir}/final"
            os.makedirs(final_save_dir, exist_ok=True)
            final_save_path = f"{final_save_dir}/final_model"
            try:
                masac_controller.save_models(final_save_path) # 使用 save_models
                print(f"最终模型已保存到: {final_save_path}")
            except Exception as e:
                print(f"保存最终模型失败: {e}")
                traceback.print_exc()
                
            # 保存最后一个任务的训练结果
            all_ep_r_mean = np.mean((np.array(all_ep_r)), axis=0)
            all_ep_r_std = np.std((np.array(all_ep_r)), axis=0)
            all_ep_L_mean = np.mean((np.array(all_ep_r0)), axis=0)
            all_ep_L_std = np.std((np.array(all_ep_r0)), axis=0)
            all_ep_F_mean = np.mean((np.array(all_ep_r1)), axis=0)
            all_ep_F_std = np.std((np.array(all_ep_r1)), axis=0)
            
            # 保存训练结果
            d = {
                "all_ep_r_mean": all_ep_r_mean, 
                "all_ep_r_std": all_ep_r_std,
                "all_ep_L_mean": all_ep_L_mean, 
                "all_ep_L_std": all_ep_L_std,
                "all_ep_F_mean": all_ep_F_mean, 
                "all_ep_F_std": all_ep_F_std,
                "alpha_history": alpha_history,
                "reward_history": reward_history,
                "success_rate_history": success_rate_history
            }
            
            # 创建唯一的结果文件名（添加时间戳和课程步骤）
            timestamp = get_timestamp()
            seed_suffix = f"_seed{seed}" if seed != 42 else ""
            run_suffix = f"_run{run_id}" if run_id is not None else ""
            # 确保最后一个任务使用正确的步骤编号（curriculum_step + 1，而不是 curriculum_step）
            result_filename = f"{result_prefix}_{timestamp}{run_suffix}{seed_suffix}_step{curriculum_step+1}.pkl"
            result_path = os.path.join(RESULTS_DIR, result_filename)
            
            # 确保结果目录存在
            ensure_dir_exists(RESULTS_DIR)
            log(f"保存最终任务训练结果到 {result_path}", LOG_INFO)
            
            try:
                with open(result_path, 'wb') as f:
                    pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)
                log(f"最终任务训练结果已成功保存", LOG_INFO)
            except Exception as e:
                log(f"保存最终任务训练结果时出错: {e}", LOG_ERROR)
                import traceback
                traceback.print_exc()
            
            # 绘制训练曲线
            import matplotlib.pyplot as plt
            
            # 创建一个包含两个子图的图形
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # 获取实际有数据的回合数
            actual_episodes = len(all_ep_r_mean)
            print(f"实际训练回合数: {actual_episodes}")
            
            # 绘制第一个子图：奖励曲线
            x_range = np.arange(actual_episodes)
            ax1.plot(x_range, all_ep_r_mean[:actual_episodes], label='Average Reward')
            ax1.fill_between(x_range,
                             all_ep_r_mean[:actual_episodes] - all_ep_r_std[:actual_episodes], 
                             all_ep_r_mean[:actual_episodes] + all_ep_r_std[:actual_episodes],
                             alpha=0.1, color='blue')
            
            ax1.set_title('MASAC with Curriculum Learning - Rewards')
            ax1.set_ylabel('Moving averaged episode reward')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # 绘制第二个子图：温度系数Alpha曲线
            if alpha_history:
                alpha_x = np.arange(len(alpha_history))
                ax2.plot(alpha_x, alpha_history, color='green', label='Alpha')
                ax2.set_title('Temperature Coefficient Alpha')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Alpha value')
                ax2.legend()
                ax2.grid(True, linestyle='--', alpha=0.7)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表 - 使用与训练结果数据文件匹配的文件名（不含扩展名）
            plot_filename = f"{result_prefix}_{timestamp}{run_suffix}{seed_suffix}_step{curriculum_step+1}.png"
            plot_path = os.path.join(RESULTS_DIR, plot_filename)
            
            try:
                plt.savefig(plot_path)
                log(f"最终任务训练曲线已保存到: {plot_path}", LOG_INFO)
            except Exception as e:
                log(f"保存最终任务训练曲线时出错: {e}", LOG_ERROR)
                import traceback
                traceback.print_exc()
            finally:
                plt.close()  # 确保图表资源被释放
                
            break # 跳出 curriculum_step 循环
        
        # 使用迁移后的参数更新控制器
        if transferred_params_dict is not None:
            # PolicyTransfer._do_transfer 现在返回参数字典
            actual_params_to_pass = transferred_params_dict
            
            # 如果是包装器类型（兼容旧版本）
            if isinstance(transferred_params_dict, TempPolicyWrapperForCurriculum):
                log("检测到 TempPolicyWrapperForCurriculum，正在提取内部参数进行迁移。", LOG_DEBUG)
                actual_params_to_pass = transferred_params_dict.get_parameters()
            elif isinstance(transferred_params_dict, dict):
                # 这是预期的返回类型 - 参数字典
                log("知识迁移返回了参数字典（正确的行为）", LOG_INFO)
                # 检查是否有智能体数量信息
                if 'agent_counts' in transferred_params_dict:
                    agent_counts = transferred_params_dict.get('agent_counts', {})
                    log(f"智能体数量变化: {agent_counts.get('source', 'N/A')} -> {agent_counts.get('target', 'N/A')}", LOG_DEBUG)
            else:
                # 如果不是包装器也不是字典，记录警告
                log(f"警告: 知识迁移返回的参数类型为 {type(transferred_params_dict)}，期望是 dict。继续尝试，但可能导致错误。", LOG_WARNING)
                
            # 使用正确的参数调用 update_components_from_transfer
            try:
                masac_controller.update_components_from_transfer(actual_params_to_pass)
                log("已成功从迁移的参数更新 MASACController 组件。", LOG_INFO)
            except Exception as e:
                log(f"从迁移参数更新组件时出错: {e}", LOG_ERROR)
                import traceback
                traceback.print_exc()
        else:
            log("知识迁移未返回有效参数，控制器组件保持不变。", LOG_INFO)
            
        # 如果智能体数量变化，控制器进行适应
        # 使用 masac_controller 内部的 n_agents 计数进行比较和更新
        new_agent_count = next_task.env_params.get("leader_count", 1) + next_task.env_params.get("follower_count", 0)
        current_agent_count = masac_controller.n_agents # 获取控制器当前的 n_agents
        if new_agent_count != current_agent_count:
            log(f"适应新的智能体数量: {current_agent_count} -> {new_agent_count}", LOG_INFO)
            masac_controller.adapt_to_agent_count(new_agent_count)
            # n_agents 变量（如果之前在 run_with_curriculum 作用域中使用）也应更新
            n_agents = new_agent_count
    
        # 更新环境到下一个任务
        env.close() # 关闭旧环境
        env = next_task.create_env()
        env.set_time_step(1.0) # 确保新环境时间步正确
        print_task_details(next_task, f"切换到新任务 (课程步骤 {curriculum_step + 2})")
            
        # 训练完成，记录结果
        all_ep_r_mean = np.mean((np.array(all_ep_r)), axis=0)
        all_ep_r_std = np.std((np.array(all_ep_r)), axis=0)
        all_ep_L_mean = np.mean((np.array(all_ep_r0)), axis=0)
        all_ep_L_std = np.std((np.array(all_ep_r0)), axis=0)
        all_ep_F_mean = np.mean((np.array(all_ep_r1)), axis=0)
        all_ep_F_std = np.std((np.array(all_ep_r1)), axis=0)
        
        # 保存训练结果
        d = {
            "all_ep_r_mean": all_ep_r_mean, 
            "all_ep_r_std": all_ep_r_std,
            "all_ep_L_mean": all_ep_L_mean, 
            "all_ep_L_std": all_ep_L_std,
            "all_ep_F_mean": all_ep_F_mean, 
            "all_ep_F_std": all_ep_F_std,
            "alpha_history": alpha_history,
            "reward_history": reward_history,
            "success_rate_history": success_rate_history
        }
        
        # 创建唯一的结果文件名（添加时间戳和课程步骤）
        timestamp = get_timestamp()
        seed_suffix = f"_seed{seed}" if seed != 42 else ""
        run_suffix = f"_run{run_id}" if run_id is not None else ""
        result_filename = f"{result_prefix}_{timestamp}{run_suffix}{seed_suffix}_step{curriculum_step+1}.pkl"
        result_path = os.path.join(RESULTS_DIR, result_filename)
        
        # 确保结果目录存在
        ensure_dir_exists(RESULTS_DIR)
        log(f"保存训练结果到 {result_path}", LOG_INFO)
        
        try:
            with open(result_path, 'wb') as f:
                pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)
            log(f"训练结果已成功保存", LOG_INFO)
        except Exception as e:
            log(f"保存训练结果时出错: {e}", LOG_ERROR)
            import traceback
            traceback.print_exc()
        
        # 绘制训练曲线
        import matplotlib.pyplot as plt
        
        # 创建一个包含两个子图的图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # 获取实际有数据的回合数
        actual_episodes = len(all_ep_r_mean)
        print(f"实际训练回合数: {actual_episodes}")
        
        # 绘制第一个子图：奖励曲线
        x_range = np.arange(actual_episodes)
        ax1.plot(x_range, all_ep_r_mean[:actual_episodes], label='Average Reward')
        ax1.fill_between(x_range,
                         all_ep_r_mean[:actual_episodes] - all_ep_r_std[:actual_episodes], 
                         all_ep_r_mean[:actual_episodes] + all_ep_r_std[:actual_episodes],
                         alpha=0.1, color='blue')
        
        ax1.set_title('MASAC with Curriculum Learning - Rewards')
        ax1.set_ylabel('Moving averaged episode reward')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制第二个子图：温度系数Alpha曲线
        if alpha_history:
            alpha_x = np.arange(len(alpha_history))
            ax2.plot(alpha_x, alpha_history, color='green', label='Alpha')
            ax2.set_title('Temperature Coefficient Alpha')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Alpha value')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表 - 使用与训练结果数据文件匹配的文件名（不含扩展名）
        plot_filename = f"{result_prefix}_{timestamp}{run_suffix}{seed_suffix}_step{curriculum_step+1}.png"
        plot_path = os.path.join(RESULTS_DIR, plot_filename)
        
        try:
            plt.savefig(plot_path)
            log(f"训练曲线已保存到: {plot_path}", LOG_INFO)
        except Exception as e:
            log(f"保存训练曲线时出错: {e}", LOG_ERROR)
            import traceback
            traceback.print_exc()
        finally:
            plt.close()  # 确保图表资源被释放
            
        # 分析课程学习过程
        analyze_curriculum_learning(curriculum_manager)
      
    # 在for curriculum_step循环结束后添加
    print("\n=========================================")
    print("课程学习训练总结:")
    print(f"- 完成的课程步骤: {curriculum_step+1}/{curriculum_manager.max_curriculum_steps}")
    print(f"- 总训练回合数: {curriculum_manager.total_episodes}")
    print(f"- 解决的任务数量: {len(curriculum_manager.task_history)}/{len(curriculum_manager.task_generator.predefined_task_configs) if hasattr(curriculum_manager.task_generator, 'predefined_task_configs') else 'N/A'}")
    print("=========================================\n")
    
    # 训练完成，记录结果
    all_ep_r_mean = np.mean((np.array(all_ep_r)), axis=0)
    
    # 构建最终训练结果并返回
    final_training_results = {
        'all_ep_r': all_ep_r,
        'success_rate_history': success_rate_history,
        'reward_history': reward_history,
        'alpha_history': alpha_history,
        'final_success_rate': np.mean(success_rate_history[-100:]) if len(success_rate_history) >= 100 else np.mean(success_rate_history) if success_rate_history else 0,
        'final_avg_reward': np.mean(all_ep_r[0][-100:]) if len(all_ep_r[0]) >= 100 else np.mean(all_ep_r[0]) if all_ep_r[0] else 0,
        'use_attention': use_attention,
        'seed': seed,
        'run_id': run_id,
        'models_base_dir': models_base_dir,
        'total_episodes': len(all_ep_r[0]) if all_ep_r[0] else 0,
        'curriculum_steps_completed': curriculum_step + 1,
        'timestamp': get_timestamp()
    }
    
    return final_training_results


# ==============================================================================
# 模块: 无课程学习训练流程
# ==============================================================================

def run_attention_no_curriculum(args, initial_n_agent, initial_m_enemy, seed=42, run_id=None):
    """使用注意力 MASAC 进行无课程学习训练。

    该流程保留 MASACController（可选注意力 Actor/Critic）训练逻辑，
    但固定在单一任务配置上，不使用 CurriculumManager 进行任务切换。
    """
    set_seed(seed)
    print(f"设置随机种子: {seed}")
    if run_id is not None:
        print(f"训练运行ID: {run_id}")

    train_episodes = max(1, int(getattr(args, "train_episodes", 500)))
    save_interval = max(1, int(getattr(args, "save_interval", 100)))
    exploration_episodes = max(0, int(getattr(args, "exploration_episodes", 20)))
    training_start_ratio = float(getattr(args, "training_start_ratio", 0.3))
    training_start_ratio = min(max(training_start_ratio, 0.0), 1.0)
    max_consecutive_nonfinite_updates = max(1, int(getattr(args, "max_consecutive_nonfinite_updates", 20)))
    share_follower_policy = bool(getattr(args, "share_follower_policy_no_curriculum", False))
    use_attention = not bool(getattr(args, "disable_attention", False))
    use_gat = bool(getattr(args, "gat", False))
    mode_name = _default_no_curriculum_tag(use_attention=use_attention, use_gat=use_gat)

    effective_ablation_tag = _sanitize_tag(getattr(args, "ablation_tag", "ablation_no_curriculum"))
    if effective_ablation_tag == "ablation_no_curriculum":
        effective_ablation_tag = _default_no_curriculum_tag(use_attention=use_attention, use_gat=use_gat)

    if use_attention:
        print(f"使用{'GAT注意力' if use_gat else '注意力'} MASAC（无课程学习）训练")
    else:
        print("使用无注意力 MASAC（无课程学习）训练")
    print(
        f"训练轮数: {train_episodes}, 保存间隔: {save_interval}, 探索轮数: {exploration_episodes}, "
        f"非有限连续阈值: {max_consecutive_nonfinite_updates}, 共享随从策略: {share_follower_policy}"
    )

    ablation_output_root = args.ablation_output_root or os.path.dirname(os.path.abspath(__file__))
    no_curriculum_paths = prepare_no_curriculum_output_paths(
        output_root=ablation_output_root,
        ablation_tag=effective_ablation_tag
    )

    model_dir = no_curriculum_paths["model_dir"]
    result_path = no_curriculum_paths["training_result_path"]
    plot_path = os.path.splitext(result_path)[0] + ".png"
    final_model_path = os.path.join(model_dir, "final_model")

    print("无课程学习输出路径:")
    print(f"- 实验标签: {effective_ablation_tag}")
    print(f"- 运行标识: {no_curriculum_paths['run_stamp']}")
    print(f"- 模型目录: {model_dir}")
    print(f"- 最终模型: {final_model_path}")
    print(f"- 结果文件: {result_path}")

    n_agents = initial_n_agent + initial_m_enemy
    state_dim = state_number
    action_dim = action_number

    alpha_history = []
    reward_history = []
    success_rate_history = []
    all_ep_r = [[] for _ in range(TRAIN_NUM)]
    all_ep_r0 = [[] for _ in range(TRAIN_NUM)]
    all_ep_r1 = [[] for _ in range(TRAIN_NUM)]
    k = 0
    consecutive_nonfinite_updates = 0
    nonfinite_events = []
    stop_due_nonfinite = False
    stop_reason = None
    stop_episode = None
    stop_timestep = None
    diagnostic_path = os.path.splitext(result_path)[0] + "_diagnostics.json"
    nonfinite_stop_model_path = os.path.join(model_dir, "nonfinite_stop_model")

    def _extract_reward_components(reward_data):
        total_reward = 0.0
        leader_reward = 0.0
        first_follower_reward = 0.0

        if isinstance(reward_data, dict):
            leader_val = reward_data.get("leader", 0.0)
            if isinstance(leader_val, (int, float, np.number)):
                leader_reward = float(leader_val)
                total_reward += leader_reward

            follower_vals = reward_data.get("followers", [])
            if isinstance(follower_vals, list):
                for idx, item in enumerate(follower_vals):
                    if isinstance(item, (int, float, np.number)):
                        item_val = float(item)
                        total_reward += item_val
                        if idx == 0:
                            first_follower_reward = item_val
        else:
            reward_arr = np.asarray(reward_data, dtype=np.float32).reshape(-1)
            if reward_arr.size > 0:
                total_reward = float(reward_arr.sum())
                leader_reward = float(reward_arr[0])
                if reward_arr.size > 1:
                    first_follower_reward = float(reward_arr[1])

        return total_reward, leader_reward, first_follower_reward

    env = None
    masac_controller = None
    try:
        env = RlGame(
            leader_count=initial_n_agent,
            follower_count=initial_m_enemy,
            obstacle_num=args.obstacle_count,
            render=RENDER
        ).unwrapped
        env.set_time_step(1.0)
        print("训练模式：时间步长dt设置为1.0")
        masac_controller = MASACController(
            n_agents=n_agents,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            memory_capacity=MemoryCapacity,
            max_replay_ratio=20,
            share_follower_policy=share_follower_policy,
            use_attention=use_attention,
            use_gat=use_gat
        )

        training_start_size = int(MemoryCapacity * training_start_ratio)
        print(f"经验回放预热阈值: {training_start_size} ({training_start_ratio:.2f} x {MemoryCapacity})")

        for episode in range(train_episodes):
            observation = env.reset()
            total_reward = 0.0
            reward_total_leader = 0.0
            reward_total_follower1 = 0.0
            done = False
            win = False
            step_count = 0
            team_formation_time = 0

            while not done and step_count < EP_LEN:
                should_add_noise = episode < exploration_episodes
                action = masac_controller.select_actions(
                    observation,
                    add_noise=should_add_noise,
                    noise_scale=0.1,
                    evaluate=False
                )

                observation_, reward, done, win, team_counter, dis = env.step(action)

                masac_controller.store_transition(
                    observation,
                    action,
                    reward,
                    observation_,
                    done,
                    current_stage_tag="no_curriculum"
                )

                step_total_reward, leader_r, follower1_r = _extract_reward_components(reward)
                total_reward += step_total_reward
                reward_total_leader += leader_r
                reward_total_follower1 += follower1_r

                if team_counter > 0:
                    team_formation_time += 1

                observation = observation_
                step_count += 1

                if RENDER:
                    env.render()

                if len(masac_controller.memory.buffer) > training_start_size:
                    try:
                        train_status = masac_controller.train(
                            batch_size=BATCH,
                            current_stage_tag="no_curriculum",
                            current_stage_number=0
                        )

                        if isinstance(train_status, dict):
                            status_reason = str(train_status.get("reason", ""))
                            status_updated = bool(train_status.get("updated", False))

                            if status_updated:
                                consecutive_nonfinite_updates = 0
                            elif status_reason.startswith("non_finite"):
                                consecutive_nonfinite_updates += 1
                                event = {
                                    "episode": int(episode),
                                    "timestep": int(step_count),
                                    "reason": status_reason,
                                    "consecutive_nonfinite_updates": int(consecutive_nonfinite_updates),
                                    "train_step": int(train_status.get("train_step", masac_controller.train_step))
                                }
                                nonfinite_events.append(event)
                                if len(nonfinite_events) > 200:
                                    nonfinite_events = nonfinite_events[-200:]

                                log(
                                    f"检测到非有限更新跳过: reason={status_reason}, 连续次数={consecutive_nonfinite_updates}/{max_consecutive_nonfinite_updates}",
                                    LOG_WARNING
                                )

                                if consecutive_nonfinite_updates >= max_consecutive_nonfinite_updates:
                                    stop_due_nonfinite = True
                                    stop_reason = f"连续非有限更新达到阈值({max_consecutive_nonfinite_updates})"
                                    stop_episode = int(episode)
                                    stop_timestep = int(step_count)
                                    log(
                                        f"触发自动停训: {stop_reason}, episode={episode+1}, timestep={step_count}",
                                        LOG_ERROR
                                    )
                                    break
                    except Exception as e:
                        print(f"训练过程中出现错误: {e}")
                        import traceback
                        traceback.print_exc()

            if stop_due_nonfinite:
                break

            alpha_stats = []
            if hasattr(masac_controller, 'entroy_leader') and n_agents >= 1:
                alpha_stats.append(masac_controller.entroy_leader.get_alpha_stats())
            if hasattr(masac_controller, 'entroy_follower') and n_agents > 1:
                alpha_stats.append(masac_controller.entroy_follower.get_alpha_stats())
            avg_alpha = np.mean([stat["current"] for stat in alpha_stats]) if alpha_stats else 0.0

            alpha_history.append(float(avg_alpha))
            reward_history.append(float(total_reward))
            success_rate_history.append(float(win))
            all_ep_r[k].append(float(total_reward))
            all_ep_r0[k].append(float(reward_total_leader))
            all_ep_r1[k].append(float(reward_total_follower1))

            if episode % 10 == 0 or win:
                status = "成功" if win else "进行中"
                formation_rate = team_formation_time / max(step_count, 1)
                log(
                    f"无课程学习训练 - 回合: {episode+1}/{train_episodes}, 状态: {status}, "
                    f"步数: {step_count}, 总奖励: {total_reward:.2f}, 编队率: {formation_rate:.2f}, Alpha: {avg_alpha:.4f}",
                    LOG_INFO
                )

            if episode % save_interval == 0 and episode > 0:
                save_path = os.path.join(model_dir, f"model_ep{episode}")
                try:
                    masac_controller.save_models(save_path)
                    print(f"阶段模型已保存: {save_path}")
                except Exception as e:
                    print(f"保存阶段模型失败: {e}")
                    import traceback
                    traceback.print_exc()

        if stop_due_nonfinite:
            print(
                f"触发自动停训：{stop_reason}，停在 episode={((stop_episode + 1) if stop_episode is not None else 'N/A')} "
                f"timestep={stop_timestep}"
            )

        if stop_due_nonfinite and masac_controller is not None:
            try:
                masac_controller.save_models(nonfinite_stop_model_path)
                print(f"非有限停训快照已保存: {nonfinite_stop_model_path}")
            except Exception as e:
                print(f"保存非有限停训快照失败: {e}")
                import traceback
                traceback.print_exc()

        diagnostics = {
            "mode": mode_name,
            "stop_due_nonfinite": stop_due_nonfinite,
            "stop_reason": stop_reason,
            "max_consecutive_nonfinite_updates": max_consecutive_nonfinite_updates,
            "consecutive_nonfinite_updates": consecutive_nonfinite_updates,
            "stop_episode": (stop_episode + 1) if stop_episode is not None else None,
            "stop_timestep": stop_timestep,
            "nonfinite_event_count": len(nonfinite_events),
            "nonfinite_events_tail": nonfinite_events[-50:],
            "last_train_status": masac_controller.last_train_status if masac_controller is not None else None,
            "memory_size": len(masac_controller.memory.buffer) if masac_controller is not None else 0,
            "train_step": masac_controller.train_step if masac_controller is not None else 0,
            "seed": seed,
            "hero_count": initial_n_agent,
            "enemy_count": initial_m_enemy,
            "obstacle_count": args.obstacle_count,
            "timestamp": no_curriculum_paths["run_stamp"]
        }

        try:
            with open(diagnostic_path, 'w', encoding='utf-8') as f:
                json.dump(convert_to_json_compatible(diagnostics), f, ensure_ascii=False, indent=2)
            print(f"训练诊断信息已保存: {diagnostic_path}")
        except Exception as e:
            print(f"保存训练诊断信息失败: {e}")
            import traceback
            traceback.print_exc()

        try:
            masac_controller.save_models(final_model_path)
            print(f"最终模型已保存: {final_model_path}")
        except Exception as e:
            print(f"保存最终模型失败: {e}")
            import traceback
            traceback.print_exc()

    finally:
        if env is not None:
            env.close()

    all_ep_r_mean = np.mean((np.array(all_ep_r)), axis=0)
    all_ep_r_std = np.std((np.array(all_ep_r)), axis=0)
    all_ep_L_mean = np.mean((np.array(all_ep_r0)), axis=0)
    all_ep_L_std = np.std((np.array(all_ep_r0)), axis=0)
    all_ep_F_mean = np.mean((np.array(all_ep_r1)), axis=0)
    all_ep_F_std = np.std((np.array(all_ep_r1)), axis=0)

    training_results = {
        "mode": mode_name,
        "all_ep_r_mean": all_ep_r_mean,
        "all_ep_r_std": all_ep_r_std,
        "all_ep_L_mean": all_ep_L_mean,
        "all_ep_L_std": all_ep_L_std,
        "all_ep_F_mean": all_ep_F_mean,
        "all_ep_F_std": all_ep_F_std,
        "alpha_history": alpha_history,
        "reward_history": reward_history,
        "success_rate_history": success_rate_history,
        "seed": seed,
        "run_id": run_id,
        "hero_count": initial_n_agent,
        "enemy_count": initial_m_enemy,
        "obstacle_count": args.obstacle_count,
        "train_episodes": train_episodes,
        "save_interval": save_interval,
        "exploration_episodes": exploration_episodes,
        "share_follower_policy": share_follower_policy,
        "use_attention": use_attention,
        "training_start_ratio": training_start_ratio,
        "max_consecutive_nonfinite_updates": max_consecutive_nonfinite_updates,
        "stop_due_nonfinite": stop_due_nonfinite,
        "stop_reason": stop_reason,
        "stop_episode": (stop_episode + 1) if stop_episode is not None else None,
        "stop_timestep": stop_timestep,
        "nonfinite_event_count": len(nonfinite_events),
        "diagnostic_path": diagnostic_path,
        "nonfinite_stop_model_path": nonfinite_stop_model_path if stop_due_nonfinite else None,
        "final_model_path": final_model_path,
        "timestamp": no_curriculum_paths["run_stamp"]
    }

    with open(result_path, 'wb') as f:
        pkl.dump(training_results, f, pkl.HIGHEST_PROTOCOL)
    print(f"训练结果已保存: {result_path}")

    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        x = np.arange(len(all_ep_r_mean))
        ax1.plot(x, all_ep_r_mean, label='Average Reward', color='tab:blue')
        ax1.fill_between(
            x,
            all_ep_r_mean - all_ep_r_std,
            all_ep_r_mean + all_ep_r_std,
            alpha=0.15,
            color='tab:blue'
        )
        ax1.set_title('MASAC without Curriculum - Rewards' + (' (Attention)' if use_attention else ' (No Attention)'))
        ax1.set_ylabel('Episode Reward')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()

        if alpha_history:
            ax2.plot(np.arange(len(alpha_history)), alpha_history, label='Alpha', color='tab:green')
        ax2.set_title('Entropy Temperature Alpha')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Alpha')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"训练曲线已保存: {plot_path}")
    except Exception as e:
        print(f"保存训练曲线失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close('all')

    return {
        'final_success_rate': np.mean(success_rate_history[-100:]) if len(success_rate_history) >= 100 else np.mean(success_rate_history) if success_rate_history else 0,
        'final_avg_reward': np.mean(all_ep_r[0][-100:]) if len(all_ep_r[0]) >= 100 else np.mean(all_ep_r[0]) if all_ep_r[0] else 0,
        'seed': seed,
        'run_id': run_id,
        'total_episodes': len(all_ep_r[0]) if all_ep_r[0] else 0,
        'share_follower_policy': share_follower_policy,
        'use_attention': use_attention,
        'stop_due_nonfinite': stop_due_nonfinite,
        'stop_reason': stop_reason,
        'stop_episode': (stop_episode + 1) if stop_episode is not None else None,
        'stop_timestep': stop_timestep,
        'nonfinite_event_count': len(nonfinite_events),
        'diagnostic_path': diagnostic_path,
        'nonfinite_stop_model_path': nonfinite_stop_model_path if stop_due_nonfinite else None,
        'final_model_path': final_model_path,
        'result_path': result_path,
        'plot_path': plot_path,
        'timestamp': no_curriculum_paths["run_stamp"]
    }


# ==============================================================================
# 模块: 多随机种子训练流程
# ==============================================================================


def run_multi_seed_curriculum(args, initial_n_agent, initial_m_enemy):

    print("=== 多次课程学习训练模式 ===")

    if args.random_seed and args.seeds:
        print("检测到同时设置 --random_seed 和 --seeds，将优先使用 --seeds 指定的种子列表")
    
    # 解析种子列表
    if args.seeds:
        try:
            seeds = [int(s.strip()) for s in args.seeds.split(',')]
            print(f"使用自定义种子: {seeds}")
        except ValueError:
            print("种子格式错误，使用默认种子")
            seeds = [42, 123, 456][:args.num_runs]
    else:
        # 按种子模式自动生成种子
        if args.random_seed:
            sys_random = random.SystemRandom()
            seeds = [sys_random.randint(1, 10000) for _ in range(args.num_runs)]
            print(f"随机种子模式，自动生成种子: {seeds}")
        else:
            random.seed(args.seed)
            seeds = [random.randint(1, 10000) for _ in range(args.num_runs)]
            print(f"固定种子模式，基于 seed={args.seed} 生成种子: {seeds}")
    
    # 确保种子数量与运行次数匹配
    if len(seeds) != args.num_runs:
        print(f"警告: 种子数量({len(seeds)})与运行次数({args.num_runs})不匹配")
        if len(seeds) < args.num_runs:
            # 补充种子
            if args.random_seed:
                sys_random = random.SystemRandom()
                while len(seeds) < args.num_runs:
                    seeds.append(sys_random.randint(1, 10000))
            else:
                random.seed(seeds[-1] if seeds else args.seed)
                while len(seeds) < args.num_runs:
                    seeds.append(random.randint(1, 10000))
        else:
            # 截取种子
            seeds = seeds[:args.num_runs]
        print(f"调整后的种子: {seeds}")
    
    all_results = []
    
    print(f"开始进行 {args.num_runs} 次课程学习训练")
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"第 {i+1}/{args.num_runs} 次训练 - 种子: {seed}")
        print(f"{'='*60}")
        
        # 运行单次课程学习训练
        result = run_with_curriculum(args, initial_n_agent, initial_m_enemy, seed=seed, run_id=i+1)
        all_results.append(result)
        
        print(f"第 {i+1} 次课程学习训练完成")
        if result:
            final_reward = result.get('final_avg_reward', 'N/A')
            final_success = result.get('final_success_rate', 'N/A')
            curriculum_steps = result.get('curriculum_steps_completed', 'N/A')
            total_episodes = result.get('total_episodes', 'N/A')
            print(f"最终平均奖励: {final_reward}")
            print(f"最终成功率: {final_success}")
            print(f"完成课程步骤: {curriculum_steps}")
            print(f"总训练回合: {total_episodes}")
    
    # 汇总所有运行的结果
    print(f"\n{'='*60}")
    print("多次课程学习训练汇总")
    print(f"{'='*60}")
    
    # 计算统计信息
    final_rewards = [r.get('final_avg_reward', 0) for r in all_results if r]
    final_success_rates = [r.get('final_success_rate', 0) for r in all_results if r]
    curriculum_steps_list = [r.get('curriculum_steps_completed', 0) for r in all_results if r]
    total_episodes_list = [r.get('total_episodes', 0) for r in all_results if r]
    
    if final_rewards:
        reward_mean = np.mean(final_rewards)
        reward_std = np.std(final_rewards)
        reward_95_ci = 1.96 * reward_std / np.sqrt(len(final_rewards))
        
        print(f"最终平均奖励: {reward_mean:.3f} ± {reward_std:.3f}")
        print(f"95% 置信区间: [{reward_mean - reward_95_ci:.3f}, {reward_mean + reward_95_ci:.3f}]")
    
    if final_success_rates:
        success_mean = np.mean(final_success_rates)
        success_std = np.std(final_success_rates)
        success_95_ci = 1.96 * success_std / np.sqrt(len(final_success_rates))
        
        print(f"最终成功率: {success_mean:.3f} ± {success_std:.3f}")
        print(f"95% 置信区间: [{success_mean - success_95_ci:.3f}, {success_mean + success_95_ci:.3f}]")
    
    if curriculum_steps_list:
        steps_mean = np.mean(curriculum_steps_list)
        steps_std = np.std(curriculum_steps_list)
        print(f"平均完成课程步骤: {steps_mean:.1f} ± {steps_std:.1f}")
    
    if total_episodes_list:
        episodes_mean = np.mean(total_episodes_list)
        episodes_std = np.std(total_episodes_list)
        print(f"平均总训练回合: {episodes_mean:.1f} ± {episodes_std:.1f}")
    
    # 保存汇总结果
    timestamp = get_timestamp()
    use_attention = not bool(getattr(args, "disable_attention", False))
    summary_prefix = "MASAC_curriculum_multi_run_summary" if use_attention else "MASAC_curriculum_no_attention_multi_run_summary"
    summary_filename = f"{summary_prefix}_{timestamp}.pkl"
    summary_path = os.path.join(RESULTS_DIR, summary_filename)
    
    summary_results = {
        'num_runs': args.num_runs,
        'seeds': seeds,
        'individual_results': all_results,
        'summary_stats': {
            'final_rewards': {
                'mean': reward_mean if final_rewards else 0,
                'std': reward_std if final_rewards else 0,
                'ci_95': reward_95_ci if final_rewards else 0,
                'values': final_rewards
            },
            'final_success_rates': {
                'mean': success_mean if final_success_rates else 0,
                'std': success_std if final_success_rates else 0,
                'ci_95': success_95_ci if final_success_rates else 0,
                'values': final_success_rates
            },
            'curriculum_steps': {
                'mean': steps_mean if curriculum_steps_list else 0,
                'std': steps_std if curriculum_steps_list else 0,
                'values': curriculum_steps_list
            },
            'total_episodes': {
                'mean': episodes_mean if total_episodes_list else 0,
                'std': episodes_std if total_episodes_list else 0,
                'values': total_episodes_list
            }
        },
        'timestamp': timestamp,
        'config': {
            'initial_n_agent': initial_n_agent,
            'initial_m_enemy': initial_m_enemy,
            'seeds': seeds,
            'num_runs': args.num_runs,
            'use_curriculum': True,
            'use_attention': use_attention
        }
    }
    
    with open(summary_path, 'wb') as f:
        pkl.dump(summary_results, f)
    
    print(f"\n多次课程学习训练汇总结果已保存: {summary_path}")
    print("多次课程学习训练完成！")
    
    return summary_results


# ==============================================================================
# 模块: 课程学习分析
# ==============================================================================


def analyze_curriculum_learning(manager: CurriculumManager):
    """分析课程学习过程
    
    Args:
        manager: 课程管理器实例
    """
    tasks = manager.get_all_tasks()
    print(f"\n课程学习统计:")
    print(f"- 总任务数: {len(tasks)}")
    print(f"- 总训练回合数: {manager.total_episodes}")  # 添加总回合数信息
    
    # 计算难度相关统计
    difficulties = [task.difficulty for task in tasks if task.difficulty is not None]
    if difficulties:
        print(f"- 难度范围: {min(difficulties):.2f} - {max(difficulties):.2f}")
        print(f"- 平均难度: {np.mean(difficulties):.2f}")
    
    # 分析任务变化维度
    hero_counts = [task.env_params.get("hero_count", 1) for task in tasks]
    enemy_counts = [task.env_params.get("enemy_count", 1) for task in tasks]
    obstacle_counts = [task.env_params.get("obstacle_count", 0) for task in tasks]
    uav_speeds = [task.env_params.get("uav_speed", 10.0) for task in tasks]
    
    print(f"- 友方无人机数量变化: {min(hero_counts)} - {max(hero_counts)}")
    print(f"- 敌方无人机数量变化: {min(enemy_counts)} - {max(enemy_counts)}")
    print(f"- 障碍物数量变化: {min(obstacle_counts)} - {max(obstacle_counts)}")
    print(f"- 无人机速度变化: {min(uav_speeds):.1f} - {max(uav_speeds):.1f}")
    
    # 任务解决情况
    solved_tasks = [task for task in tasks if task.is_solved()]
    print(f"- 已解决任务数: {len(solved_tasks)}/{len(tasks)} ({len(solved_tasks)/len(tasks)*100:.1f}%)")
    
    # 任务难度分布
    difficulty_ranges = {
        "低难度(0.0-0.3)": len([t for t in tasks if t.difficulty is not None and t.difficulty <= 0.3]),
        "中难度(0.3-0.7)": len([t for t in tasks if t.difficulty is not None and 0.3 < t.difficulty <= 0.7]),
        "高难度(0.7-1.0)": len([t for t in tasks if t.difficulty is not None and t.difficulty > 0.7])
    }
    print("- 任务难度分布:")
    for range_name, count in difficulty_ranges.items():
        print(f"  - {range_name}: {count} 个任务 ({count/len(tasks)*100:.1f}%)")
        
    # 任务解决情况与难度的关系
    if solved_tasks:
        solved_difficulties = [task.difficulty for task in solved_tasks if task.difficulty is not None]
        if solved_difficulties:
            print(f"- 已解决任务的平均难度: {np.mean(solved_difficulties):.2f}")
            
    # 任务解决所需的平均训练轮数
    episodes_to_solve = []
    for task in solved_tasks:
        if task.performance_history:
            episodes_to_solve.append(len(task.performance_history))
    
    if episodes_to_solve:
        print(f"- 任务解决平均训练轮数: {np.mean(episodes_to_solve):.1f}")
        print(f"- 任务解决最少训练轮数: {min(episodes_to_solve)}")
        print(f"- 任务解决最多训练轮数: {max(episodes_to_solve)}")
    
    print("\n课程学习分析完成")


# ==============================================================================
# 模块: 单配置蒙特卡洛测试
# ==============================================================================

def run_monte_carlo_test(model_path, test_episodes=None, test_options=None, collect_formation_data=True, share_follower_policy=False, use_attention=True, use_gat=False):
    """运行蒙特卡洛测试以评估模型性能
    
    Args:
        model_path: 要测试的模型路径
        test_episodes: 测试回合数，如果为None则使用全局变量TEST_EPIOSDE
        test_options: 测试选项字典，包含如障碍物数量、无人机速度等配置；
            可选 fixed_goal_pos=(x, y) 用于固定目标点，否则每回合随机目标
        collect_formation_data: 是否收集详细的编队数据，默认为True
        share_follower_policy: 是否启用共享随从策略（所有随从共用同一个Actor）
        use_attention: 是否按注意力结构创建策略网络
        
    Returns:
        测试结果统计字典
    """
    import numpy as np  # 移到函数开始处避免UnboundLocalError
    global RENDER, action_number
    
    if test_episodes is None:
        test_episodes = TEST_EPIOSDE
    
    if test_options is None:
        test_options = {}
        
    # 从测试选项中提取环境参数
    obstacle_count = test_options.get('obstacle_count', 1)
    uav_speed = test_options.get('uav_speed', None)
    hero_count = test_options.get('hero_count', 1)
    enemy_count = test_options.get('enemy_count', 3) 
        
    print(f"开始蒙特卡洛测试，测试回合数: {test_episodes}")
    print(f"加载模型: {model_path}")
    print(f"测试配置: 友方={hero_count}, 敌方={enemy_count}, 障碍物={obstacle_count}")
    print(f"共享随从策略: {bool(share_follower_policy)}")
    print(f"网络结构: {'注意力' if use_attention else '无注意力'}")
    if uav_speed:
        print(f"无人机速度: {uav_speed}")
    if collect_formation_data:
        print("启用详细编队数据收集")
    
    # 默认每回合随机目标位置；如需固定目标，可通过 test_options 传入 fixed_goal_pos=(x, y)
    fixed_goal_pos = test_options.get('fixed_goal_pos', None)
    predefined_positions = None
    if fixed_goal_pos is not None:
        if isinstance(fixed_goal_pos, (list, tuple)) and len(fixed_goal_pos) == 2:
            goal_x, goal_y = fixed_goal_pos
            predefined_positions = {"goals": [(goal_x, goal_y)]}
            print(f"使用固定目标位置: ({goal_x}, {goal_y})")
        else:
            print("警告: fixed_goal_pos 格式无效，应为 (x, y)。将使用随机目标位置。")
    else:
        print("目标位置模式: 每回合随机")
    
    # 创建环境；默认不传预定义坐标以启用 reset 随机位置
    env = RlGame(leader_count=hero_count, follower_count=enemy_count, obstacle_num=obstacle_count, render=RENDER, 
                predefined_positions=predefined_positions).unwrapped
    
    env.set_time_step(0.1)
    print(f"测试模式：时间步长dt设置为0.1")
    

    if hasattr(env, 'entity_manager') and hasattr(env.entity_manager, 'images_loaded'):
        env.entity_manager.images_loaded = False
        print("已重置EntityManager的图像加载状态，确保测试时图像正确加载")
    

    n_agents = hero_count + enemy_count
    state_dim = state_number
    action_dim = action_number
    masac_controller = MASACController(
        n_agents,
        state_dim,
        action_dim,
        device=device,
        share_follower_policy=share_follower_policy,
        use_attention=use_attention,
        use_gat=use_gat
    )
    loaded = masac_controller.load_models(model_path, strict=False)
    if not loaded:
        raise RuntimeError(f"模型加载失败，测试中止: {model_path}")
    masac_controller.set_eval_mode()
    
    # 初始化统计数据
    win_count = 0
    rewards = []
    steps = []
    formation_rates = []
    distances = []  # 智能体最终与目标的距离
    trajectory_lengths = []  # 飞行轨迹长度列表
    energy_consumptions = []  # 能量消耗列表
    
    # 初始化编队数据收集结构
    formation_data = {
        'test_info': {
            'test_id': get_timestamp(),
            'timestamp': time.time(),
            'total_episodes': test_episodes,
            'agents_config': {
                'leader_count': hero_count,
                'follower_count': enemy_count,
                'obstacle_count': obstacle_count,
                'uav_speed': uav_speed,
                'share_follower_policy': bool(share_follower_policy),
                'use_attention': bool(use_attention)
            }
        },
        'episodes': [],
        'summary_stats': {}
    } if collect_formation_data else None
    
    # 运行测试
    for episode in range(test_episodes):
        observation = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        team_formation_time = 0
        last_distance = None
        
        # 初始化当前回合的数据收集
        episode_formation_data = {
            'episode_id': episode,
            'timesteps': [],
            'episode_summary': {}
        } if collect_formation_data else None
        
        episode_trajectory_length = 0.0  # 当前回合轨迹长度
        episode_energy_consumption = 0.0  # 当前回合能量消耗
        
        while not done and step_count < EP_LEN:
            # 收集当前时间步的编队状态数据
            if collect_formation_data:
                timestep_state = env.get_formation_state()
                episode_formation_data['timesteps'].append(timestep_state)
            
            # 选择动作（无噪声且使用确定性策略）
            action = masac_controller.select_actions(observation, add_noise=False, evaluate=True)
            
            # 执行动作
            observation_, reward, done, win, team_counter, dis = env.step(action)
            
            # 获取速度和控制输入数据
            if hasattr(env, 'entity_manager') and env.entity_manager.leaders:
                leader = env.entity_manager.leaders[0]
                # 计算轨迹长度增量
                speed = getattr(leader, 'speed', 0.0)
                episode_trajectory_length += float(speed)
                
                # 计算能量消耗增量（基于动作）
                if isinstance(action, dict) and "leader" in action:
                    leader_action = action["leader"]
                else:
                    leader_action = action[0] if len(action) > 0 else [0, 0]
                
                # 假设动作为[加速度, 角速度]
                u = abs(float(leader_action[0])) if len(leader_action) > 0 else 0.0
                omega = abs(float(leader_action[1])) if len(leader_action) > 1 else 0.0
                episode_energy_consumption += (u + omega)
            
            # 记录最后一步的距离
            last_distance = dis
            
            # 更新状态和统计
           # 更新状态和统计
            observation = observation_
            
# 累加当前时间步的总奖励
            current_step_reward = 0.0
            if isinstance(reward, dict):
                leader_r = reward.get("leader", 0.0)      # 安全获取leader奖励，默认为0.0
                followers_r = reward.get("followers", []) # 安全获取followers奖励列表，默认为空列表

    # 累加 Leader 奖励 (确保是数字)
                if isinstance(leader_r, (int, float, np.number)):
                    current_step_reward += float(leader_r)

    # 累加 Followers 奖励 (确保列表中的元素是数字)
                if isinstance(followers_r, list):
                    for r in followers_r:
                        if isinstance(r, (int, float, np.number)):
                            current_step_reward += float(r)
            elif isinstance(reward, (np.ndarray, list)): # 兼容旧格式
                try:
                    current_step_reward = np.mean(reward)
                except:
        # 忽略无法处理的旧格式
                    pass
            elif isinstance(reward, (int, float, np.number)): # 单个数值奖励
                current_step_reward = float(reward)

            total_reward += current_step_reward
            step_count += 1
            
            # 计算编队时间
            if team_counter > 0:
                team_formation_time += 1
            
            # 渲染环境
            if RENDER:
                env.render()
        
        # 完成当前回合的数据收集
        if collect_formation_data and episode_formation_data:
            # 计算回合级别的编队质量指标
            if episode_formation_data['timesteps']:
                timesteps_data = episode_formation_data['timesteps']
                
                # 计算平均编队距离误差
                formation_errors = []
                for ts in timesteps_data:
                    for follower in ts['followers']:
                        formation_errors.append(follower['formation_distance_error'])
                
                avg_formation_error = np.mean(formation_errors) if formation_errors else 0.0
                
                episode_formation_data['episode_summary'] = {
                    'total_steps': len(timesteps_data),
                    'formation_rate': team_formation_time / step_count if step_count > 0 else 0,
                    'avg_formation_error': float(avg_formation_error),
                    'total_reward': float(total_reward),
                    'success': bool(win)
                }
            
            formation_data['episodes'].append(episode_formation_data)
        
        # 收集统计信息
        win_count += int(win)
        rewards.append(total_reward)
        steps.append(step_count)
        
        # 计算这一回合的编队保持率
        formation_rate = team_formation_time / step_count if step_count > 0 else 0
        formation_rates.append(formation_rate)
        
        # 记录最终距离
        if last_distance is not None:
            if isinstance(last_distance, dict):
                # 如果是字典，取所有值的最小值
                if last_distance:
                    distances.append(min(last_distance.values()))
                else:
                    distances.append(0)  # 如果字典为空，使用0
            elif isinstance(last_distance, (list, tuple, np.ndarray)):
                # 如果是列表、元组或数组，取最小值
                distances.append(min(last_distance))
            else:
                # 如果是标量，直接添加
                distances.append(last_distance)
        
        # 存储轨迹长度和能量消耗
        trajectory_lengths.append(episode_trajectory_length)
        energy_consumptions.append(episode_energy_consumption)
        
        # 输出回合信息
        status = "成功" if win else "失败"
        print(f"测试回合 {episode+1}/{test_episodes}, 状态: {status}, 奖励: {total_reward:.1f}, "
              f"步数: {step_count}, 编队率: {formation_rate:.2f}")
    
    # 计算编队数据的汇总统计
    if collect_formation_data and formation_data:
        # 计算所有回合的平均编队质量指标
        all_heading_angles = {'leaders': [], 'followers': []}
        all_speeds = {'leaders': [], 'followers': []}
        all_distances = []
        
        for episode_data in formation_data['episodes']:
            for timestep in episode_data['timesteps']:
                # 收集航向角数据
                for leader in timestep['leaders']:
                    all_heading_angles['leaders'].append(leader['heading_angle'])
                for follower in timestep['followers']:
                    all_heading_angles['followers'].append(follower['heading_angle'])
                
                # 收集速度数据
                for leader in timestep['leaders']:
                    all_speeds['leaders'].append(leader['speed'])
                for follower in timestep['followers']:
                    all_speeds['followers'].append(follower['speed'])
                
                # 收集距离数据
                for follower in timestep['followers']:
                    all_distances.append(follower['leader_distance'])
        
        formation_data['summary_stats'] = {
            'mean_formation_quality': float(np.mean(formation_rates)) if formation_rates else 0.0,
            'avg_heading_angles': {
                'leaders': float(np.mean(all_heading_angles['leaders'])) if all_heading_angles['leaders'] else 0.0,
                'followers': float(np.mean(all_heading_angles['followers'])) if all_heading_angles['followers'] else 0.0
            },
            'avg_speeds': {
                'leaders': float(np.mean(all_speeds['leaders'])) if all_speeds['leaders'] else 0.0,
                'followers': float(np.mean(all_speeds['followers'])) if all_speeds['followers'] else 0.0
            },
            'avg_leader_follower_distance': float(np.mean(all_distances)) if all_distances else 0.0
        }
    
    # 计算统计结果
    success_rate = win_count / test_episodes
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_steps = np.mean(steps)
    std_steps = np.std(steps)
    avg_formation_rate = np.mean(formation_rates)
    std_formation_rate = np.std(formation_rates)
    
    # 计算最终距离的平均值和标准差
    avg_distance = np.mean(distances) if distances else 0
    std_distance = np.std(distances) if distances else 0
    
    # 计算新增指标
    avg_trajectory_length = np.mean(trajectory_lengths) if trajectory_lengths else 0
    std_trajectory_length = np.std(trajectory_lengths) if trajectory_lengths else 0
    avg_energy_consumption = np.mean(energy_consumptions) if energy_consumptions else 0
    std_energy_consumption = np.std(energy_consumptions) if energy_consumptions else 0
    
    # 计算成功率加权探索时间(SET)
    set_score = success_rate * avg_steps
    
    # 输出总体统计信息
    print("\n蒙特卡洛测试结果统计 (平均值±标准差):")
    print(f"测试回合数: {test_episodes}")
    print(f"1. 任务完成率(MCR): {success_rate:.2f}")
    print(f"2. 编队保持率(FKR): {avg_formation_rate:.2f}±{std_formation_rate:.2f}")
    print(f"3. 成功率加权探索时间(SET): {set_score:.2f} (SR: {success_rate:.2f} × 平均时间: {avg_steps:.2f})")
    print(f"4. 飞行轨迹(J_S): {avg_trajectory_length:.2f}±{std_trajectory_length:.2f}")
    print(f"5. 能量消耗(J_C): {avg_energy_consumption:.2f}±{std_energy_consumption:.2f}")
    print(f"平均奖励: {avg_reward:.2f}±{std_reward:.2f}")
    print(f"平均最终距离: {avg_distance:.2f}±{std_distance:.2f}")
    
    # 构建结果字典
    results = {
        "test_episodes": test_episodes,
        "success_rate": success_rate,
        "rewards": {
            "mean": float(avg_reward),
            "std": float(std_reward),
            "values": [float(r) for r in rewards]
        },
        "steps": {
            "mean": float(avg_steps),
            "std": float(std_steps),
            "values": [float(s) for s in steps]
        },
        "set_score": {
            "value": float(set_score),
            "success_rate": float(success_rate),
            "avg_exploration_time": float(avg_steps)
        },
        "formation_rates": {
            "mean": float(avg_formation_rate),
            "std": float(std_formation_rate),
            "values": [float(f) for f in formation_rates]
        },
        "distances": {
            "mean": float(avg_distance),
            "std": float(std_distance),
            "values": [float(d) for d in distances]
        },
        "trajectory_lengths": {
            "mean": float(avg_trajectory_length),
            "std": float(std_trajectory_length),
            "values": [float(t) for t in trajectory_lengths]
        },
        "energy_consumptions": {
            "mean": float(avg_energy_consumption),
            "std": float(std_energy_consumption),
            "values": [float(e) for e in energy_consumptions]
        },
        "test_config": {
            "hero_count": hero_count,
            "enemy_count": enemy_count,
            "obstacle_count": obstacle_count,
            "uav_speed": uav_speed,
            "use_attention": bool(use_attention)
        },
        "timestamp": time.time()
    }
    
    # 如果收集了编队数据，将其添加到结果中
    if collect_formation_data and formation_data:
        results["formation_data"] = formation_data
    
    # 创建唯一的测试ID和保存目录
    timestamp = get_timestamp()
    config_str = f"_h{hero_count}_e{enemy_count}_o{obstacle_count}"
    if uav_speed:
        config_str += f"_s{int(uav_speed)}"
    
    # 创建保存目录
    test_dir_name = f"test_{timestamp}{config_str}"
    test_dir = os.path.join(TEST_RESULTS_BASE, test_dir_name)
    ensure_dir_exists(test_dir)
    
    # 保存测试结果 (pickle格式)
    pickle_path = os.path.join(test_dir, "test_results.pkl")
    with open(pickle_path, 'wb') as f:
        pkl.dump(results, f, pkl.HIGHEST_PROTOCOL)
    print(f"测试结果已保存到: {pickle_path}")
    
    # 同时保存为JSON格式
    json_path = os.path.join(test_dir, "test_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_json_compatible(results), f, ensure_ascii=False, indent=4)
    print(f"测试结果(JSON格式)已保存到: {json_path}")
    
    # 保存测试信息
    date_str = datetime.datetime.fromtimestamp(results["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
    model_name = os.path.basename(model_path)
    
    test_info = {
        "timestamp": timestamp,
        "date": date_str,
        "model": model_name,
        "config": results["test_config"],
        "success_rate": success_rate,
        "avg_reward": float(avg_reward),
        "avg_steps": float(avg_steps),
        "formation_rate": float(avg_formation_rate)
    }
    
    info_path = os.path.join(test_dir, "test_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(test_info, f, ensure_ascii=False, indent=4)
    
    # 单独保存编队数据 (PKL格式)
    if collect_formation_data and formation_data:
        formation_pkl_path = os.path.join(test_dir, "formation_data.pkl")
        with open(formation_pkl_path, 'wb') as f:
            pkl.dump(formation_data, f, pkl.HIGHEST_PROTOCOL)
        print(f"编队数据已保存到: {formation_pkl_path}")
        # 生成热力图（基于 formation_data.pkl）
        try:
            heatmaps_module = __import__("visualization.plot_heatmaps", fromlist=["generate_heatmaps"])
            generate_heatmaps = getattr(heatmaps_module, "generate_heatmaps", None)
            if callable(generate_heatmaps):
                heatmap_paths = generate_heatmaps(formation_pkl_path, test_dir)
                if heatmap_paths:
                    print("已生成热力图:")
                    for k, v in heatmap_paths.items():
                        print(f" - {k}: {v}")
            else:
                print("未在 visualization.plot_heatmaps 中找到 generate_heatmaps，跳过热力图生成")
        except ModuleNotFoundError:
            print("未找到 visualization.plot_heatmaps，跳过热力图生成")
        except Exception as e:
            print(f"生成热力图时出错: {e}")
            
        # 生成综合分析曲线图
        try:
            formation_curves_module = __import__("visualization.plot_formation_curves", fromlist=["generate_formation_curves"])
            generate_formation_curves = getattr(formation_curves_module, "generate_formation_curves", None)
            if callable(generate_formation_curves):
                curves_path = generate_formation_curves(formation_pkl_path, test_dir, "AC-MASAC")
                if curves_path:
                    print(f"已生成综合分析曲线图: {curves_path}")
            else:
                print("未在 visualization.plot_formation_curves 中找到 generate_formation_curves，跳过曲线图生成")
        except ModuleNotFoundError:
            print("未找到 visualization.plot_formation_curves，跳过曲线图生成")
        except Exception as e:
            print(f"生成综合分析曲线图时出错: {e}")
        
        # 保存编队数据汇总 
        formation_summary_path = os.path.join(test_dir, "formation_summary.pkl")
        formation_summary = {
            'summary_stats': formation_data['summary_stats'],
            'test_info': formation_data['test_info'],
            'episode_summaries': [ep['episode_summary'] for ep in formation_data['episodes']]
        }
        with open(formation_summary_path, 'wb') as f:
            pkl.dump(formation_summary, f, pkl.HIGHEST_PROTOCOL)
        print(f"编队数据汇总已保存到: {formation_summary_path}")
    
    # 绘制测试奖励分布直方图
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # 奖励分布直方图
        plt.subplot(2, 2, 1)
        plt.hist(rewards, bins=min(20, test_episodes//5), alpha=0.7)
        plt.title('奖励分布')
        plt.xlabel('奖励')
        plt.ylabel('频次')
        plt.axvline(avg_reward, color='r', linestyle='dashed', linewidth=1, label=f'平均值: {avg_reward:.2f}')
        plt.legend()
        
        # 步数分布直方图
        plt.subplot(2, 2, 2)
        plt.hist(steps, bins=min(20, test_episodes//5), alpha=0.7)
        plt.title('步数分布')
        plt.xlabel('步数')
        plt.ylabel('频次')
        plt.axvline(avg_steps, color='r', linestyle='dashed', linewidth=1, label=f'平均值: {avg_steps:.2f}')
        plt.legend()
        
        # 编队率分布直方图
        plt.subplot(2, 2, 3)
        plt.hist(formation_rates, bins=min(20, test_episodes//5), alpha=0.7)
        plt.title('编队保持率分布')
        plt.xlabel('编队保持率')
        plt.ylabel('频次')
        plt.axvline(avg_formation_rate, color='r', linestyle='dashed', linewidth=1, label=f'平均值: {avg_formation_rate:.2f}')
        plt.legend()
        
        # 距离分布直方图
        if distances:
            plt.subplot(2, 2, 4)
            plt.hist(distances, bins=min(20, test_episodes//5), alpha=0.7)
            plt.title('最终距离分布')
            plt.xlabel('距离')
            plt.ylabel('频次')
            plt.axvline(avg_distance, color='r', linestyle='dashed', linewidth=1, label=f'平均值: {avg_distance:.2f}')
            plt.legend()
        
        plt.tight_layout()
        title = f"蒙特卡洛测试结果 (友方:{hero_count}, 敌方:{enemy_count}, 障碍:{obstacle_count})"
        if uav_speed:
            title += f", 速度:{uav_speed}"
        plt.suptitle(title)
        
        # 保存图片到测试结果目录
        save_img_path = os.path.join(test_dir, "histogram.png")
        plt.savefig(save_img_path)
        plt.close()
        print(f"测试结果直方图已保存到: {save_img_path}")
    except Exception as e:
        print(f"绘制直方图时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 更新测试结果索引
    create_test_results_index()
    
    return results


# ==============================================================================
# 模块: 测试结果分析
# ==============================================================================

def analyze_test_results(result_files=None, base_path=None):
    """分析和比较一组测试结果
    
    Args:
        result_files: 测试结果文件列表，不提供则自动搜索base_path下的所有测试结果
        base_path: 测试结果文件的基本路径，默认为TEST_RESULTS_BASE目录
        
    Returns:
        分析结果汇总
    """
    import os
    import matplotlib.pyplot as plt
    
    if base_path is None:
        base_path = TEST_RESULTS_BASE
        
    if not os.path.exists(base_path):
        print(f"测试结果目录不存在: {base_path}")
        ensure_dir_exists(base_path)
        print("没有找到任何测试结果文件")
        return None
    
    if result_files is None:
        # 自动搜索所有测试结果文件 - 优先查找JSON文件
        all_results = []
        
        # 遍历所有测试目录
        for root, dirs, files in os.walk(base_path):
            # 查找JSON结果文件
            json_results = [os.path.join(root, f) for f in files 
                           if f in ["test_results.json", "all_results.json"]]
            
            # 如果找到了JSON文件，优先使用
            if json_results:
                all_results.extend(json_results)
            else:
                # 否则查找pickle文件
                pkl_results = [os.path.join(root, f) for f in files 
                              if f in ["test_results.pkl", "all_results.pkl"]]
                all_results.extend(pkl_results)
        
        result_files = all_results
        
        if not result_files:
            print(f"在{base_path}目录及其子目录下未找到测试结果文件")
            return None
    
    print(f"找到{len(result_files)}个测试结果文件:")
    for i, file in enumerate(result_files):
        print(f"{i+1}. {file}")
    
    # 加载测试结果
    test_results = []
    for file in result_files:
        try:
            # 根据文件扩展名决定加载方式
            if file.endswith('.json'):
                with open(file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
            else:  # 假设是pickle文件
                with open(file, 'rb') as f:
                    result = pkl.load(f)
                    
            # 处理多难度测试结果
            if "level_1" in result:
                # 多难度测试结果 - 提取每个难度级别作为单独的结果
                for level_name, level_result in result.items():
                    if isinstance(level_result, dict) and "test_config" in level_result:
                        level_result['level_name'] = level_name
                        level_result['file_name'] = os.path.basename(file) + f":{level_name}"
                        test_results.append(level_result)
                        print(f"已加载测试结果: {level_result['file_name']}")
            else:
                # 单一难度测试结果
                result['file_name'] = os.path.basename(file)
                test_results.append(result)
                print(f"已加载测试结果: {result['file_name']}")
                
        except Exception as e:
            print(f"加载文件{file}时出错: {e}")
            import traceback
            traceback.print_exc()
    
    if not test_results:
        print("没有成功加载任何测试结果")
        return None
    
    # 分析和比较测试结果
    print("\n测试结果比较:")
    print("-" * 80)
    print(f"{'配置信息':^40} | {'成功率':^10} | {'平均奖励':^15} | {'平均步数':^15} | {'编队率':^15}")
    print("-" * 80)
    
    for result in test_results:
        config = result.get('test_config', {})
        hero_count = config.get('hero_count', 'N/A')
        enemy_count = config.get('enemy_count', 'N/A')
        obstacle_count = config.get('obstacle_count', 'N/A')
        uav_speed = config.get('uav_speed', 'N/A')
        
        # 添加难度级别标识（如果有）
        level_prefix = ""
        if 'level_name' in result:
            level_prefix = f"{result['level_name']}: "
        
        config_str = f"{level_prefix}友方:{hero_count}, 敌方:{enemy_count}, 障碍:{obstacle_count}"
        if uav_speed != 'N/A':
            config_str += f", 速度:{uav_speed}"
            
        success_rate = result.get('success_rate', 0)
        reward_mean = result.get('rewards', {}).get('mean', 0)
        reward_std = result.get('rewards', {}).get('std', 0)
        steps_mean = result.get('steps', {}).get('mean', 0)
        steps_std = result.get('steps', {}).get('std', 0)
        formation_mean = result.get('formation_rates', {}).get('mean', 0)
        formation_std = result.get('formation_rates', {}).get('std', 0)
        
        print(f"{config_str:40} | {success_rate:10.2f} | {reward_mean:6.2f}±{reward_std:6.2f} | "
              f"{steps_mean:6.2f}±{steps_std:6.2f} | {formation_mean:6.2f}±{formation_std:6.2f}")
    
    print("-" * 80)
    
    # 绘制对比图表
    try:
        plt.figure(figsize=(16, 10))
        
        # 提取数据
        labels = []
        success_rates = []
        reward_means = []
        reward_stds = []
        steps_means = []
        formation_means = []
        
        for result in test_results:
            config = result.get('test_config', {})
            hero_count = config.get('hero_count', 'N/A')
            enemy_count = config.get('enemy_count', 'N/A')
            obstacle_count = config.get('obstacle_count', 'N/A')
            
            # 添加难度级别标识（如果有）
            if 'level_name' in result:
                label = f"{result['level_name']}"
            else:
                label = f"H{hero_count}E{enemy_count}O{obstacle_count}"
                
            if config.get('uav_speed', 'N/A') != 'N/A':
                label += f"S{config['uav_speed']}"
                
            labels.append(label)
            success_rates.append(result.get('success_rate', 0))
            reward_means.append(result.get('rewards', {}).get('mean', 0))
            reward_stds.append(result.get('rewards', {}).get('std', 0))
            steps_means.append(result.get('steps', {}).get('mean', 0))
            formation_means.append(result.get('formation_rates', {}).get('mean', 0))
        
        # 绘制成功率对比
        plt.subplot(2, 2, 1)
        plt.bar(labels, success_rates, alpha=0.7)
        plt.title('成功率对比')
        plt.ylabel('成功率')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)
        
        # 绘制平均奖励对比
        plt.subplot(2, 2, 2)
        plt.bar(labels, reward_means, yerr=reward_stds, alpha=0.7, capsize=5)
        plt.title('平均奖励对比')
        plt.ylabel('平均奖励')
        plt.xticks(rotation=45)
        
        # 绘制平均步数对比
        plt.subplot(2, 2, 3)
        plt.bar(labels, steps_means, alpha=0.7)
        plt.title('平均步数对比')
        plt.ylabel('平均步数')
        plt.xticks(rotation=45)
        
        # 绘制编队率对比
        plt.subplot(2, 2, 4)
        plt.bar(labels, formation_means, alpha=0.7)
        plt.title('编队保持率对比')
        plt.ylabel('编队保持率')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)
        
        plt.tight_layout()
        plt.suptitle('测试结果比较', fontsize=16)
        
        # 保存比较图
        analysis_dir = os.path.join(TEST_RESULTS_BASE, "analysis")
        ensure_dir_exists(analysis_dir)
        comparison_path = os.path.join(analysis_dir, f"test_comparison_{get_timestamp()}.png")
        plt.savefig(comparison_path)
        plt.close()
        print(f"测试结果比较图已保存到: {comparison_path}")
    except Exception as e:
        print(f"绘制比较图时出错: {e}")
        import traceback
        traceback.print_exc()
    
    return test_results


# ==============================================================================
# 模块: 多难度蒙特卡洛测试
# ==============================================================================

def monte_carlo_test(actor_path, critic_path=None, test_nums=100, base_difficulty_levels=None, share_follower_policy=False, use_attention=True, use_gat=False):
    """执行蒙特卡洛测试
    
    Args:
        actor_path: Actor路径
        critic_path: Critic路径(可选)
        test_nums: 测试次数
        base_difficulty_levels: 基础难度级别列表
        use_attention: 是否按注意力结构创建测试控制器
    """
    # 如果没有提供难度级别，使用默认的1-5个障碍物配置
    if base_difficulty_levels is None:
        base_difficulty_levels = [
            {'obstacle_count': 2, 'uav_speed': 1.0},  # 默认从2个障碍物开始
            {'obstacle_count': 3, 'uav_speed': 1.0},
            {'obstacle_count': 4, 'uav_speed': 1.0},
            {'obstacle_count': 5, 'uav_speed': 1.0},
            {'obstacle_count': 6, 'uav_speed': 1.0}
        ]
    
    # 加载环境和模型
    # from curriculum.fixed_task_generator import FixedTaskGenerator  # 已在顶部导入
    from rl_env.path_env import RlGame
    
    # 创建多难度测试专用目录
    timestamp = get_timestamp()
    model_name = os.path.basename(actor_path)
    multi_test_dir = os.path.join(TEST_RESULTS_BASE, f"multi_diff_test_{timestamp}_{model_name}")
    ensure_dir_exists(multi_test_dir)
    
    # 保存测试配置信息
    test_config = {
        "timestamp": timestamp,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "test_episodes": test_nums,
        "difficulty_levels": base_difficulty_levels,
        "hero_count": N_Agent,
        "enemy_count": M_Enemy,
        "use_attention": bool(use_attention),
        "goal_position": (500, 200)  # 记录固定目标位置
    }
    
    config_path = os.path.join(multi_test_dir, "test_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_json_compatible(test_config), f, ensure_ascii=False, indent=4)
    
    # 初始化结果存储
    all_results = {}
    
    # 对每个难度级别进行测试
    for difficulty_idx, difficulty_config in enumerate(base_difficulty_levels):
        print(f"\n{'-'*50}")
        print(f"测试难度级别 {difficulty_idx+1}/{len(base_difficulty_levels)}")
        print(f"配置: {difficulty_config}")
        print(f"{'-'*50}")
        
        # 运行测试 - 使用评估模式加载模型（仅加载Actor）
        result = run_monte_carlo_test(
            model_path=actor_path,
            test_episodes=test_nums,
            test_options=difficulty_config,
            share_follower_policy=share_follower_policy,
            use_attention=use_attention,
            use_gat=use_gat
        )
        
        # 存储结果
        level_key = f"level_{difficulty_idx+1}"
        all_results[level_key] = result
        
        # 打印当前难度级别的主要指标
        print(f"\n难度级别 {difficulty_idx+1} 测试结果摘要:")
        print(f"成功率: {result['success_rate']:.2f}")
        print(f"平均奖励: {result['rewards']['mean']:.2f}±{result['rewards']['std']:.2f}")
        print(f"平均步数: {result['steps']['mean']:.2f}±{result['steps']['std']:.2f}")
        print(f"平均编队率: {result['formation_rates']['mean']:.2f}±{result['formation_rates']['std']:.2f}")
        print(f"平均路径效率: {1.0 / result['steps']['mean']:.4f}")
        print(f"平均碰撞率: {1.0 - result['success_rate']:.2f}")
    
    # 输出整体结果摘要
    print(f"\n{'='*60}")
    print(f"多难度蒙特卡洛测试完成")
    print(f"{'='*60}\n")
    
    print("各难度级别测试结果汇总:")
    print(f"{'难度级别':^12} | {'成功率':^8} | {'平均奖励':^15} | {'平均步数':^15} | {'编队率':^15}")
    print("-" * 75)
    
    for level_idx, (level_name, level_result) in enumerate(all_results.items()):
        print(f"{level_name:^12} | {level_result['success_rate']:8.2f} | "
              f"{level_result['rewards']['mean']:6.2f}±{level_result['rewards']['std']:6.2f} | "
              f"{level_result['steps']['mean']:6.2f}±{level_result['steps']['std']:6.2f} | "
              f"{level_result['formation_rates']['mean']:6.2f}±{level_result['formation_rates']['std']:6.2f}")
    
    print("-" * 75)
    
    # 绘制不同难度级别的对比图表
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 从结果中提取数据
        levels = list(all_results.keys())
        success_rates = [all_results[level]['success_rate'] for level in levels]
        rewards = [all_results[level]['rewards']['mean'] for level in levels]
        reward_stds = [all_results[level]['rewards']['std'] for level in levels]
        steps = [all_results[level]['steps']['mean'] for level in levels]
        step_stds = [all_results[level]['steps']['std'] for level in levels]
        formation_rates = [all_results[level]['formation_rates']['mean'] for level in levels]
        formation_stds = [all_results[level]['formation_rates']['std'] for level in levels]
        
        # 计算路径效率和碰撞率
        path_efficiencies = [1.0 / step if step > 0 else 0 for step in steps]
        collision_rates = [1.0 - sr for sr in success_rates]
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 成功率
        axes[0, 0].bar(levels, success_rates, color='green')
        axes[0, 0].set_title('成功率')
        axes[0, 0].set_ylim(0, 1.0)
        for i, v in enumerate(success_rates):
            axes[0, 0].text(i, v + 0.02, f'{v:.2f}', ha='center')
        
        # 平均奖励
        axes[0, 1].bar(levels, rewards, color='blue', yerr=reward_stds, alpha=0.7, capsize=5)
        axes[0, 1].set_title('平均奖励')
        for i, v in enumerate(rewards):
            axes[0, 1].text(i, v + 0.5 if v >= 0 else v - 1.5, f'{v:.2f}', ha='center')
        
        # 平均步数
        axes[0, 2].bar(levels, steps, color='orange', yerr=step_stds, capsize=5)
        axes[0, 2].set_title('平均步数')
        for i, v in enumerate(steps):
            axes[0, 2].text(i, v + 2, f'{v:.2f}', ha='center')
        
        # 编队率
        axes[1, 0].bar(levels, formation_rates, color='purple', yerr=formation_stds, capsize=5)
        axes[1, 0].set_title('编队保持率')
        axes[1, 0].set_ylim(0, 1.0)
        for i, v in enumerate(formation_rates):
            axes[1, 0].text(i, v + 0.02, f'{v:.2f}', ha='center')
        
        # 路径效率
        axes[1, 1].bar(levels, path_efficiencies, color='teal')
        axes[1, 1].set_title('路径效率')
        for i, v in enumerate(path_efficiencies):
            axes[1, 1].text(i, v + 0.002, f'{v:.4f}', ha='center')
        
        # 碰撞率
        axes[1, 2].bar(levels, collision_rates, color='red')
        axes[1, 2].set_title('碰撞率')
        axes[1, 2].set_ylim(0, 1.0)
        for i, v in enumerate(collision_rates):
            axes[1, 2].text(i, v + 0.02, f'{v:.2f}', ha='center')
        
        # 设置整体标题
        model_name = actor_path.split('/')[-1] if '/' in actor_path else actor_path
        plt.suptitle(f'模型 {model_name} 在不同难度级别的性能', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图表到多难度测试目录
        comparison_img_path = os.path.join(multi_test_dir, "difficulty_comparison.png")
        plt.savefig(comparison_img_path)
        plt.close()
        print(f"难度级别比较图已保存到: {comparison_img_path}")
    except Exception as e:
        print(f"绘制性能比较图时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 保存完整结果 (Pickle格式)
    try:
        pickle_results_path = os.path.join(multi_test_dir, "all_results.pkl")
        with open(pickle_results_path, 'wb') as f:
            pkl.dump(all_results, f, pkl.HIGHEST_PROTOCOL)
        print(f"完整测试结果(Pickle格式)已保存到: {pickle_results_path}")
        
        # 同时保存为JSON格式
        json_results_path = os.path.join(multi_test_dir, "all_results.json")
        with open(json_results_path, 'w', encoding='utf-8') as f:
            json.dump(convert_to_json_compatible(all_results), f, ensure_ascii=False, indent=4)
        print(f"完整测试结果(JSON格式)已保存到: {json_results_path}")
    except Exception as e:
        print(f"保存完整测试结果时出错: {e}")
    
    # 更新测试结果索引
    create_test_results_index()
    
    return all_results


# ==============================================================================
# 模块: 命令行入口与模式分发
# ==============================================================================

def main():
    """主函数
    """
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='MASAC with Curriculum Learning')
    parser.add_argument('--use_curriculum', action='store_true', help='使用课程学习框架')
    parser.add_argument('--render', action='store_true', help='是否渲染环境')
    parser.add_argument('--test', action='store_true', help='测试模式（加载已训练的模型）')
    parser.add_argument('--model_path', type=str, default='models/final/final_model', 
                        help='测试模式下加载的模型路径')
    # 添加更多测试相关选项
    parser.add_argument('--test_episodes', type=int, default=None, 
                       help='蒙特卡洛测试的回合数，默认使用全局变量TEST_EPIOSDE')
    parser.add_argument('--hero_count', type=int, default=1,
                       help='测试或训练时使用的友方无人机数量，默认为1')
    parser.add_argument('--enemy_count', type=int, default=3,
                       help='测试或训练时使用的敌方无人机数量，默认为3')
    parser.add_argument('--obstacle_count', type=int, default=2,
                       help='测试时使用的障碍物数量，默认为2')
    parser.add_argument('--test_speed', type=float, default=None,
                       help='测试时使用的无人机速度，不设置则使用默认速度')
    # 添加多难度测试选项
    parser.add_argument('--multi_difficulty_test', action='store_true',
                       help='在多个难度级别上进行蒙特卡洛测试')
    parser.add_argument('--max_obstacle', type=int, default=5,
                       help='多难度测试时的最大障碍物数量，默认为5')
    parser.add_argument('--step_size', type=float, default=1.0,
                       help='多难度测试时的障碍物数量步长，默认为1')
    parser.add_argument('--test_difficulty', type=str, default=None,
                       help='自定义难度测试，格式为逗号分隔的难度值，例如"1,2,3,4,5"')
    # 添加测试结果分析选项
    parser.add_argument('--analyze', action='store_true', 
                       help='分析已有的测试结果而非进行新的测试')
    parser.add_argument('--result_path', type=str, default=None,
                       help='测试结果文件路径，默认自动搜索')
    # 添加结果保存目录选项
    parser.add_argument('--results_dir', type=str, default=None,
                       help='测试结果保存目录，默认为"D:/pa/path planning2/results"')
    parser.add_argument('--create_index', action='store_true',
                       help='生成测试结果索引HTML文件')
    # 添加日志级别控制
    parser.add_argument('--log_level', type=str, choices=['debug', 'info', 'warning', 'error'], default='info',
                      help='设置日志级别：debug(调试), info(信息), warning(警告), error(错误)')
    # 添加多次训练模式参数
    parser.add_argument('--multi_run', action='store_true',
                       help='启用多次训练模式，使用不同随机种子进行多次训练')
    parser.add_argument('--num_runs', type=int, default=3,
                       help='多次训练模式下的训练次数，默认为3次')
    parser.add_argument('--seeds', type=str, default=None,
                       help='自定义随机种子列表，格式为逗号分隔的数字，例如"42,123,456"。如不指定则自动生成')
    parser.add_argument('--seed', type=int, default=42,
                       help='固定种子模式的基础种子，默认42')
    parser.add_argument('--random_seed', action='store_true',
                       help='启用随机种子模式，每次运行使用不同种子（不启用时使用固定种子）')
    parser.add_argument('--no_curriculum_backend', type=str, choices=['attention_masac', 'standard_sac'],
                       default='attention_masac',
                       help='无课程学习时使用的训练后端：attention_masac(注意力MASAC) 或 standard_sac(旧版标准SAC)')
    parser.add_argument('--share_follower_policy_no_curriculum', action='store_true',
                       help='无课程学习注意力后端下启用共享随从策略（所有随从共用同一个Actor）')
    parser.add_argument('--disable_attention', action='store_true',
                       help='禁用注意力机制，使用无注意力 Actor/Critic（用于消融实验）')
    parser.add_argument('--train_episodes', type=int, default=500,
                       help='训练回合数（无课程学习时生效）')
    parser.add_argument('--save_interval', type=int, default=100,
                       help='模型保存间隔（无课程学习时生效）')
    parser.add_argument('--exploration_episodes', type=int, default=20,
                       help='加噪探索回合数（无课程学习时生效）')
    parser.add_argument('--training_start_ratio', type=float, default=0.3,
                       help='经验池达到容量比例后开始训练（无课程学习时生效，范围0~1）')
    parser.add_argument('--max_consecutive_nonfinite_updates', type=int, default=20,
                       help='连续出现非有限更新时的自动停训阈值（无课程学习注意力后端生效）')
    parser.add_argument('--ablation_tag', type=str, default='ablation_no_curriculum',
                       help='无课程学习训练输出标签；保留默认值时将按注意力/GAT自动分流到独立目录')
    parser.add_argument('--ablation_output_root', type=str, default=None,
                       help='无课程学习训练输出根目录，默认当前项目目录')
    parser.add_argument('--gat', action='store_true', help='启用图注意力机制(GAT)代替普通多头注意力')
                       
    args = parser.parse_args()
    
    # 设置全局日志级别
    log_level_map = {
        'debug': LOG_DEBUG,
        'info': LOG_INFO,
        'warning': LOG_WARNING,
        'error': LOG_ERROR
    }
    set_log_level(log_level_map.get(args.log_level, LOG_INFO))
    log(f"日志级别设置为: {args.log_level.upper()}", LOG_INFO)

    if args.random_seed:
        startup_seed = random.SystemRandom().randint(1, 10000)
        print(f"随机种子模式已启用，启动种子: {startup_seed}")
    else:
        startup_seed = args.seed
        print(f"固定种子模式，使用种子: {startup_seed}")

    set_seed(startup_seed)
    
    global RENDER, action_number, RESULTS_DIR, TRAINING_RESULTS_FILE, TEST_RESULTS_BASE
    
    # 如果指定了结果目录，更新全局变量
    if args.results_dir:
        RESULTS_DIR = args.results_dir
        TEST_RESULTS_BASE = os.path.join(RESULTS_DIR, "test_results")
        TRAINING_RESULTS_FILE = os.path.join(RESULTS_DIR, "MASAC_curriculum")
        print(f"结果将保存在: {RESULTS_DIR}")
        ensure_dir_exists(RESULTS_DIR)
        ensure_dir_exists(TEST_RESULTS_BASE)

    # 有课程学习默认写入独立目录，避免注意力/GAT/无注意力结果混放。
    if (not args.results_dir) and args.use_curriculum:
        curriculum_output_root = args.ablation_output_root or os.path.dirname(os.path.abspath(__file__))
        curriculum_use_attention = not bool(args.disable_attention)
        curriculum_use_gat = bool(args.gat)
        curriculum_paths = prepare_curriculum_result_roots(
            output_root=curriculum_output_root,
            use_attention=curriculum_use_attention,
            use_gat=curriculum_use_gat
        )
        RESULTS_DIR = curriculum_paths["result_dir"]
        TEST_RESULTS_BASE = curriculum_paths["test_results_base"]
        TRAINING_RESULTS_FILE = curriculum_paths["training_results_file_prefix"]
        if not curriculum_use_attention:
            curriculum_mode_name = "有课程+无注意力消融"
        elif curriculum_use_gat:
            curriculum_mode_name = "有课程+GAT注意力"
        else:
            curriculum_mode_name = "有课程+注意力"
        print(f"{curriculum_mode_name}将使用独立输出目录:")
        print(f"- 结果目录: {RESULTS_DIR}")
        print(f"- 测试目录: {TEST_RESULTS_BASE}")

    # 无课程学习测试默认写入独立目录，保持与课程学习消融一致的目录层级。
    is_no_curriculum_testing = (not args.use_curriculum) and (args.test or args.multi_difficulty_test)
    if (not args.results_dir) and is_no_curriculum_testing:
        no_curriculum_output_root = args.ablation_output_root or os.path.dirname(os.path.abspath(__file__))
        no_curriculum_use_attention = not bool(args.disable_attention)

        inferred_tag = infer_ablation_tag_from_model_path(args.model_path)
        fallback_tag = _sanitize_tag(getattr(args, "ablation_tag", "ablation_no_curriculum"))
        if fallback_tag == "ablation_no_curriculum":
            fallback_tag = _default_no_curriculum_tag(
                use_attention=no_curriculum_use_attention,
                use_gat=bool(args.gat)
            )

        no_curriculum_tag = inferred_tag or fallback_tag
        no_curriculum_result_paths = prepare_no_curriculum_result_roots(
            output_root=no_curriculum_output_root,
            ablation_tag=no_curriculum_tag
        )

        RESULTS_DIR = no_curriculum_result_paths["result_dir"]
        TEST_RESULTS_BASE = no_curriculum_result_paths["test_results_base"]
        TRAINING_RESULTS_FILE = no_curriculum_result_paths["training_results_file_prefix"]

        print("无课程学习测试将使用独立输出目录:")
        print(f"- 实验标签: {no_curriculum_tag}")
        print(f"- 结果目录: {RESULTS_DIR}")
        print(f"- 测试目录: {TEST_RESULTS_BASE}")
    
    # 如果只是创建索引，直接调用索引创建函数后返回
    if args.create_index:
        print("正在创建测试结果索引...")
        index_path = create_test_results_index()
        print(f"索引已创建: {index_path}")
        return
    
    # 如果只是分析结果，直接调用分析函数
    if args.analyze:
        print("分析测试结果模式")
        if args.result_path:
            analyze_test_results([args.result_path])
        else:
            analyze_test_results()
        return

    # 在无课程学习训练模式下，若用户未显式传参，则默认使用 1/1/1 场景配置。
    def _arg_provided(flag_name):
        return any(
            token == flag_name or token.startswith(f"{flag_name}=")
            for token in sys.argv[1:]
        )

    is_no_curriculum_training = (
        (not args.use_curriculum)
        and (not args.test)
        and (not args.multi_difficulty_test)
    )
    if is_no_curriculum_training:
        if not _arg_provided('--hero_count'):
            args.hero_count = 1
        if not _arg_provided('--enemy_count'):
            args.enemy_count = 1
        if not _arg_provided('--obstacle_count'):
            args.obstacle_count = 1
        print(
            "无课程学习训练默认配置: hero_count=1, enemy_count=1, obstacle_count=1 "
            "(可通过命令行参数覆盖)"
        )
    
    # 设置渲染标志：测试时默认渲染，训练时根据参数决定
    if args.test or args.multi_difficulty_test:
        RENDER = True
    else:
        RENDER = args.render
    
    # 使用解析后的参数或默认值
    n_agent = args.hero_count
    m_enemy = args.enemy_count
    share_follower_policy_no_curriculum = bool(
        args.share_follower_policy_no_curriculum and (not args.use_curriculum)
    )
    use_attention = not bool(args.disable_attention)
    print(f"设置友方无人机数量 (n_agent): {n_agent}")
    print(f"设置敌方无人机数量 (m_enemy): {m_enemy}")
    print(f"注意力机制: {'启用' if use_attention else '禁用'}")
    
    # 创建一个临时环境实例以获取动作空间信息
    # 使用 n_agent 和 m_enemy 变量
    temp_env = RlGame(leader_count=n_agent, follower_count=m_enemy, obstacle_num=args.obstacle_count, render=False).unwrapped
    action_number = temp_env.action_space.shape[0]
    temp_env.close()
    
    # 更新main_SAC模块中的全局变量action_number
    import main_SAC
    main_SAC.action_number = action_number
    
    if args.multi_difficulty_test:
        # 多难度测试模式
        print("启动多难度蒙特卡洛测试模式")
        # 设置难度级别配置
        difficulty_levels = []
        
        # 如果提供了自定义难度列表
        if args.test_difficulty:
            try:
                custom_difficulties = [int(d) for d in args.test_difficulty.split(',')]
                print(f"使用自定义难度级别: {custom_difficulties}")
                difficulty_levels = [
                    {'obstacle_count': d, 'uav_speed': args.test_speed or 1.0}
                    for d in custom_difficulties
                ]
            except ValueError:
                print(f"无效的自定义难度格式: {args.test_difficulty}，将使用默认设置")
                difficulty_levels = []
        
        # 如果没有提供自定义难度或解析失败，使用默认生成的难度级别
        if not difficulty_levels:
            max_obstacle = args.max_obstacle
            step = args.step_size
            
            print(f"生成难度级别 - 最大障碍物: {max_obstacle}, 步长: {step}")
            obstacle_counts = [int(1 + i * step) for i in range(int((max_obstacle - 1) / step) + 1)]
            
            difficulty_levels = [
                {'obstacle_count': d, 'uav_speed': args.test_speed or 1.0}
                for d in obstacle_counts
            ]
        
        # 输出最终使用的难度级别
        print(f"将测试的难度级别配置: {difficulty_levels}")
                
        # 运行多难度测试
        monte_carlo_test(
            actor_path=args.model_path,
            critic_path=args.model_path,  # 使用相同的路径前缀，加载函数会自动添加_critic_{i}.pth
            test_nums=args.test_episodes or TEST_EPIOSDE,
            base_difficulty_levels=difficulty_levels,
            share_follower_policy=share_follower_policy_no_curriculum,
            use_attention=use_attention,
            use_gat = args.gat
        )
    elif args.test:
        # 单难度测试模式
        print(f"启动单一难度蒙特卡洛测试模式: 友方={n_agent}, 敌方={m_enemy}, 障碍物={args.obstacle_count}")
        if args.test_speed:
            print(f"指定无人机速度: {args.test_speed}")
        
        # 设置测试选项
        test_options = {
            'hero_count': n_agent, # 使用 n_agent
            'enemy_count': m_enemy, # 使用 m_enemy
            'obstacle_count': args.obstacle_count,
            'uav_speed': args.test_speed
        }
        
        # 运行测试
        run_monte_carlo_test(
            model_path=args.model_path,
            test_episodes=args.test_episodes or TEST_EPIOSDE, # TEST_EPIOSDE 也需要定义
            test_options=test_options,
            share_follower_policy=share_follower_policy_no_curriculum,
            use_attention=use_attention,
            use_gat=args.gat
        )
    else:
        # 训练模式
        if args.use_curriculum:
            if args.multi_run:
                print("使用多次课程学习框架进行训练")
                # 传递 n_agent 和 m_enemy 给 run_multi_seed_curriculum
                run_multi_seed_curriculum(args, n_agent, m_enemy)
            else:
                print("使用单次课程学习框架进行训练")
                # 传递 n_agent 和 m_enemy 给 run_with_curriculum
                run_with_curriculum(args, n_agent, m_enemy, seed=startup_seed)
        else:
            if args.no_curriculum_backend == 'standard_sac':
                print("使用标准SAC（无注意力、无课程学习）进行训练")
                if args.share_follower_policy_no_curriculum:
                    print("提示: --share_follower_policy_no_curriculum 仅对 attention_masac 后端生效，standard_sac 后端将忽略该开关")
                import main_SAC

                # 同步标准SAC的智能体配置，避免与默认全局值不一致。
                main_SAC.N_Agent = n_agent
                main_SAC.M_Enemy = m_enemy
                main_SAC.RENDER = RENDER
                main_SAC.Switch = 1

                # 为无课程学习消融实验创建独立输出路径，避免覆盖历史模型和结果。
                ablation_output_root = args.ablation_output_root or os.path.dirname(os.path.abspath(__file__))
                no_curriculum_paths = prepare_no_curriculum_output_paths(
                    output_root=ablation_output_root,
                    ablation_tag=args.ablation_tag
                )

                main_SAC.configure_output_paths(
                    model_path_leader=no_curriculum_paths["leader_model_path"],
                    model_path_follower=no_curriculum_paths["follower_model_path"],
                    training_results_path=no_curriculum_paths["training_result_path"]
                )

                print("标准SAC无课程学习输出路径:")
                print(f"- 运行标识: {no_curriculum_paths['run_stamp']}")
                print(f"- 模型目录: {no_curriculum_paths['model_dir']}")
                print(f"- 结果文件: {no_curriculum_paths['training_result_path']}")

                env = RlGame(leader_count=n_agent, follower_count=m_enemy, obstacle_num=args.obstacle_count, render=RENDER).unwrapped
                main_SAC.run(env)
            else:
                print(f"使用{'注意力' if use_attention else '无注意力'}MASAC（无课程学习）进行训练")
                if args.multi_run:
                    print("提示: 当前无课程学习注意力模式暂不启用多次训练，执行单次训练")
                run_attention_no_curriculum(args, n_agent, m_enemy, seed=startup_seed)


# ==============================================================================
# 模块: 程序启动与全局默认参数
# ==============================================================================

if __name__ == "__main__":
    # 定义一些全局常量（如果测试模式需要）
    # 这些值应该与 main_SAC.py 中的默认值或 args 的默认值对齐
    TRAIN_NUM = 1 # 添加 TRAIN_NUM 定义
    TEST_EPIOSDE = 100 
    # 定义 state_number 和 action_number 的默认值，它们会被覆盖
    state_number = 7 # 假设默认状态维度
    action_number = 2 # 假设默认动作维度
    
    # 设置默认的智能体数量，与命令行参数默认值保持一致
    N_Agent = 1  # 主机数量默认为1
    M_Enemy = 3  # 从机数量默认为3
    
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 定义其他可能在全局范围使用的常量
    EP_LEN = 1000
    MemoryCapacity = 50000
    BATCH = 256
    RESULTS_DIR = "D:/pa2/path planning2/results"
    TEST_RESULTS_BASE = os.path.join(RESULTS_DIR, "test_results")
    TRAINING_RESULTS_FILE = os.path.join(RESULTS_DIR, "MASAC_curriculum.pkl")

    main()
    
