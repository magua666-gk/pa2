import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 从 masac_adapter 导入注意力机制和角色 ID 常量
from masac_adapter.masac_adapter import MultiHeadAttention, GraphAttention, LEADER_TYPE_ID, FOLLOWER_TYPE_ID

class CriticObsActEncoder(nn.Module):
    """Critic 观测动作编码器
    
    将观测与动作编码为隐藏表示
    """
    def __init__(self, state_dim, action_dim, embed_dim, hidden_dims=[256, 128]):
        """初始化观测动作编码器
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            embed_dim: 输出嵌入维度
            hidden_dims: 隐藏层维度列表
        """
        super(CriticObsActEncoder, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        
        # 构建 MLP 层
        layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # 最终输出层
        layers.append(nn.Linear(input_dim, embed_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # 使用 Xavier 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, obs, act):
        """前向传播
        
        Args:
            obs: 观测张量 [..., state_dim]
            act: 动作张量 [..., action_dim]
            
        Returns:
            embedding: 编码后的嵌入 [..., embed_dim]
        """
        # 拼接观测和动作
        sa = torch.cat([obs, act], dim=-1)
        
        # 编码
        return self.mlp(sa)

class QHead(nn.Module):
    """Q 值头部
    
    双 Q 结构，用于计算 Q 值
    """
    def __init__(self, input_dim, hidden_dims=[256, 128]):
        """初始化 Q 值头部
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
        """
        super(QHead, self).__init__()
        
        # 共享层
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # 双 Q 输出层
        self.q1_out = nn.Linear(hidden_dims[-1], 1)
        self.q2_out = nn.Linear(hidden_dims[-1], 1)
        
        # 使用 Xavier 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Q 值输出层使用较小的初始化值
        nn.init.uniform_(self.q1_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q2_out.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.q1_out.bias)
        nn.init.zeros_(self.q2_out.bias)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入特征 [..., input_dim]
            
        Returns:
            q1: 第一个 Q 值 [..., 1]
            q2: 第二个 Q 值 [..., 1]
        """
        features = self.shared_layers(x)
        q1 = self.q1_out(features)
        q2 = self.q2_out(features)
        return q1, q2

class StructuredAttentionCriticNet(nn.Module):
    """结构化注意力 Critic 网络
    
    处理结构化的观测和动作，使用注意力机制聚合信息
    """
    def __init__(self, state_dim, action_dim, embed_dim=128, n_heads=4, hidden_dims=[256, 128], dropout=0.1,
                 use_attention=True, use_gat=False, pso_dim=0, pso_for_followers=True):
        """初始化结构化注意力 Critic 网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            embed_dim: 嵌入维度
            n_heads: 注意力头数
            hidden_dims: 隐藏层维度列表
            dropout: Dropout 概率
            use_attention: 是否启用注意力机制
        """
        super(StructuredAttentionCriticNet, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.use_attention = bool(use_attention)
        self.use_gat = bool(use_gat)
        self.pso_dim = int(pso_dim) if pso_dim else 0
        self.pso_for_followers = bool(pso_for_followers)
        
        # 编码器
        self.leader_encoder = CriticObsActEncoder(state_dim, action_dim, embed_dim)
        self.follower_encoder = CriticObsActEncoder(state_dim, action_dim, embed_dim)

        if self.use_attention:
            if self.use_gat:
                self.leader_sees_followers_attention = GraphAttention(embed_dim, n_heads, dropout)
                self.follower_context_attention = GraphAttention(embed_dim, n_heads, dropout)
            else:
                self.leader_sees_followers_attention = MultiHeadAttention(embed_dim, n_heads, dropout)
                self.follower_context_attention = MultiHeadAttention(embed_dim, n_heads, dropout)
        else:
            self.leader_sees_followers_attention = nn.Identity()
            self.follower_context_attention = nn.Identity()
        
        # Q 头
        # Leader输入: embed_dim * 2 + pso_dim (自身编码 + Follower上下文 + PSO特征)
        q_input_dim_leader = embed_dim * 2 + self.pso_dim
        # Follower输入: embed_dim * 2 + pso_dim (自身编码 + 全局上下文 + PSO特征)
        follower_pso_dim = self.pso_dim if self.pso_for_followers else 0
        q_input_dim_follower = embed_dim * 2 + follower_pso_dim
        
        self.leader_q_head = QHead(q_input_dim_leader, hidden_dims)
        self.follower_q_head = QHead(q_input_dim_follower, hidden_dims)
        
        # 目标网络
        self.target_leader_encoder = CriticObsActEncoder(state_dim, action_dim, embed_dim)
        self.target_follower_encoder = CriticObsActEncoder(state_dim, action_dim, embed_dim)

        if self.use_attention:
            if self.use_gat:
                self.target_leader_sees_followers_attention = GraphAttention(embed_dim, n_heads, dropout)
                self.target_follower_context_attention = GraphAttention(embed_dim, n_heads, dropout)
            else:
                self.target_leader_sees_followers_attention = MultiHeadAttention(embed_dim, n_heads, dropout)
                self.target_follower_context_attention = MultiHeadAttention(embed_dim, n_heads, dropout)
        else:
            self.target_leader_sees_followers_attention = nn.Identity()
            self.target_follower_context_attention = nn.Identity()
        
        self.target_leader_q_head = QHead(q_input_dim_leader, hidden_dims)
        self.target_follower_q_head = QHead(q_input_dim_follower, hidden_dims)
        
        # 加载目标网络初始参数
        self._load_initial_target_params()
    
    def _load_initial_target_params(self):
        """加载目标网络初始参数"""
        self.target_leader_encoder.load_state_dict(self.leader_encoder.state_dict())
        self.target_follower_encoder.load_state_dict(self.follower_encoder.state_dict())
        self.target_leader_sees_followers_attention.load_state_dict(self.leader_sees_followers_attention.state_dict())
        self.target_follower_context_attention.load_state_dict(self.follower_context_attention.state_dict())
        self.target_leader_q_head.load_state_dict(self.leader_q_head.state_dict())
        self.target_follower_q_head.load_state_dict(self.follower_q_head.state_dict())

    @staticmethod
    def _masked_mean(features, validity_mask):
        """Compute masked mean over agent dimension.

        Args:
            features: [B, N, E]
            validity_mask: [B, N] bool
        """
        if features.size(1) == 0:
            return torch.zeros(
                features.size(0),
                features.size(-1),
                device=features.device,
                dtype=features.dtype
            )

        mask = validity_mask.to(dtype=features.dtype).unsqueeze(-1)  # [B, N, 1]
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (features * mask).sum(dim=1) / denom

    def _prepare_pso_features(self, pso_features, batch_size, device, dtype):
        if self.pso_dim <= 0:
            return None

        if pso_features is None:
            return torch.zeros(batch_size, self.pso_dim, device=device, dtype=dtype)

        pso = pso_features.to(device=device, dtype=dtype).reshape(batch_size, -1)
        if pso.shape[1] == self.pso_dim:
            return pso

        if pso.shape[1] > self.pso_dim:
            return pso[:, :self.pso_dim]

        padded = torch.zeros(batch_size, self.pso_dim, device=device, dtype=dtype)
        padded[:, :pso.shape[1]] = pso
        return padded

    def _build_no_attention_context(self, leader_embedding, follower_embeddings, mask_followers):
        """Construct leader/follower contexts without attention for ablation."""
        batch_size = leader_embedding.size(0)
        max_f = follower_embeddings.size(1)
        follower_validity = mask_followers.bool()

        # Leader context: valid followers mean; if no follower, zeros.
        leader_context = self._masked_mean(follower_embeddings, follower_validity)

        # Follower context: global (leader + valid followers) mean.
        leader_as_agent = leader_embedding.unsqueeze(1)  # [B, 1, E]
        all_agents = torch.cat([leader_as_agent, follower_embeddings], dim=1)  # [B, 1+F, E]
        leader_valid = torch.ones(batch_size, 1, dtype=torch.bool, device=leader_embedding.device)
        all_validity = torch.cat([leader_valid, follower_validity], dim=1)  # [B, 1+F]
        global_context = self._masked_mean(all_agents, all_validity)  # [B, E]

        if max_f > 0:
            follower_context = global_context.unsqueeze(1).expand(-1, max_f, -1)  # [B, F, E]
        else:
            follower_context = torch.zeros(
                batch_size,
                0,
                self.embed_dim,
                device=leader_embedding.device,
                dtype=leader_embedding.dtype
            )

        return leader_context, follower_context
    
    def forward(self, obs_leader, obs_followers, act_leader, act_followers, mask_followers, pso_features=None):
        """前向传播
        
        Args:
            obs_leader: Leader 的观测 [batch_size, state_dim]
            obs_followers: Followers 的观测 [batch_size, max_followers, state_dim]
            act_leader: Leader 的动作 [batch_size, action_dim]
            act_followers: Followers 的动作 [batch_size, max_followers, action_dim]
            mask_followers: Followers 的掩码 [batch_size, max_followers]
            
        Returns:
            q1_leader: Leader 的第一个 Q 值 [batch_size, 1]
            q2_leader: Leader 的第二个 Q 值 [batch_size, 1]
            q1_followers: Followers 的第一个 Q 值 [batch_size, max_followers, 1]
            q2_followers: Followers 的第二个 Q 值 [batch_size, max_followers, 1]
        """
        B = obs_leader.shape[0]
        max_F = obs_followers.shape[1]

        pso = self._prepare_pso_features(pso_features, B, obs_leader.device, obs_leader.dtype)
        
        # 编码 Leader
        leader_embedding = self.leader_encoder(obs_leader, act_leader)  # [B, E]
        
        # 编码 Followers (处理 B, max_F, D -> B*max_F, D -> B*max_F, E -> B, max_F, E)
        obs_f_flat = obs_followers.reshape(B * max_F, self.state_dim)  # [B*max_F, D_s]
        act_f_flat = act_followers.reshape(B * max_F, self.action_dim)  # [B*max_F, D_a]
        follower_embeds_flat = self.follower_encoder(obs_f_flat, act_f_flat)  # [B*max_F, E]
        follower_embeddings = follower_embeds_flat.reshape(B, max_F, self.embed_dim)  # [B, max_F, E]
        
        # === Leader Q 值计算 ===
        if self.use_attention:
            # 计算 Follower 上下文 (使用注意力)
            attn_mask = mask_followers.unsqueeze(1).unsqueeze(1).expand(-1, self.n_heads, -1, -1)

            # 注意力输入：Leader作为Query，follower_embeddings 作为 K, V
            leader_embedding_unsqueezed = leader_embedding.unsqueeze(1)  # [B, 1, E]
            attn_output = self.leader_sees_followers_attention(
                queries=leader_embedding_unsqueezed,
                keys=follower_embeddings,
                values=follower_embeddings,
                mask=attn_mask
            )  # [B, 1, E]
            attn_output = attn_output.squeeze(1)  # [B, E]
        else:
            attn_output, _ = self._build_no_attention_context(
                leader_embedding,
                follower_embeddings,
                mask_followers
            )
        
        # 融合Leader自身特征和Follower上下文
        fused_leader_feature = torch.cat([leader_embedding, attn_output], dim=-1)  # [B, E*2]
        if pso is not None:
            fused_leader_feature = torch.cat([fused_leader_feature, pso], dim=-1)
        
        # 计算Leader的Q值
        q1_leader, q2_leader = self.leader_q_head(fused_leader_feature)  # [B, 1], [B, 1]

        # === Follower Q 值计算 (新增) ===
        if self.use_attention:
            # 构建全局上下文
            leader_embedding_unsqueezed_for_follower = leader_embedding.unsqueeze(1)  # Shape: [B, 1, E]
            all_agents_embeddings = torch.cat([leader_embedding_unsqueezed_for_follower, follower_embeddings], dim=1)  # Shape: [B, 1 + max_F, E]

            # 创建注意力掩码
            # mask_followers: [B, max_F], True表示有效的followers
            batch_size = obs_leader.size(0)
            leader_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=obs_leader.device)  # Shape: [B, 1]
            all_agents_validity = torch.cat([leader_mask, mask_followers.bool()], dim=1)  # Shape: [B, 1 + N_f]

            # 构建key_padding_mask，对于MultiHeadAttention，True表示被掩码(忽略)的位置
            key_padding_mask_for_follower_context_attn = ~all_agents_validity

            # 将key_padding_mask转换为与MultiHeadAttention兼容的mask格式
            # 创建形状为 [B, num_heads, max_F, 1 + max_F] 的mask
            follower_context_attn_mask = key_padding_mask_for_follower_context_attn.unsqueeze(1).unsqueeze(1)
            follower_context_attn_mask = follower_context_attn_mask.expand(-1, self.n_heads, follower_embeddings.size(1), -1)

            # Follower上下文注意力计算 (使用mask而不是key_padding_mask)
            contextual_info_for_followers = self.follower_context_attention(
                queries=follower_embeddings,  # [B, max_F, E]
                keys=all_agents_embeddings,  # [B, 1 + max_F, E]
                values=all_agents_embeddings,  # [B, 1 + max_F, E]
                mask=~follower_context_attn_mask  # 注意：MultiHeadAttention中mask为0的位置会被忽略，所以再次取反
            )  # [B, max_F, E]
        else:
            _, contextual_info_for_followers = self._build_no_attention_context(
                leader_embedding,
                follower_embeddings,
                mask_followers
            )
        
        # 融合Follower自身特征和上下文信息
        fused_follower_feature = torch.cat([follower_embeddings, contextual_info_for_followers], dim=-1)  # [B, max_F, 2*E]
        if pso is not None and self.pso_for_followers:
            pso_expanded = pso.unsqueeze(1).expand(-1, max_F, -1)
            fused_follower_feature = torch.cat([fused_follower_feature, pso_expanded], dim=-1)
        
        # 计算Follower的Q值
        q1_followers = torch.zeros(B, max_F, 1, device=obs_leader.device)  # [B, max_F, 1]
        q2_followers = torch.zeros(B, max_F, 1, device=obs_leader.device)  # [B, max_F, 1]
        
        for i in range(max_F):
            # 获取第i个follower的融合特征
            f_fused_i = fused_follower_feature[:, i, :]  # [B, E*2]
            
            # 计算Q值
            q1_f_i, q2_f_i = self.follower_q_head(f_fused_i)  # [B, 1], [B, 1]
            
            # 存储Q值
            q1_followers[:, i:i+1, :] = q1_f_i.unsqueeze(-1)  # [B, 1] -> [B, 1, 1]
            q2_followers[:, i:i+1, :] = q2_f_i.unsqueeze(-1)  # [B, 1] -> [B, 1, 1]
        
        return q1_leader, q2_leader, q1_followers, q2_followers
    
    def forward_target(self, obs_leader, obs_followers, act_leader, act_followers, mask_followers, pso_features=None):
        """使用目标网络计算 Q 值
        
        Args:
            obs_leader: Leader 的观测 [batch_size, state_dim]
            obs_followers: Followers 的观测 [batch_size, max_followers, state_dim]
            act_leader: Leader 的动作 [batch_size, action_dim]
            act_followers: Followers 的动作 [batch_size, max_followers, action_dim]
            mask_followers: Followers 的掩码 [batch_size, max_followers]
            
        Returns:
            q1_leader: Leader 的第一个 Q 值 [batch_size, 1]
            q2_leader: Leader 的第二个 Q 值 [batch_size, 1]
            q1_followers: Followers 的第一个 Q 值 [batch_size, max_followers, 1]
            q2_followers: Followers 的第二个 Q 值 [batch_size, max_followers, 1]
        """
        with torch.no_grad():
            B = obs_leader.shape[0]
            max_F = obs_followers.shape[1]

            pso = self._prepare_pso_features(pso_features, B, obs_leader.device, obs_leader.dtype)
            
            # 编码 Leader
            leader_embedding = self.target_leader_encoder(obs_leader, act_leader)  # [B, E]
            
            # 编码 Followers
            obs_f_flat = obs_followers.reshape(B * max_F, self.state_dim)  # [B*max_F, D_s]
            act_f_flat = act_followers.reshape(B * max_F, self.action_dim)  # [B*max_F, D_a]
            follower_embeds_flat = self.target_follower_encoder(obs_f_flat, act_f_flat)  # [B*max_F, E]
            follower_embeddings = follower_embeds_flat.reshape(B, max_F, self.embed_dim)  # [B, max_F, E]
            
            # === Leader Q 值计算 ===
            if self.use_attention:
                # 注意力掩码
                attn_mask = mask_followers.unsqueeze(1).unsqueeze(1).expand(-1, self.n_heads, -1, -1)

                # 注意力计算 (Leader关注Followers)
                leader_embedding_unsqueezed = leader_embedding.unsqueeze(1)  # [B, 1, E]
                attn_output = self.target_leader_sees_followers_attention(
                    queries=leader_embedding_unsqueezed,
                    keys=follower_embeddings,
                    values=follower_embeddings,
                    mask=attn_mask
                )  # [B, 1, E]
                attn_output = attn_output.squeeze(1)  # [B, E]
            else:
                attn_output, _ = self._build_no_attention_context(
                    leader_embedding,
                    follower_embeddings,
                    mask_followers
                )
            
            # 融合Leader自身特征和Follower上下文
            fused_leader_feature = torch.cat([leader_embedding, attn_output], dim=-1)  # [B, E*2]
            if pso is not None:
                fused_leader_feature = torch.cat([fused_leader_feature, pso], dim=-1)
            
            # 计算Leader的Q值
            q1_leader, q2_leader = self.target_leader_q_head(fused_leader_feature)  # [B, 1], [B, 1]
            
            # === Follower Q 值计算 (新增) ===
            if self.use_attention:
                # 构建全局上下文
                leader_embedding_unsqueezed_for_follower = leader_embedding.unsqueeze(1)  # Shape: [B, 1, E]
                all_agents_embeddings = torch.cat([leader_embedding_unsqueezed_for_follower, follower_embeddings], dim=1)  # Shape: [B, 1 + max_F, E]

                # 创建注意力掩码
                batch_size = obs_leader.size(0)
                leader_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=obs_leader.device)  # Shape: [B, 1]
                all_agents_validity = torch.cat([leader_mask, mask_followers.bool()], dim=1)  # Shape: [B, 1 + N_f]

                # 构建key_padding_mask
                key_padding_mask_for_follower_context_attn = ~all_agents_validity

                # 将key_padding_mask转换为与MultiHeadAttention兼容的mask格式
                # 创建形状为 [B, num_heads, max_F, 1 + max_F] 的mask
                follower_context_attn_mask = key_padding_mask_for_follower_context_attn.unsqueeze(1).unsqueeze(1)
                follower_context_attn_mask = follower_context_attn_mask.expand(-1, self.n_heads, follower_embeddings.size(1), -1)

                # Follower上下文注意力计算 (使用mask而不是key_padding_mask)
                contextual_info_for_followers = self.target_follower_context_attention(
                    queries=follower_embeddings,  # [B, max_F, E]
                    keys=all_agents_embeddings,  # [B, 1 + max_F, E]
                    values=all_agents_embeddings,  # [B, 1 + max_F, E]
                    mask=~follower_context_attn_mask  # 注意：MultiHeadAttention中mask为0的位置会被忽略，所以再次取反
                )  # [B, max_F, E]
            else:
                _, contextual_info_for_followers = self._build_no_attention_context(
                    leader_embedding,
                    follower_embeddings,
                    mask_followers
                )
            
            # 融合Follower自身特征和上下文信息
            fused_follower_feature = torch.cat([follower_embeddings, contextual_info_for_followers], dim=-1)  # [B, max_F, 2*E]
            if pso is not None and self.pso_for_followers:
                pso_expanded = pso.unsqueeze(1).expand(-1, max_F, -1)
                fused_follower_feature = torch.cat([fused_follower_feature, pso_expanded], dim=-1)
            
            # 计算Follower的Q值
            q1_followers = torch.zeros(B, max_F, 1, device=obs_leader.device)  # [B, max_F, 1]
            q2_followers = torch.zeros(B, max_F, 1, device=obs_leader.device)  # [B, max_F, 1]
            
            for i in range(max_F):
                # 获取第i个follower的融合特征
                f_fused_i = fused_follower_feature[:, i, :]  # [B, E*2]
                
                # 计算Q值
                q1_f_i, q2_f_i = self.target_follower_q_head(f_fused_i)  # [B, 1], [B, 1]
                
                # 存储Q值
                q1_followers[:, i:i+1, :] = q1_f_i.unsqueeze(-1)  # [B, 1] -> [B, 1, 1]
                q2_followers[:, i:i+1, :] = q2_f_i.unsqueeze(-1)  # [B, 1] -> [B, 1, 1]
            
            return q1_leader, q2_leader, q1_followers, q2_followers
    
    def soft_update(self, tau):
        """软更新目标网络参数
        
        Args:
            tau: 软更新系数
        """
        # 更新 Leader 编码器
        for target_param, param in zip(self.target_leader_encoder.parameters(), self.leader_encoder.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        # 更新 Follower 编码器
        for target_param, param in zip(self.target_follower_encoder.parameters(), self.follower_encoder.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        # 更新 Leader关注Followers注意力层
        for target_param, param in zip(self.target_leader_sees_followers_attention.parameters(), self.leader_sees_followers_attention.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        # 更新 Follower上下文注意力层 (新增)
        for target_param, param in zip(self.target_follower_context_attention.parameters(), self.follower_context_attention.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        # 更新 Q 头
        for target_param, param in zip(self.target_leader_q_head.parameters(), self.leader_q_head.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.target_follower_q_head.parameters(), self.follower_q_head.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau) 