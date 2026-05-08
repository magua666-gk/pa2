import math
import numpy as np

from rl_env.components.entities import Constants


class PSOFeatureGenerator:
    """Lightweight PSO-style global feature generator.

    The features are designed for Critic-only inputs and are normalized
    to be independent of the environment scale.
    """

    def __init__(
        self,
        use_pso=True,
        update_interval=5,
        pso_particles=8,
        pso_iterations=10,
        pso_inertia=0.5,
        pso_c1=1.0,
        pso_c2=1.0,
        obstacle_weight=30.0,
        obstacle_influence_radius=150.0,
        front_fov_deg=90.0,
        formation_target_distance=40.0,
        ema_alpha=0.2,
        feature_scales=None,
    ):
        self.feature_dim = 6
        self.use_pso = bool(use_pso)
        self.update_interval = max(1, int(update_interval))
        self.pso_particles = max(0, int(pso_particles))
        self.pso_iterations = max(0, int(pso_iterations))
        self.pso_inertia = float(pso_inertia)
        self.pso_c1 = float(pso_c1)
        self.pso_c2 = float(pso_c2)
        self.obstacle_weight = float(obstacle_weight)
        self.obstacle_influence_radius = float(obstacle_influence_radius)
        self.front_fov_rad = math.radians(float(front_fov_deg))
        self.formation_target_distance = float(formation_target_distance)

        self.ema_alpha = float(ema_alpha)
        if not 0.0 < self.ema_alpha <= 1.0:
            self.ema_alpha = 1.0

        if feature_scales is None:
            feature_scales = [0.5, 1.0, 0.5, 0.5, 1.0, 1.0]
        scales = np.asarray(feature_scales, dtype=np.float32).reshape(-1)
        if scales.size < self.feature_dim:
            padded = np.ones(self.feature_dim, dtype=np.float32)
            padded[:scales.size] = scales
            scales = padded
        elif scales.size > self.feature_dim:
            scales = scales[:self.feature_dim]
        self.feature_scales = scales

        self._rng = np.random.default_rng()
        self._step = 0
        self._last_features = np.zeros(self.feature_dim, dtype=np.float32)
        self._ema_features = np.zeros(self.feature_dim, dtype=np.float32)

        self._min_x = float(Constants.AREA_X)
        self._max_x = float(Constants.AREA_WITH)
        self._min_y = float(Constants.AREA_Y)
        self._max_y = float(Constants.AREA_HEIGHT)
        self._map_diag = math.hypot(self._max_x - self._min_x, self._max_y - self._min_y)
        if self._map_diag <= 0:
            self._map_diag = 1.0

    def reset(self):
        self._step = 0
        self._last_features = np.zeros(self.feature_dim, dtype=np.float32)
        self._ema_features = np.zeros(self.feature_dim, dtype=np.float32)

    def compute_from_env(self, env):
        global_state = self._extract_global_state(env)
        return self.compute(global_state)

    def compute(self, global_state):
        self._step += 1
        if self.update_interval > 1 and (self._step - 1) % self.update_interval != 0:
            return self._last_features.copy()

        features = self._compute_features(global_state)
        features = features * self.feature_scales
        if self.ema_alpha < 1.0:
            if self._step == 1:
                self._ema_features = features
            else:
                self._ema_features = self.ema_alpha * features + (1.0 - self.ema_alpha) * self._ema_features
            output = self._ema_features
        else:
            output = features
        self._last_features = output
        return output.copy()

    def _extract_global_state(self, env):
        if env is None or not hasattr(env, "entity_manager"):
            return {}

        entity_manager = env.entity_manager
        leaders = getattr(entity_manager, "leaders", []) or []
        followers = getattr(entity_manager, "followers", []) or []
        obstacles = getattr(entity_manager, "obstacles", []) or []
        goals = getattr(entity_manager, "goals", []) or []

        leader = leaders[0] if leaders else None
        goal = goals[0] if goals else None

        return {
            "leader": leader,
            "followers": followers,
            "obstacles": obstacles,
            "goal": goal,
        }

    def _compute_features(self, global_state):
        leader = global_state.get("leader")
        goal = global_state.get("goal")
        if leader is None or goal is None:
            return np.zeros(self.feature_dim, dtype=np.float32)

        leader_pos = np.array([leader.pos_x, leader.pos_y], dtype=np.float32)
        goal_pos = np.array([goal.pos_x, goal.pos_y], dtype=np.float32)
        obstacles = global_state.get("obstacles", []) or []
        followers = global_state.get("followers", []) or []

        obstacle_positions = np.array(
            [[obs.pos_x, obs.pos_y] for obs in obstacles],
            dtype=np.float32
        ) if obstacles else np.zeros((0, 2), dtype=np.float32)

        path_length = self._estimate_path_length(leader_pos, goal_pos, obstacle_positions)
        dist_to_goal = float(path_length) / self._map_diag

        desired_heading = math.atan2(goal_pos[1] - leader_pos[1], goal_pos[0] - leader_pos[0])
        heading_sin = math.sin(desired_heading)
        heading_cos = math.cos(desired_heading)

        obstacle_potential = self._obstacle_potential(leader_pos, leader.theta, obstacle_positions)
        formation_error = self._formation_error(leader_pos, followers)
        alignment = self._alignment(leader.theta, followers)

        features = np.array(
            [
                dist_to_goal,
                obstacle_potential,
                heading_sin,
                heading_cos,
                formation_error,
                alignment,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def _estimate_path_length(self, leader_pos, goal_pos, obstacle_positions):
        direct = float(np.linalg.norm(goal_pos - leader_pos))
        if not self.use_pso or self.pso_particles <= 0 or self.pso_iterations <= 0:
            return direct

        return self._pso_path_length(leader_pos, goal_pos, obstacle_positions, direct)

    def _pso_path_length(self, leader_pos, goal_pos, obstacle_positions, fallback):
        try:
            positions = self._rng.uniform(
                low=[self._min_x, self._min_y],
                high=[self._max_x, self._max_y],
                size=(self.pso_particles, 2),
            ).astype(np.float32)
            velocities = np.zeros_like(positions)

            personal_best = positions.copy()
            personal_best_cost = self._path_cost(personal_best, leader_pos, goal_pos, obstacle_positions)
            global_best_idx = int(np.argmin(personal_best_cost))
            global_best = personal_best[global_best_idx].copy()

            for _ in range(self.pso_iterations):
                r1 = self._rng.random(size=positions.shape).astype(np.float32)
                r2 = self._rng.random(size=positions.shape).astype(np.float32)

                velocities = (
                    self.pso_inertia * velocities
                    + self.pso_c1 * r1 * (personal_best - positions)
                    + self.pso_c2 * r2 * (global_best - positions)
                )
                positions = positions + velocities
                positions[:, 0] = np.clip(positions[:, 0], self._min_x, self._max_x)
                positions[:, 1] = np.clip(positions[:, 1], self._min_y, self._max_y)

                costs = self._path_cost(positions, leader_pos, goal_pos, obstacle_positions)
                improved = costs < personal_best_cost
                personal_best[improved] = positions[improved]
                personal_best_cost[improved] = costs[improved]

                global_best_idx = int(np.argmin(personal_best_cost))
                global_best = personal_best[global_best_idx].copy()

            return float(personal_best_cost[global_best_idx])
        except Exception:
            return float(fallback)

    def _path_cost(self, waypoints, leader_pos, goal_pos, obstacle_positions):
        dist_leader = np.linalg.norm(waypoints - leader_pos, axis=1)
        dist_goal = np.linalg.norm(goal_pos - waypoints, axis=1)
        cost = dist_leader + dist_goal

        if obstacle_positions.size > 0:
            diff = waypoints[:, None, :] - obstacle_positions[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            penalty = np.exp(-dists / max(self.obstacle_influence_radius, 1e-3))
            penalty = penalty.sum(axis=1) * self.obstacle_weight
            cost = cost + penalty

        return cost

    def _obstacle_potential(self, leader_pos, leader_theta, obstacle_positions):
        if obstacle_positions.size == 0:
            return 0.0

        heading = np.array([math.cos(leader_theta), math.sin(leader_theta)], dtype=np.float32)
        cos_limit = math.cos(self.front_fov_rad * 0.5)

        potential = 0.0
        for obs_pos in obstacle_positions:
            vec = obs_pos - leader_pos
            dist = float(np.linalg.norm(vec))
            if dist <= 1e-6 or dist > self.obstacle_influence_radius:
                continue

            direction = vec / dist
            if float(np.dot(heading, direction)) < cos_limit:
                continue

            potential += max(0.0, (self.obstacle_influence_radius - dist) / self.obstacle_influence_radius)

        normalized = potential / max(1.0, float(len(obstacle_positions)))
        return float(np.clip(normalized, 0.0, 1.0))

    def _formation_error(self, leader_pos, followers):
        if not followers:
            return 0.0

        distances = []
        for follower in followers:
            follower_pos = np.array([follower.pos_x, follower.pos_y], dtype=np.float32)
            distances.append(float(np.linalg.norm(follower_pos - leader_pos)))

        target = max(self.formation_target_distance, 1.0)
        error = np.mean(np.abs(np.asarray(distances) - target) / target)
        return float(np.clip(error, 0.0, 1.0))

    def _alignment(self, leader_theta, followers):
        if not followers:
            return 0.0

        alignments = []
        for follower in followers:
            alignments.append(math.cos(follower.theta - leader_theta))

        return float(np.clip(np.mean(alignments), -1.0, 1.0))
