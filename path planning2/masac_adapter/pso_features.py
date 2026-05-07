import math
import numpy as np

from rl_env.components.entities import Constants


class PSOFeatureExtractor:
    """Lightweight PSO-inspired feature extractor."""

    def __init__(self, obstacle_radius=120.0, formation_distance=45.0):
        width = float(getattr(Constants, "AREA_WITH", 600.0))
        height = float(getattr(Constants, "AREA_HEIGHT", 500.0))
        self.map_diag = float(math.hypot(width, height))
        if self.map_diag <= 0.0:
            self.map_diag = 1.0

        self.obstacle_radius = float(obstacle_radius)
        self.formation_distance = float(formation_distance)
        self.feature_dim = 6

    def extract(self, entity_manager):
        """Extract normalized global features from the entity manager."""
        if entity_manager is None:
            return np.zeros((self.feature_dim,), dtype=np.float32)

        leaders = getattr(entity_manager, "leaders", []) or []
        goals = getattr(entity_manager, "goals", []) or []
        if not leaders or not goals:
            return np.zeros((self.feature_dim,), dtype=np.float32)

        leader = leaders[0]
        goal = goals[0]

        dx = goal.pos_x - leader.pos_x
        dy = goal.pos_y - leader.pos_y
        dist_to_goal = math.hypot(dx, dy)
        dist_norm = min(dist_to_goal / self.map_diag, 1.0)

        angle_to_goal = math.atan2(dy, dx)
        angle_sin = math.sin(angle_to_goal)
        angle_cos = math.cos(angle_to_goal)

        obstacles = getattr(entity_manager, "obstacles", []) or []
        obstacle_potential = 0.0
        if obstacles:
            heading_x = math.cos(leader.theta)
            heading_y = math.sin(leader.theta)
            for obs in obstacles:
                vx = obs.pos_x - leader.pos_x
                vy = obs.pos_y - leader.pos_y
                dist = math.hypot(vx, vy)
                if dist <= 1e-6 or dist > self.obstacle_radius:
                    continue
                ux = vx / dist
                uy = vy / dist
                forwardness = max(0.0, ux * heading_x + uy * heading_y)
                obstacle_potential += (1.0 - dist / self.obstacle_radius) * forwardness
            obstacle_potential = obstacle_potential / max(1, len(obstacles))
        obstacle_potential = float(np.clip(obstacle_potential, 0.0, 1.0))

        followers = getattr(entity_manager, "followers", []) or []
        formation_error = 0.0
        alignment_mean = 0.0
        if followers:
            distances = [leader.distance_to(follower) for follower in followers]
            formation_error = float(np.mean([abs(d - self.formation_distance) for d in distances]))
            formation_error = formation_error / self.map_diag
            alignment_mean = float(np.mean([math.cos(follower.theta - leader.theta) for follower in followers]))

        formation_error = float(np.clip(formation_error, 0.0, 1.0))
        alignment_mean = float(np.clip(alignment_mean, -1.0, 1.0))

        features = np.array(
            [
                dist_norm,
                obstacle_potential,
                angle_sin,
                angle_cos,
                formation_error,
                alignment_mean,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def extract_from_env(self, env):
        if env is None or not hasattr(env, "entity_manager"):
            return np.zeros((self.feature_dim,), dtype=np.float32)
        return self.extract(env.entity_manager)
