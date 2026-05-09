import numpy as np
from rl_env.components.entities import Constants


class RewardCalculator:
    """Calculate environment rewards."""

    def __init__(self, entity_manager, weights=None):
        """Initialize reward calculator.

        Args:
            entity_manager: Entity manager.
            weights: Weights for different reward components.
        """
        self.entity_manager = entity_manager

        default_weights = {
            "edge": 1.0,
            "collision": 1.0,
            "goal": 1.0,
            "progress": 1.0,
            "formation": 1.0,
            "speed": 1.0,
            "position": 1.0,
            "time": 1.0,
            "follower_goal": 1.0,
        }

        self.weights = default_weights.copy()
        if weights is not None:
            self.weights.update(weights)

        self.goal_progress_scale = 2.0
        self.terminal_goal_reward = 1000.0
        self.terminal_follower_goal_reward = 500.0
        self.collision_penalty = -500.0
        self.boundary_penalty_scale = 5.0
        self.ideal_formation_distance = 40.0
        self.formation_tolerance = 10.0
        self.formation_error_falloff = 100.0
        self.follower_formation_bonus = 4.0
        self.leader_formation_bonus = 2.0
        self.formation_progress_scale = 1.0

        self._prev_goal_distances = {}
        self._goal_progress_time = {}
        self._last_goal_progress = {}
        self._prev_formation_errors = {}

    def reset(self):
        """Reset per-episode reward shaping history."""
        self._prev_goal_distances.clear()
        self._goal_progress_time.clear()
        self._last_goal_progress.clear()
        self._prev_formation_errors.clear()
        self._goal_reached_printed = False

    def _get_weight(self, name):
        return float(self.weights.get(name, 1.0))

    def _leader_has_won(self, leader):
        return bool(getattr(leader, "has_won", False))

    def _compute_boundary_penalty(self, agent, threshold=50.0):
        distances = [
            agent.pos_x - Constants.AREA_X,
            Constants.AREA_WITH - agent.pos_x,
            agent.pos_y - Constants.AREA_Y,
            Constants.AREA_HEIGHT - agent.pos_y,
        ]
        min_distance = max(0.0, min(distances))
        if min_distance >= threshold:
            return 0.0
        return -self.boundary_penalty_scale * (threshold - min_distance) / threshold

    def _compute_goal_progress(self, leader, current_distance):
        agent_key = id(leader)
        time_counter = int(getattr(self.entity_manager, "time_counter", 0))
        last_time = self._goal_progress_time.get(agent_key)

        if last_time == time_counter:
            return self._last_goal_progress.get(agent_key, 0.0)

        previous_distance = self._prev_goal_distances.get(agent_key)
        if previous_distance is None or time_counter <= 1 or (last_time is not None and time_counter < last_time):
            progress = 0.0
        else:
            progress = previous_distance - current_distance

        self._prev_goal_distances[agent_key] = current_distance
        self._goal_progress_time[agent_key] = time_counter
        self._last_goal_progress[agent_key] = progress
        return progress

    def _formation_error(self, distance):
        error = abs(distance - self.ideal_formation_distance) - self.formation_tolerance
        return max(0.0, error)

    def _formation_quality(self, distance):
        error = self._formation_error(distance)
        return max(0.0, 1.0 - error / self.formation_error_falloff)

    def _compute_formation_progress(self, follower, current_error):
        agent_key = id(follower)
        time_counter = int(getattr(self.entity_manager, "time_counter", 0))
        previous_error = self._prev_formation_errors.get(agent_key)
        self._prev_formation_errors[agent_key] = current_error

        if previous_error is None or time_counter <= 1:
            return 0.0
        return previous_error - current_error

    def compute_rewards(self):
        """Compute rewards for all agents.

        Returns:
            Reward array, one reward value per agent.
        """
        rewards = np.zeros(self.entity_manager.total_agents)

        for i, leader in enumerate(self.entity_manager.leaders):
            rewards[i] = self._compute_leader_reward(leader)

        for i, follower in enumerate(self.entity_manager.followers):
            rewards[i + len(self.entity_manager.leaders)] = self._compute_follower_reward(follower)

        return rewards

    def _compute_leader_reward(self, leader):
        """Compute leader reward."""
        edge_reward = self._compute_boundary_penalty(leader)

        obstacle_reward = 0.0
        for obstacle in self.entity_manager.obstacles:
            distance = leader.distance_to(obstacle)
            if distance < 20.0:
                obstacle_reward = self.collision_penalty
                break
            if distance < 40.0:
                obstacle_reward = min(obstacle_reward, -2.0 * (40.0 - distance) / 20.0)

        goal_reward = 0.0
        progress_reward = 0.0
        if self.entity_manager.goals:
            goal = self.entity_manager.goals[0]
            distance = leader.distance_to(goal)
            if self._leader_has_won(leader):
                team_quality = 1.0
                if self.entity_manager.followers:
                    follower_qualities = [
                        self._formation_quality(leader.distance_to(follower))
                        for follower in self.entity_manager.followers
                    ]
                    team_quality = float(np.mean(follower_qualities)) if follower_qualities else 0.0

                goal_reward = self.terminal_goal_reward * (0.25 + 0.75 * team_quality)
                if not getattr(self, "_goal_reached_printed", False):
                    print("\n********************************************")
                    print(f"* Agent successfully reached goal! Distance: {distance:.2f} *")
                    print("********************************************\n")
                    self._goal_reached_printed = True
            else:
                progress = self._compute_goal_progress(leader, distance)
                progress_reward = self.goal_progress_scale * progress
                self._goal_reached_printed = False

        formation_reward = 0.0
        speed_reward = 0.0
        position_reward = 0.0

        if self.entity_manager.followers:
            followers_distances = [leader.distance_to(follower) for follower in self.entity_manager.followers]
            max_formation_error = max((self._formation_error(distance) for distance in followers_distances), default=0.0)
            formation_qualities = [self._formation_quality(distance) for distance in followers_distances]
            mean_formation_quality = float(np.mean(formation_qualities)) if formation_qualities else 0.0
            formation_reward = (
                self.leader_formation_bonus * mean_formation_quality
                - 0.02 * max_formation_error
            )

            for follower in self.entity_manager.followers:
                if leader.distance_to(follower) < 50.0:
                    speed_diff = abs(leader.speed - follower.speed)
                    if speed_diff >= 1.0:
                        speed_reward -= 0.5 * speed_diff

            if self.entity_manager.goals:
                goal = self.entity_manager.goals[0]
                leader_to_goal = leader.distance_to(goal)

                all_correct_position = True
                for follower in self.entity_manager.followers:
                    follower_to_goal = follower.distance_to(goal)
                    if leader_to_goal >= follower_to_goal:
                        all_correct_position = False
                        break

                position_reward = 0.0 if all_correct_position else -0.5

        time_penalty = -1.0

        return (
            edge_reward * self._get_weight("edge")
            + obstacle_reward * self._get_weight("collision")
            + goal_reward * self._get_weight("goal")
            + progress_reward * self._get_weight("progress")
            + formation_reward * self._get_weight("formation")
            + speed_reward * self._get_weight("speed")
            + position_reward * self._get_weight("position")
            + time_penalty * self._get_weight("time")
        )

    def _compute_follower_reward(self, follower):
        """Compute follower reward."""
        if not follower.is_alive():
            return 0.0

        formation_reward = 0.0
        speed_reward_follower = 0.0
        goal_bonus = 0.0
        time_penalty = -0.5

        if self.entity_manager.leaders:
            leader = self.entity_manager.leaders[0]
            distance = follower.distance_to(leader)

            formation_error = self._formation_error(distance)
            formation_quality = self._formation_quality(distance)
            formation_progress = self._compute_formation_progress(follower, formation_error)
            formation_reward = (
                self.follower_formation_bonus * formation_quality
                + self.formation_progress_scale * formation_progress
                - 0.03 * formation_error
            )

            speed_diff_to_leader = abs(leader.speed - follower.speed)
            if distance < 80.0:
                if speed_diff_to_leader >= 1.0:
                    speed_reward_follower = -1.0 * speed_diff_to_leader

            if self.entity_manager.goals and self._leader_has_won(leader):
                goal_bonus = self.terminal_follower_goal_reward * formation_quality

        return (
            formation_reward * self._get_weight("formation")
            + speed_reward_follower * self._get_weight("speed")
            + time_penalty * self._get_weight("time")
            + goal_bonus * self._get_weight("follower_goal")
        )

    def compute_shaped_rewards(self):
        """Compute shaped rewards for curriculum learning.

        Returns:
            Shaped reward array.
        """
        return self.compute_rewards()

    def compute_team_reward(self):
        """Compute overall team reward.

        Returns:
            Team reward value.
        """
        individual_rewards = self.compute_rewards()
        return np.mean(individual_rewards)

    def update_weights(self, new_weights):
        """Update reward weights.

        Args:
            new_weights: New weight dictionary.
        """
        self.weights.update(new_weights)

    def compute_structured_rewards(self):
        """Compute structured rewards.

        Returns:
            rewards: Structured reward dictionary {"leader": reward_leader, "followers": [reward_f1, ...]}.
        """
        leader_reward = 0.0
        if self.entity_manager.leaders:
            leader_reward = self._compute_leader_reward(self.entity_manager.leaders[0])

        follower_rewards = []
        for follower in self.entity_manager.followers:
            follower_rewards.append(self._compute_follower_reward(follower))

        return {
            "leader": leader_reward,
            "followers": follower_rewards,
        }
