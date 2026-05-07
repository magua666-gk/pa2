import random
import pygame
import numpy as np
from rl_env.components.entities import LeaderAgent, FollowerAgent, Obstacle, Goal, Constants
from rl_env.components.position_generator import PositionGenerator

class EntityManager:
    """Manage all entities in the environment"""
    
    def __init__(self, leader_count=1, follower_count=1, obstacle_count=1, goal_count=1, predefined_positions=None):
        """Initialize entity manager
        
        Args:
            leader_count: Number of leaders
            follower_count: Number of followers
            obstacle_count: Number of obstacles
            goal_count: Number of goals
            predefined_positions: Predefined position dictionary, format:
                {
                    'leaders': [(x1,y1), (x2,y2), ...],
                    'followers': [(x1,y1), (x2,y2), ...],
                    'obstacles': [(x1,y1), (x2,y2), ...],
                    'goals': [(x1,y1), (x2,y2), ...]
                }
        """
        self.leader_count = leader_count
        self.follower_count = follower_count
        self.obstacle_count = obstacle_count
        self.goal_count = goal_count
        self.total_agents = leader_count + follower_count
        
        # Save predefined positions
        self.predefined_positions = predefined_positions
        
        # Create entity containers
        self.leaders = []
        self.followers = []
        self.obstacles = []
        self.goals = []
        
        # State flags
        self.done = False
        self.team_counter = 0
        self.time_counter = 0
        self.images_loaded = False 
        self.goal_reach_radius = 40.0
        
        # Create entities
        self._create_entities()
        print("<<<<< EntityManager VERSION XYZ RUNNING >>>>>") 
    
    def _create_entities(self):
        """Create all entities"""
        # ---- BEGIN DEBUG PRINT ----
        print(f"[EM._create_entities] Creating entities with: leader_count={self.leader_count}, follower_count={self.follower_count}")
        # ---- END DEBUG PRINT ----
        
        # Clear existing entities
        self.leaders = []
        self.followers = []
        self.obstacles = []
        self.goals = []
        
        # Create leaders
        for i in range(self.leader_count):
            leader = LeaderAgent()
            self.leaders.append(leader)
        
        # Create followers
        for i in range(self.follower_count):
            follower = FollowerAgent()
            self.followers.append(follower)
        
        # Create obstacles
        for i in range(self.obstacle_count):
            obstacle = Obstacle()
            self.obstacles.append(obstacle)
        
        # Create goals
        for i in range(self.goal_count):
            goal = Goal()
            self.goals.append(goal)


        # Note: Image loading will be handled by Renderer, after environment initialization
        # Don't load images here to avoid image_dict undefined error
            
        # Randomly place entities
        self._randomize_positions()
    
    def _randomize_positions(self):
        """Randomly place entity initial positions"""
        # Use PositionGenerator to generate all positions
        positions = PositionGenerator.generate_all_positions(
            leader_count=self.leader_count,
            follower_count=self.follower_count,
            obstacle_count=self.obstacle_count,
            goal_count=self.goal_count
        )
        
        # If there are predefined positions, override corresponding positions
        if self.predefined_positions is not None:
            for entity_type, pos_list in self.predefined_positions.items():
                if entity_type in positions and pos_list:
                    positions[entity_type] = pos_list
        
        # Set leader positions
        for i, leader in enumerate(self.leaders):
            if i < len(positions['leaders']):
                leader.set_position(*positions['leaders'][i])
        
        # Set follower positions
        for i, follower in enumerate(self.followers):
            if i < len(positions['followers']):
                follower.set_position(*positions['followers'][i])
        
        # Set obstacle positions
        for i, obstacle in enumerate(self.obstacles):
            if i < len(positions['obstacles']):
                obstacle.set_position(*positions['obstacles'][i])
        
        # Set goal positions
        for i, goal in enumerate(self.goals):
            if i < len(positions['goals']):
                goal.set_position(*positions['goals'][i])
    
    def apply_actions(self, leader_action, follower_actions):
        """Apply structured actions to entities
        
        Args:
            leader_action: Leader action array
            follower_actions: Followers action list
        """
        # Apply Leader action
        if len(self.leaders) > 0:
            self.leaders[0].apply_action(leader_action)
        
        # Apply Followers actions
        for i, action in enumerate(follower_actions):
            if i < len(self.followers):
                self.followers[i].apply_action(action)
    
    def update(self):
        """Update all entity states"""
        # Update leaders
        for leader in self.leaders:
            leader.update()
            # Rotate image to match heading
            leader.rotate()
        
        # Update followers
        for follower in self.followers:
            follower.update()
            # Rotate image to match heading
            follower.rotate()
        
        # Check collisions and goal achievement
        self._check_collisions()
        self._check_goals()
        
        # Check formation status
        self._check_formation()
        
        # Update counters
        self.time_counter += 1
    
    def _check_collisions(self):
        """Check collisions"""
        for leader in self.leaders:
            if not leader.is_alive():
                continue
                
            # Check collisions with obstacles
            for obstacle in self.obstacles:
                if leader.distance_to(obstacle) < 20.0:
                    leader.kill(won=False)
                    self.done = True
                    return
            
            # Boundary collision detection removed
            # Position constraints implemented by np.clip in Agent.update() method
    
    def _check_goals(self):
        """Check goal achievement"""
        for leader in self.leaders:
            if not leader.is_alive():
                continue
                
            for goal in self.goals:
                if leader.distance_to(goal) < self.goal_reach_radius:
                    leader.kill(won=True)
                    self.done = True
                    return
    
    def _check_formation(self):
        """Check formation maintenance status"""
        if not self.leaders or not self.followers:
            return
        
        # Check if all followers maintain formation with leader
        all_in_formation = True
        leader = self.leaders[0]  # Assume first leader is the leader
        
        # Only keep distance check, remove position relationship condition
        for follower in self.followers:
            # Calculate follower to leader distance
            distance = follower.distance_to(leader)
            
            # Formation condition: only check distance < 50
            if distance >= 50:
                all_in_formation = False
                break
        
        # Only increase counter when all followers meet the condition
        if all_in_formation:
            self.team_counter += 1
    
    def reset(self):
        """Reset entity manager state"""
        # Reset entities
        self._create_entities()
        
        # Reset state
        self.done = False
        self.team_counter = 0
        self.time_counter = 0
        self.images_loaded = False  # Reset image loading status
    
    def is_episode_done(self):
        """Check if episode is done
        
        Returns:
            Whether done
        """
        return self.done
    
    def is_hero_win(self):
        """Check if leader wins
        
        Returns:
            Whether wins
        """
        if not self.leaders:
            return False
        
        return any(leader.has_won for leader in self.leaders)
    
    def get_formation_rate(self):
        """Get formation maintenance rate
        
        Returns:
            Formation maintenance rate
        """
        if self.time_counter == 0:
            return 0.0
        
        return self.team_counter / self.time_counter
    
    def get_agent_distances(self):
        """Get distance matrix between agents
        
        Returns:
            Distance dictionary, keys are agent pair names (e.g., "leader_to_goal"), values are corresponding distances
        """
        distances = {}
        
        # Leader to goal distance
        if self.leaders and self.goals:
            leader = self.leaders[0]
            goal = self.goals[0]
            distances['leader_to_goal'] = leader.distance_to(goal)
        
        # Leader to follower distance
        if self.leaders and self.followers:
            leader = self.leaders[0]
            for i, follower in enumerate(self.followers):
                distances[f"leader_to_follower_{i}"] = leader.distance_to(follower)
        
        # Leader to obstacle minimum distance
        if self.leaders and self.obstacles:
            leader = self.leaders[0]
            min_obstacle_dist = min([leader.distance_to(obs) for obs in self.obstacles], default=float('inf'))
            distances["leader_to_obstacle"] = min_obstacle_dist
        
        return distances
    
    def reconfigure(self, leader_count, follower_count, obstacle_count, goal_count=1):
        """Reconfigure entity counts
        
        Args:
            leader_count: New leader count
            follower_count: New follower count
            obstacle_count: New obstacle count
            goal_count: New goal count
        """
        # Update counts
        self.leader_count = leader_count
        self.follower_count = follower_count
        self.obstacle_count = obstacle_count
        self.goal_count = goal_count
        self.total_agents = leader_count + follower_count
        
        # Recreate entities
        self._create_entities()
        
        # Reset state
        self.done = False
        self.team_counter = 0
        self.time_counter = 0
    
    def render(self, screen):
        """Render all entities
        
        Args:
            screen: pygame screen object
        """
        # Render obstacles
        for obstacle in self.obstacles:
            obstacle.render(screen)
        
        # Render goals
        for goal in self.goals:
            goal.render(screen)
        
        # Render followers
        for follower in self.followers:
            follower.render(screen)
        
        # Render leaders
        for leader in self.leaders:
            leader.render(screen)
    
    def load_images(self, image_dict):
        """Load entity images
        
        Args:
            image_dict: Image dictionary, format: {"leader": img, "follower": img, "obstacle": img, "goal": img}
        """
        # Load leader images
        if "leader" in image_dict and self.leaders:
            for leader in self.leaders:
                leader.load_image(image_dict["leader"], (30, 30))
        
        # Load follower images
        if "follower" in image_dict and self.followers:
            for follower in self.followers:
                follower.load_image(image_dict["follower"], (30, 30))
        
        # Load obstacle images
        if "obstacle" in image_dict and self.obstacles:
            for obstacle in self.obstacles:
                obstacle.load_image(image_dict["obstacle"], (40, 40))
        
        # Load goal images
        if "goal" in image_dict and self.goals:
            for goal in self.goals:
                goal.load_image(image_dict["goal"], (30, 30)) 