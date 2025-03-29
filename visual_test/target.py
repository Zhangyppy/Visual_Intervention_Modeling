import numpy as np

# Constants from environment
LAYER_ENVIRONMENT = 0
LAYER_NOTIFICATION = 1

# Default sizes
ENVIRONMENT_BLOCK_SIZE = 20
NOTIFICATION_BLOCK_SIZE = 40

# Colors (RGB)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Textures
BLUE_TEXTURE = (70, 70, 220)
GREEN_TEXTURE = (0, 100, 0)

class Target:
    """
    Represents a target in the simulation environment.
    
    Attributes:
        pos (numpy.ndarray): Position [x, y] of the target
        layer (int): Layer level (LAYER_ENVIRONMENT or LAYER_NOTIFICATION)
        color (tuple): RGB color tuple
        size (int): Size of the target block
        required_steps (int): Steps agent needs on target to complete (1 for immediate completion)
        current_steps (int): Current steps agent has spent on target
        completed (bool): Whether the target has been completed
    """
    
    def __init__(self, 
                 pos, 
                 layer=None, 
                 color=None, 
                 size=None, 
                 required_steps=1):
        """
        Initialize a target.
        
        Args:
            pos (list or numpy.ndarray): Position [x, y] of the target
            layer (int, optional): Layer level. If None, randomly chosen.
            color (tuple, optional): RGB color tuple. If None, default color based on layer.
            size (int, optional): Size of target. If None, based on layer.
            required_steps (int): Steps agent needs on target to complete. 
                                 Set to 1 for immediate completion (equivalent to "infinite" duration)
        """
        self.pos = np.array(pos, dtype=np.float32)
        self.layer = layer if layer is not None else np.random.randint(0, 2)
        if size is None:
            self.size = ENVIRONMENT_BLOCK_SIZE if self.layer == LAYER_ENVIRONMENT else NOTIFICATION_BLOCK_SIZE
        else:
            self.size = size
            
        if color is None:
            self.color = GREEN if self.layer == LAYER_ENVIRONMENT else BLUE
        else:
            self.color = color
            
        self.texture = GREEN_TEXTURE if self.layer == LAYER_ENVIRONMENT else BLUE_TEXTURE
        self.required_steps = max(1, required_steps)  # Ensure at least 1 step is required
        self.current_steps = 0
        self.completed = False
        
    def get_boundaries(self):
        """
        Calculate the target boundaries.
        
        Returns:
            tuple: (x_min, y_min, x_max, y_max) of target boundaries
        """
        x_min = int(self.pos[0] - self.size // 2)
        y_min = int(self.pos[1] - self.size // 2)
        x_max = x_min + self.size
        y_max = y_min + self.size
        return x_min, y_min, x_max, y_max
    
    def check_collision(self, agent_pos, agent_radius):
        """
        Check if the agent collides with the target.
        
        Args:
            agent_pos (numpy.ndarray): Position of the agent
            agent_radius (int): Radius of the agent
            
        Returns:
            bool: True if collision, False otherwise
        """
        # Calculate the distance between the agent's center and the target's center
        distance = np.linalg.norm(agent_pos - self.pos)
        return distance < (agent_radius + self.size / 2)
        
    def update_progress(self, agent_on_target, agent_on_correct_layer):
        """
        Update progress toward completion if agent is on target.
        
        Args:
            agent_on_target (bool): Whether agent is physically on the target
            agent_on_correct_layer (bool): Whether agent is on the same layer as target
            
        Returns:
            bool: True if target is completed, False otherwise
        """
        if agent_on_target and agent_on_correct_layer:
            self.current_steps += 1
            if self.current_steps >= self.required_steps:
                self.completed = True
        else:
            # Optional: Reset progress if agent leaves target
            # TODO: review this based on our task
            # self.current_steps = 0
            pass
                
        return self.completed
    
    def is_immediate(self):
        """Check if this target completes immediately (1 required step)."""
        return self.required_steps == 1
    
    def reset(self):
        """Reset the target's progress."""
        self.current_steps = 0
        self.completed = False 