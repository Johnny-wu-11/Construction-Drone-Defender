import pygame
import random
import sys
import numpy as np
import os

# 1. basic environment
WIDTH = 400
HEIGHT = 600
FPS = 120  

WHITE = (255, 255, 255)
BLACK = (30, 30, 30)      # background
BLUE = (50, 150, 255)     # color of plan
GRAY = (150, 150, 150)    # calor of falling object
GREEN = (50, 255, 50)     # color of battery
YELLOW = (255, 255, 0)    # color of shield

# size
GRID_SIZE = 40
DRONE_WIDTH = 40
DRONE_HEIGHT = 20
ITEM_SIZE = 40

# 2. Q-Learning
class QLearningAgent:
    def __init__(self, action_size):
        self.action_size = action_size # 4action，left，right，static，shield
        self.q_table = {} 
        
        self.learning_rate = 0.1      # Alpha
        self.discount_factor = 0.95   # Gamma
        self.epsilon = 1.0            
        self.epsilon_decay = 0.995    
        self.epsilon_min = 0.05       

    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0, 0.0] 
        return self.q_table[state][action]

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0, 0.0]

        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1) # random walk
        
        return np.argmax(self.q_table[state]) # choose highest q value

    def learn(self, state, action, reward, next_state, done):
        """核心公式更新 Q 表"""
        current_q = self.get_q_value(state, action)
        
        if done:
            max_future_q = 0 # game over
        else:
            if next_state not in self.q_table:
                self.q_table[next_state] = [0.0, 0.0, 0.0, 0.0]
            max_future_q = np.max(self.q_table[next_state])
        
        # New Q value = Old Q value + Learning rate * (Immediate reward + Discount factor * Maximum future expectation - Old Q value)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[state][action] = new_q
        
        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# game environment
class ConstructionDroneGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("High-rise construction site - Intelligent drone training")
        self.clock = pygame.time.Clock()
        
        # UI
        try:
            current_dir = os.path.dirname(__file__)
            
            bg_path = os.path.join(current_dir, 'background.jpg')
            drone_path = os.path.join(current_dir, 'drone.png')
            steel_path = os.path.join(current_dir, 'steel.png')
            battery_path = os.path.join(current_dir, 'battery.png')

            self.bg_img = pygame.transform.scale(pygame.image.load(bg_path), (WIDTH, HEIGHT))
            self.drone_img = pygame.transform.scale(pygame.image.load(drone_path), (DRONE_WIDTH, DRONE_HEIGHT))
            self.steel_img = pygame.transform.scale(pygame.image.load(steel_path), (ITEM_SIZE, ITEM_SIZE))
            self.battery_img = pygame.transform.scale(pygame.image.load(battery_path), (ITEM_SIZE, ITEM_SIZE))
            
            self.images_loaded = True
            print("success")
        except FileNotFoundError as e:
            print(f"Warning: Unable to find image materials! Error details: {e}")
            self.images_loaded = False
            
        self.reset()

    def reset(self):
        """reset status"""
        self.drone_x = WIDTH // 2 - DRONE_WIDTH // 2
        self.drone_y = HEIGHT - 50
        self.drone_speed = GRID_SIZE 
        
        self.items = [] # List of items that have fallen and need to be stored
        self.score = 0
        self.game_over = False
        
        # Shield-related variables
        self.shield_timer = 0  # Shield remaining frames
        self.shield_duration = 5 # Using one shield once can last for 5 frames.
        
        return self.get_state()

    def get_state(self):
        """
        The AI only focuses on the item that is closest to it (the one with the highest Y coordinate).
        """
        lowest_item = None
        max_y = -100
        for item in self.items:
            if item['rect'].y > max_y:
                max_y = item['rect'].y
                lowest_item = item

        # shield status：1open 0close
        shield_active = 1 if self.shield_timer > 0 else 0

        if lowest_item:
            item_type_id = 1 if lowest_item['type'] == 'steel' else 2
            return (self.drone_x // GRID_SIZE, item_type_id, lowest_item['rect'].x // GRID_SIZE, lowest_item['rect'].y // GRID_SIZE, shield_active)
        else:
            # status of nothing on the scream
            return (self.drone_x // GRID_SIZE, 0, 0, 0, shield_active)

    def drop_item(self):
        if len(self.items) < 2: 
            grid_x = random.randint(0, (WIDTH // GRID_SIZE) - 1) * GRID_SIZE
            # 70% lose the steel bars (dangerous), 30% lose the batteries (reward)
            item_type = 'steel' if random.random() < 0.7 else 'battery'
            new_item = {
                'rect': pygame.Rect(grid_x, -ITEM_SIZE, ITEM_SIZE, ITEM_SIZE),
                'type': item_type
            }
            self.items.append(new_item)

    def step(self, action):
        """
        Perform the action and return: the next state, the reward, and whether it is the end.
        """
        reward = 1  # Basic survival bonus: 1 point awarded for being alive.
        
        # 1. deal with action
        if action == 0:   # left
            self.drone_x -= self.drone_speed
        elif action == 1: # right
            self.drone_x += self.drone_speed
        elif action == 3: # shield
            if self.shield_timer == 0: # if not open then open
                self.shield_timer = self.shield_duration
        
        # Border control penalties
        if self.drone_x < 0:
            self.drone_x = 0
            reward -= 5
        elif self.drone_x > WIDTH - DRONE_WIDTH:
            self.drone_x = WIDTH - DRONE_WIDTH
            reward -= 5

        # 2. Update shield status and deduction
        if self.shield_timer > 0:
            self.shield_timer -= 1
            reward -= 2 # Opening the shield will consume energy. 2 points will be deducted per frame.

        # 3. Item falling down
        for item in self.items:
            item['rect'].y += 20 
            
        for item in self.items[:]:
            if item['rect'].y > HEIGHT:
                self.items.remove(item)
                if item['type'] == 'steel':
                    self.score += 1
                    reward += 5 # Successfully avoided the steel bars and received the reward.
                elif item['type'] == 'battery':
                    reward -= 10 # The battery was not plugged in, resulting in a deduction.

        # 4. collision detection
        drone_rect = pygame.Rect(self.drone_x, self.drone_y, DRONE_WIDTH, DRONE_HEIGHT)
        for item in self.items[:]:
            if drone_rect.colliderect(item['rect']):
                if item['type'] == 'battery':
                    # get the battery
                    self.score += 5
                    reward += 30 # high reward
                    self.items.remove(item)
                elif item['type'] == 'steel':
                    # crush
                    if self.shield_timer > 0:
                        # opening the shield：not die and crush the Fallen objects
                        self.score += 2
                        reward += 20 
                        self.items.remove(item)
                    else:
                        # die
                        self.game_over = True
                        reward -= 100 

        next_state = self.get_state()
        return next_state, reward, self.game_over

    def draw(self, episode, ai_epsilon):
        # 1. background
        if getattr(self, 'images_loaded', False):
            self.screen.blit(self.bg_img, (0, 0))
        else:
            self.screen.fill(BLACK)
        
        if not self.game_over:
            # 2. UAV
            if getattr(self, 'images_loaded', False):
                self.screen.blit(self.drone_img, (self.drone_x, self.drone_y))
            else:
                pygame.draw.rect(self.screen, BLUE, (self.drone_x, self.drone_y, DRONE_WIDTH, DRONE_HEIGHT))
            
            # 3. shield
            if self.shield_timer > 0:
                shield_rect = (self.drone_x - 10, self.drone_y - 10, DRONE_WIDTH + 20, DRONE_HEIGHT + 20)
                pygame.draw.ellipse(self.screen, YELLOW, shield_rect, 3) 

            # 4. falling object
            for item in self.items:
                if getattr(self, 'images_loaded', False):
                    if item['type'] == 'steel':
                        self.screen.blit(self.steel_img, item['rect'])
                    else:
                        self.screen.blit(self.battery_img, item['rect'])
                else:
                    color = GRAY if item['type'] == 'steel' else GREEN
                    pygame.draw.rect(self.screen, color, item['rect'])
            
            # 5. UI HUD
            font = pygame.font.SysFont(None, 24)
            info_bg = pygame.Surface((WIDTH, 35))
            info_bg.set_alpha(150) 
            info_bg.fill((0, 0, 0))
            self.screen.blit(info_bg, (0, 0))
            
            score_text = font.render(f"Score: {self.score} | Shield: {'ON' if self.shield_timer > 0 else 'OFF'}", True, WHITE)
            ai_text = font.render(f"Ep: {episode} | Explore: {ai_epsilon:.2f}", True, YELLOW)
            
            self.screen.blit(score_text, (10, 10))
            self.screen.blit(ai_text, (200, 10))
            
        pygame.display.flip()



if __name__ == "__main__":
    env = ConstructionDroneGame()
    agent = QLearningAgent(action_size=4) # left，right，no move，shield
    
    episodes = 500 
    e = 1
    #for e in range(episodes):
    while 1:
        state = env.reset() 
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            # random falling
            if random.randint(1, 100) < 30: 
                env.drop_item()

            # AI choose action
            action = agent.choose_action(state)
            
            # The environment performs an action and provides feedback.
            next_state, reward, done = env.step(action)
            
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            
            env.draw(e, agent.epsilon)
            env.clock.tick(FPS)
        
            if done:
                print(f"round: {e}, score: {env.score}, running randomly Probability: {agent.epsilon:.2f}")
                e += 1
                break  