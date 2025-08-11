# ---------------main.py---------------------
import pygame
import os
import math
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from agent import DqnGPU  # Import the GPU-accelerated agent

# Initialize pygame
pygame.init()


# Screen dimensions - will be set based on fullscreen
SCREEN_WIDTH = 0
SCREEN_HEIGHT = 0
FULLSCREEN = True  # Set to False for windowed mode

# Set up display
if FULLSCREEN:
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    SCREEN_WIDTH, SCREEN_HEIGHT = screen.get_size()
else:
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

pygame.display.set_caption("Autonomous Car - GPU DQN")

# Clock for controlling FPS
clock = pygame.time.Clock()
FPS = 60

# Modern UI Color Palette
DARK_BLUE = (25, 35, 60)
LIGHT_BLUE = (70, 130, 180)
ACCENT_BLUE = (0, 150, 255)
DARK_GRAY = (40, 45, 60)
LIGHT_GRAY = (200, 200, 210)
WHITE = (240, 245, 255)
BLACK = (15, 20, 30)
RED = (255, 80, 80)
GREEN = (80, 255, 150)
YELLOW = (255, 230, 80)
CYAN = (80, 220, 255)
PURPLE = (180, 100, 255)

# Panel and button colors
PANEL_COLOR = (*DARK_BLUE, 220)
BUTTON_COLOR = LIGHT_BLUE
BUTTON_HOVER_COLOR = ACCENT_BLUE
BUTTON_ACTIVE_COLOR = (50, 100, 180)

# Font
font = pygame.font.SysFont("Arial", 32)
small_font = pygame.font.SysFont("Arial", 22)
button_font = pygame.font.SysFont("Arial", 26)
tiny_font = pygame.font.SysFont("Arial", 18)

# Track management
tracks = []
overlays = []
current_track_index = 0


# Load available tracks
def load_tracks():
    tracks_dir = "tracks"
    if not os.path.exists(tracks_dir):
        os.makedirs(tracks_dir)
        print(f"Created tracks directory at {tracks_dir}")

    for file in os.listdir(tracks_dir):
        if "overlay" in file:
            overlays.append(os.path.join(tracks_dir, file))
        elif file.endswith(".png"):
            tracks.append(os.path.join(tracks_dir, file))

    # Match tracks with their overlays
    track_overlay_pairs = []
    for track in tracks:
        base_name = os.path.basename(track).split(".")[0]
        overlay = None
        for ov in overlays:
            if base_name + "-overlay" in ov:
                overlay = ov
                break
        if overlay:
            track_overlay_pairs.append((track, overlay))

    return track_overlay_pairs


# Create sand representation from track overlay
def create_sand(track_overlay_path):
    img = PILImage.open(track_overlay_path).convert("L")
    img = img.resize((SCREEN_WIDTH, SCREEN_HEIGHT), PILImage.Resampling.LANCZOS)
    img_array = np.array(img)
    return (img_array < 128).astype(
        int
    )  # 1 for off-track (white), 0 for on-track (black)


# Button class for UI
class Button:
    def __init__(self, x, y, width, height, text, action=None, key_hint=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.hovered = False
        self.active = False
        self.key_hint = key_hint

    def draw(self, surface):
        # Button background
        if self.active:
            color = BUTTON_ACTIVE_COLOR
        elif self.hovered:
            color = BUTTON_HOVER_COLOR
        else:
            color = BUTTON_COLOR

        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, WHITE, self.rect, 2, border_radius=6)

        # Button text
        text_surf = button_font.render(self.text, True, WHITE)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

        # Key hint
        if self.key_hint:
            hint_surf = small_font.render(self.key_hint, True, YELLOW)
            hint_rect = hint_surf.get_rect(
                midtop=(self.rect.centerx, self.rect.bottom + 5)
            )
            surface.blit(hint_surf, hint_rect)

    def check_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hovered and self.action:
                self.active = True
                return self.action()
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.active = False
        return False


# Car class with radar visualization
class Car:
    def __init__(self, x=480, y=270):
        self.original_image = pygame.image.load("car.png").convert_alpha()
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(x, y))
        self.angle = 0
        self.velocity = pygame.math.Vector2(0.8, 0)
        self.rotation_vel = 5
        self.direction = 0
        self.alive = True
        self.radars = []
        self.radar_endpoints = []  # Store radar endpoints for drawing
        self.collision_point_right = (0, 0)
        self.collision_point_left = (0, 0)
        self.speed = 3.0
        self.max_speed = 6.0
        self.acceleration = 0.1
        self.slip_factor = 0.95  # Simulate wheel slip
        self.off_track_timer = 0
        self.max_off_track_time = 60  # Frames before reset (1 second)
        self.total_distance = 0
        self.last_position = (x, y)

    def reset(self, x=480, y=270):
        self.rect.center = (x, y)
        self.angle = 0
        self.velocity = pygame.math.Vector2(0.8, 0)
        self.direction = 0
        self.alive = True
        self.radars.clear()
        self.radar_endpoints.clear()
        self.speed = 3.0
        self.off_track_timer = 0
        self.total_distance = 0
        self.last_position = (x, y)

    def update(self, sand):
        self.radars.clear()
        self.radar_endpoints.clear()

        # Calculate distance traveled
        current_pos = self.rect.center
        distance = math.sqrt(
            (current_pos[0] - self.last_position[0]) ** 2
            + (current_pos[1] - self.last_position[1]) ** 2
        )
        self.total_distance += distance
        self.last_position = current_pos

        self.drive()
        self.rotate()

        # Update radars
        for radar_angle in (-75, -45, -15, 15, 45, 75):
            self.radar(radar_angle, sand)

        self.collision(sand)

    def drive(self):
        # Apply direction to rotation
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.velocity.rotate_ip(self.rotation_vel)
        elif self.direction == -1:
            self.angle += self.rotation_vel
            self.velocity.rotate_ip(-self.rotation_vel)

        # Update position with current velocity
        self.rect.center += self.velocity * self.speed

        # Update image rotation
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def rotate(self):
        # Smooth rotation
        if self.direction != 0:
            rotation = self.rotation_vel * self.direction
            self.angle = (self.angle + rotation) % 360

    def radar(self, radar_angle, sand):
        length = 0
        x, y = int(self.rect.center[0]), int(self.rect.center[1])
        end_point = (0, 0)

        # Calculate radar line
        while length < 250:
            x = int(
                self.rect.center[0]
                + math.cos(math.radians(self.angle + radar_angle)) * length
            )
            y = int(
                self.rect.center[1]
                - math.sin(math.radians(self.angle + radar_angle)) * length
            )

            # Check boundaries
            if 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT:
                # Check if point is off-track (white space)
                if sand[y, x] == 1:  # 1 indicates off-track (white)
                    break
            else:
                break

            length += 1
            end_point = (x, y)

        # Store radar endpoint for drawing
        self.radar_endpoints.append(end_point)

        # Calculate distance
        dist = math.sqrt(
            (self.rect.center[0] - x) ** 2 + (self.rect.center[1] - y) ** 2
        )
        self.radars.append([radar_angle, dist])

    def collision(self, sand):
        # Check car's center position
        center_x = int(self.rect.center[0])
        center_y = int(self.rect.center[1])

        # Check if center is off-track
        if 0 <= center_x < SCREEN_WIDTH and 0 <= center_y < SCREEN_HEIGHT:
            if sand[center_y, center_x] == 1:  # Off-track (white)
                self.off_track_timer += 1
            else:
                self.off_track_timer = max(0, self.off_track_timer - 2)
        else:
            self.off_track_timer += 1

        # Check if car has been off-track for too long
        if self.off_track_timer >= self.max_off_track_time:
            self.alive = False

        # Additional collision points
        length = 25
        self.collision_point_right = [
            int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
            int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length),
        ]
        self.collision_point_left = [
            int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
            int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length),
        ]

        # Check collision points with off-track areas
        try:
            # Check right collision point
            if (
                0 <= self.collision_point_right[0] < SCREEN_WIDTH
                and 0 <= self.collision_point_right[1] < SCREEN_HEIGHT
            ):
                if (
                    sand[
                        int(self.collision_point_right[1]),
                        int(self.collision_point_right[0]),
                    ]
                    == 1
                ):
                    self.alive = False

            # Check left collision point
            if (
                0 <= self.collision_point_left[0] < SCREEN_WIDTH
                and 0 <= self.collision_point_left[1] < SCREEN_HEIGHT
            ):
                if (
                    sand[
                        int(self.collision_point_left[1]),
                        int(self.collision_point_left[0]),
                    ]
                    == 1
                ):
                    self.alive = False
        except IndexError:
            self.alive = False

    def draw_radars(self, surface):
        # Draw radars
        for end_point in self.radar_endpoints:
            pygame.draw.line(surface, RED, self.rect.center, end_point, 1)
            pygame.draw.circle(surface, GREEN, end_point, 3)

        # Draw collision points
        pygame.draw.circle(surface, CYAN, self.collision_point_right, 3)
        pygame.draw.circle(surface, CYAN, self.collision_point_left, 3)

        # Draw off-track timer bar
        if self.off_track_timer > 0:
            bar_width = 40
            bar_height = 5
            bar_x = self.rect.centerx - bar_width // 2
            bar_y = self.rect.top - 15
            fill_width = (self.off_track_timer / self.max_off_track_time) * bar_width

            pygame.draw.rect(
                surface, DARK_GRAY, (bar_x, bar_y, bar_width, bar_height), 1
            )
            pygame.draw.rect(surface, RED, (bar_x, bar_y, fill_width, bar_height))

    def get_state(self):
        # Normalize radar distances to [0, 1]
        state = [min(radar[1] / 250, 1.0) for radar in self.radars]
        # Add speed and off-track timer as features
        state.append(self.speed / self.max_speed)
        state.append(self.off_track_timer / self.max_off_track_time)
        return state


# Main game class
class Game:
    def __init__(self):
        self.track_pairs = load_tracks()
        if not self.track_pairs:
            print("No tracks found! Please add tracks to the tracks directory.")
            pygame.quit()
            sys.exit()

        self.current_track_index = 0
        self.track_path, self.overlay_path = self.track_pairs[self.current_track_index]
        self.track_img = pygame.image.load(self.track_path).convert()
        self.track_img = pygame.transform.scale(
            self.track_img, (SCREEN_WIDTH, SCREEN_HEIGHT)
        )

        # Load overlay image
        self.overlay_img = pygame.image.load(self.overlay_path).convert_alpha()
        self.overlay_img = pygame.transform.scale(
            self.overlay_img, (SCREEN_WIDTH, SCREEN_HEIGHT)
        )

        self.sand = create_sand(self.overlay_path)

        self.car = Car(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

        # Initialize GPU-accelerated DQN agent
        self.agent = DqnGPU(
            input_size=8,  # 6 radars + speed + off-track timer
            nb_action=3,  # straight, left, right
            gamma=0.99,  # Discount factor
            lr=0.0001,  # Learning rate
            memory_capacity=100000,
        )

        self.score = 0
        self.episode = 0
        self.max_score = 0
        self.scores = []
        self.episode_rewards = []
        self.running = True
        self.paused = False
        self.training = True
        self.last_save_time = pygame.time.get_ticks()
        self.save_interval = 300000  # 5 minutes in milliseconds
        self.last_reward = 0
        self.step_count = 0

        # Load existing model if available
        self.agent.load("last_brain_gpu.pth")

        # Create UI buttons
        button_width = 200
        button_height = 45
        button_margin = 12
        button_y = 20
        panel_height = 120

        # Create semi-transparent panel for buttons
        self.button_panel = pygame.Surface(
            (SCREEN_WIDTH, panel_height), pygame.SRCALPHA
        )
        self.button_panel.fill(PANEL_COLOR)

        # Create buttons with key hints
        self.buttons = [
            Button(
                20,
                button_y,
                button_width,
                button_height,
                "Save Model",
                self.save_model,
                "(S)",
            ),
            Button(
                20 + button_width + button_margin,
                button_y,
                button_width,
                button_height,
                "Load Model",
                self.load_model,
                "(L)",
            ),
            Button(
                20 + 2 * (button_width + button_margin),
                button_y,
                button_width,
                button_height,
                "Plot Progress",
                self.plot_progress,
                "(P)",
            ),
            Button(
                20 + 3 * (button_width + button_margin),
                button_y,
                button_width,
                button_height,
                "Next Track",
                self.next_track,
                "(N)",
            ),
            Button(
                20 + 4 * (button_width + button_margin),
                button_y,
                button_width,
                button_height,
                "Pause Train" if self.training else "Resume Train",
                self.toggle_training,
                "(T)",
            ),
            Button(
                20 + 5 * (button_width + button_margin),
                button_y,
                button_width,
                button_height,
                "Fullscreen" if not FULLSCREEN else "Windowed",
                self.toggle_fullscreen,
                "(F)",
            ),
            Button(
                20 + 6 * (button_width + button_margin),
                button_y,
                button_width,
                button_height,
                "Exit",
                self.exit_game,
                "(ESC)",
            ),
        ]

        # Enhanced stats panel for GPU information
        self.stats_panel = pygame.Surface((350, 220), pygame.SRCALPHA)
        self.stats_panel.fill(PANEL_COLOR)
        self.stats_panel_rect = self.stats_panel.get_rect(
            topright=(SCREEN_WIDTH - 20, 20)
        )

    def toggle_fullscreen(self):
        global FULLSCREEN, screen, SCREEN_WIDTH, SCREEN_HEIGHT

        # Toggle fullscreen state
        FULLSCREEN = not FULLSCREEN

        # Recreate the screen
        if FULLSCREEN:
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            SCREEN_WIDTH, SCREEN_HEIGHT = screen.get_size()
            self.buttons[5].text = "Windowed"
        else:
            screen = pygame.display.set_mode((1920, 1080))
            SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
            self.buttons[5].text = "Fullscreen"

        # Update UI elements
        self.stats_panel_rect.topright = (SCREEN_WIDTH - 20, 20)

        # Reload track images to fit new size
        self.track_img = pygame.image.load(self.track_path).convert()
        self.track_img = pygame.transform.scale(
            self.track_img, (SCREEN_WIDTH, SCREEN_HEIGHT)
        )
        self.overlay_img = pygame.image.load(self.overlay_path).convert_alpha()
        self.overlay_img = pygame.transform.scale(
            self.overlay_img, (SCREEN_WIDTH, SCREEN_HEIGHT)
        )
        self.sand = create_sand(self.overlay_path)

        # Reset car position
        self.car.reset(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

        print(f"Switched to {'fullscreen' if FULLSCREEN else 'windowed'} mode")
        return True

    def exit_game(self):
        # Save model before exit
        self.agent.save("last_brain_gpu.pth")
        self.running = False
        return True

    def next_track(self):
        self.current_track_index = (self.current_track_index + 1) % len(
            self.track_pairs
        )
        self.track_path, self.overlay_path = self.track_pairs[self.current_track_index]
        self.track_img = pygame.image.load(self.track_path).convert()
        self.track_img = pygame.transform.scale(
            self.track_img, (SCREEN_WIDTH, SCREEN_HEIGHT)
        )

        # Update overlay image
        self.overlay_img = pygame.image.load(self.overlay_path).convert_alpha()
        self.overlay_img = pygame.transform.scale(
            self.overlay_img, (SCREEN_WIDTH, SCREEN_HEIGHT)
        )

        self.sand = create_sand(self.overlay_path)
        self.car.reset(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        print(f"Switched to track: {os.path.basename(self.track_path)}")
        return True

    def save_model(self):
        self.agent.save("manual_save_gpu.pth")
        print("Model manually saved!")
        return True

    def load_model(self):
        self.agent.load("last_brain_gpu.pth")
        print("Model loaded!")
        return True

    def plot_progress(self):
        if self.scores:
            plt.figure(figsize=(15, 10))

            # Plot scores
            plt.subplot(2, 2, 1)
            plt.plot(self.scores, color="blue", alpha=0.7)
            plt.title(f"Episode Scores - Current Episode {self.episode}")
            plt.xlabel("Episode")
            plt.ylabel("Score")
            plt.grid(True, alpha=0.3)

            # Plot moving average
            if len(self.scores) > 10:
                window = min(100, len(self.scores) // 10)
                moving_avg = []
                for i in range(len(self.scores)):
                    start = max(0, i - window + 1)
                    moving_avg.append(np.mean(self.scores[start : i + 1]))
                plt.plot(
                    moving_avg, color="red", linewidth=2, label=f"Moving Avg ({window})"
                )
                plt.legend()

            # Plot agent statistics
            stats = self.agent.get_stats()

            plt.subplot(2, 2, 2)
            if self.agent.losses:
                recent_losses = self.agent.losses[-1000:]  # Last 1000 losses
                plt.plot(recent_losses, color="orange", alpha=0.7)
                plt.title("Training Loss (Recent)")
                plt.xlabel("Training Step")
                plt.ylabel("Loss")
                plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 3)
            epsilon_history = []
            for step in range(0, self.agent.steps, max(1, self.agent.steps // 100)):
                epsilon = max(0.01, 0.5 * (0.995 ** (step // 500)))
                epsilon_history.append(epsilon)
            if epsilon_history:
                plt.plot(epsilon_history, color="green")
                plt.title("Epsilon Decay")
                plt.xlabel("Steps (scaled)")
                plt.ylabel("Epsilon")
                plt.grid(True, alpha=0.3)

            plt.subplot(2, 2, 4)
            info_text = f"""
            Agent Statistics:
            Steps: {stats['steps']:,}
            Current Score: {stats['current_score']:.3f}
            Epsilon: {stats['epsilon']:.4f}
            Memory Size: {stats['memory_size']:,}
            Avg Loss: {stats['avg_loss']:.4f}
            Learning Rate: {stats['learning_rate']:.6f}
            
            Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
            """
            plt.text(
                0.1,
                0.1,
                info_text,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment="bottom",
                fontfamily="monospace",
            )
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(
                f"training_progress_gpu_ep{self.episode}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.show()
        return True

    def toggle_training(self):
        self.training = not self.training
        self.buttons[4].text = "Pause Train" if self.training else "Resume Train"
        print(f"Training {'resumed' if self.training else 'paused'}")
        return True

    def reset_episode(self):
        self.car.reset(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        self.episode_rewards.append(self.score)
        self.scores.append(self.score)
        self.max_score = max(self.max_score, self.score)
        self.score = 0
        self.episode += 1
        self.step_count = 0

        # Print episode summary every 10 episodes
        if self.episode % 10 == 0:
            self.agent.print_stats()
            recent_avg = (
                np.mean(self.scores[-10:])
                if len(self.scores) >= 10
                else np.mean(self.scores)
            )
            print(
                f"Episode {self.episode} completed. Recent average score: {recent_avg:.2f}"
            )

    def calculate_reward(self):
        """Calculate reward based on car performance"""
        reward = 0

        # Base reward for staying alive
        reward += 0.1

        # Reward for distance traveled (encourage movement)
        if hasattr(self.car, "total_distance"):
            reward += min(self.car.total_distance * 0.001, 0.5)

        # Penalty for being off-track
        if self.car.off_track_timer > 0:
            off_track_penalty = (
                self.car.off_track_timer / self.car.max_off_track_time
            ) * 1.0
            reward -= off_track_penalty

        # Reward for staying on track
        if self.car.off_track_timer == 0:
            reward += 0.2

        # Speed reward (encourage maintaining good speed)
        speed_ratio = self.car.speed / self.car.max_speed
        if speed_ratio > 0.5:
            reward += 0.1

        return reward

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            # Handle button events
            mouse_pos = pygame.mouse.get_pos()
            for button in self.buttons:
                button.check_hover(mouse_pos)
                button.handle_event(event)

            # Keyboard controls for manual testing
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset_episode()
                elif event.key == pygame.K_UP:
                    self.car.speed = min(self.car.speed + 0.5, self.car.max_speed)
                elif event.key == pygame.K_DOWN:
                    self.car.speed = max(self.car.speed - 0.5, 1.0)
                elif event.key == pygame.K_LEFT:
                    self.car.direction = -1
                elif event.key == pygame.K_RIGHT:
                    self.car.direction = 1
                elif event.key == pygame.K_f:
                    self.toggle_fullscreen()
                elif event.key == pygame.K_s:
                    self.save_model()
                elif event.key == pygame.K_l:
                    self.load_model()
                elif event.key == pygame.K_p:
                    self.plot_progress()
                elif event.key == pygame.K_n:
                    self.next_track()
                elif event.key == pygame.K_t:
                    self.toggle_training()

            if event.type == pygame.KEYUP:
                if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    self.car.direction = 0

    def update(self):
        if self.paused:
            return

        # Update car
        self.car.update(self.sand)
        self.step_count += 1

        if self.training:
            # Get current state
            state = self.car.get_state()

            # Calculate reward
            reward = self.calculate_reward()

            # Get action from agent
            done = not self.car.alive
            action = self.agent.update(reward, state, done)

            # Apply action
            if action == 0:  # Straight
                self.car.direction = 0
            elif action == 1:  # Left
                self.car.direction = -1
            elif action == 2:  # Right
                self.car.direction = 1

            # Update score
            if self.car.alive:
                self.score += 1

            # Check if episode is done
            if not self.car.alive or self.step_count > 5000:  # Max steps per episode
                if not self.car.alive:
                    # Give final negative reward for crash
                    final_reward = -10.0
                    self.agent.update(final_reward, state, True)

                self.reset_episode()

        # Auto-save model
        current_time = pygame.time.get_ticks()
        if current_time - self.last_save_time > self.save_interval:
            self.agent.save("auto_save_gpu.pth")
            self.last_save_time = current_time
            print(f"Auto-saved model after {self.save_interval//60000} minutes")

    def draw(self):
        # Draw track
        screen.fill(DARK_BLUE)
        screen.blit(self.track_img, (0, 0))

        # Draw car
        screen.blit(self.car.image, self.car.rect)

        # Draw radars
        self.car.draw_radars(screen)

        # Draw overlay
        screen.blit(self.overlay_img, (0, 0))

        # Draw button panel
        screen.blit(self.button_panel, (0, 0))

        # Draw UI buttons
        for button in self.buttons:
            button.draw(screen)

        # Draw enhanced stats panel
        screen.blit(self.stats_panel, self.stats_panel_rect)

        # Draw enhanced stats with GPU information
        stats = self.agent.get_stats()

        score_text = font.render(f"Score: {self.score}", True, WHITE)
        episode_text = font.render(f"Episode: {self.episode}", True, WHITE)
        max_score_text = font.render(f"Max Score: {self.max_score}", True, WHITE)
        steps_text = small_font.render(f"Agent Steps: {stats['steps']:,}", True, WHITE)

        status_color = GREEN if self.training else YELLOW
        status_text = font.render(
            f"Status: {'Training' if self.training else 'Paused'}",
            True,
            status_color,
        )

        # GPU and training info
        device_text = small_font.render(
            f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}",
            True,
            CYAN if torch.cuda.is_available() else YELLOW,
        )
        epsilon_text = small_font.render(
            f"Epsilon: {stats['epsilon']:.4f}", True, WHITE
        )
        memory_text = small_font.render(
            f"Memory: {stats['memory_size']:,}", True, WHITE
        )
        loss_text = small_font.render(f"Avg Loss: {stats['avg_loss']:.4f}", True, WHITE)
        lr_text = tiny_font.render(
            f"Learning Rate: {stats['learning_rate']:.6f}", True, WHITE
        )

        track_name = os.path.basename(self.track_path)
        track_text = small_font.render(f"Track: {track_name}", True, WHITE)

        # Position stats relative to panel
        stats_x = self.stats_panel_rect.x + 15
        stats_y = self.stats_panel_rect.y + 15
        line_height = 25
        small_line_height = 20

        # Draw main stats
        screen.blit(score_text, (stats_x, stats_y))
        screen.blit(episode_text, (stats_x, stats_y + line_height))
        screen.blit(max_score_text, (stats_x, stats_y + 2 * line_height))
        screen.blit(status_text, (stats_x, stats_y + 3 * line_height))

        # Draw technical stats
        screen.blit(device_text, (stats_x, stats_y + 4 * line_height + 5))
        screen.blit(
            steps_text, (stats_x, stats_y + 4 * line_height + 5 + small_line_height)
        )
        screen.blit(
            epsilon_text,
            (stats_x, stats_y + 4 * line_height + 5 + 2 * small_line_height),
        )
        screen.blit(
            memory_text,
            (stats_x, stats_y + 4 * line_height + 5 + 3 * small_line_height),
        )
        screen.blit(
            loss_text, (stats_x, stats_y + 4 * line_height + 5 + 4 * small_line_height)
        )
        screen.blit(
            lr_text, (stats_x, stats_y + 4 * line_height + 5 + 5 * small_line_height)
        )

        # Draw track name at bottom
        screen.blit(track_text, (20, SCREEN_HEIGHT - 40))

        # Draw FPS and performance info
        fps_text = small_font.render(f"FPS: {int(clock.get_fps())}", True, WHITE)
        screen.blit(fps_text, (SCREEN_WIDTH - 120, SCREEN_HEIGHT - 60))

        # GPU memory usage if available
        if torch.cuda.is_available():
            try:
                memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                memory_cached = torch.cuda.memory_reserved() / 1024**2  # MB
                gpu_memory_text = tiny_font.render(
                    f"GPU Mem: {memory_allocated:.1f}/{memory_cached:.1f} MB",
                    True,
                    CYAN,
                )
                screen.blit(gpu_memory_text, (SCREEN_WIDTH - 200, SCREEN_HEIGHT - 40))
            except:
                pass

        pygame.display.flip()

    def run(self):
        print("Starting GPU-accelerated DQN training...")
        print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            clock.tick(FPS)

        # Save final model before exit
        print("Saving final model...")
        self.agent.save("final_model_gpu.pth")
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    try:
        game = Game()
        game.run()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        pygame.quit()
        sys.exit()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
        pygame.quit()
        sys.exit()
