# /usr/bin/env python3
"""
test.py - Simple testing script for trained DQN model
Tests the saved model in the same environment without training
"""

import pygame
import os
import math
import sys
import numpy as np
import torch
from PIL import Image as PILImage
from agent import DqnGPU  # Import the GPU-accelerated agent

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Autonomous Car - Testing Trained Model")

# Clock for controlling FPS
clock = pygame.time.Clock()
FPS = 60

# Colors
DARK_BLUE = (25, 35, 60)
WHITE = (240, 245, 255)
RED = (255, 80, 80)
GREEN = (80, 255, 150)
CYAN = (80, 220, 255)
YELLOW = (255, 230, 80)

# Font
font = pygame.font.SysFont("Arial", 28)
small_font = pygame.font.SysFont("Arial", 20)


# Load available tracks
def load_tracks():
    tracks = []
    overlays = []
    tracks_dir = "tracks"

    if not os.path.exists(tracks_dir):
        print(f"Tracks directory '{tracks_dir}' not found!")
        return []

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
    return (img_array < 128).astype(int)  # 1 for off-track, 0 for on-track


# Car class (simplified version for testing)
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
        self.radar_endpoints = []
        self.collision_point_right = (0, 0)
        self.collision_point_left = (0, 0)
        self.speed = 3.0
        self.max_speed = 6.0
        self.off_track_timer = 0
        self.max_off_track_time = 60
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
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.velocity.rotate_ip(self.rotation_vel)
        elif self.direction == -1:
            self.angle += self.rotation_vel
            self.velocity.rotate_ip(-self.rotation_vel)

        self.rect.center += self.velocity * self.speed
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def rotate(self):
        if self.direction != 0:
            rotation = self.rotation_vel * self.direction
            self.angle = (self.angle + rotation) % 360

    def radar(self, radar_angle, sand):
        length = 0
        x, y = int(self.rect.center[0]), int(self.rect.center[1])
        end_point = (0, 0)

        while length < 250:
            x = int(
                self.rect.center[0]
                + math.cos(math.radians(self.angle + radar_angle)) * length
            )
            y = int(
                self.rect.center[1]
                - math.sin(math.radians(self.angle + radar_angle)) * length
            )

            if 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT:
                if sand[y, x] == 1:
                    break
            else:
                break

            length += 1
            end_point = (x, y)

        self.radar_endpoints.append(end_point)
        dist = math.sqrt(
            (self.rect.center[0] - x) ** 2 + (self.rect.center[1] - y) ** 2
        )
        self.radars.append([radar_angle, dist])

    def collision(self, sand):
        center_x = int(self.rect.center[0])
        center_y = int(self.rect.center[1])

        if 0 <= center_x < SCREEN_WIDTH and 0 <= center_y < SCREEN_HEIGHT:
            if sand[center_y, center_x] == 1:
                self.off_track_timer += 1
            else:
                self.off_track_timer = max(0, self.off_track_timer - 2)
        else:
            self.off_track_timer += 1

        if self.off_track_timer >= self.max_off_track_time:
            self.alive = False

        # Collision points
        length = 25
        self.collision_point_right = [
            int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
            int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length),
        ]
        self.collision_point_left = [
            int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
            int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length),
        ]

        try:
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
        for end_point in self.radar_endpoints:
            pygame.draw.line(surface, RED, self.rect.center, end_point, 1)
            pygame.draw.circle(surface, GREEN, end_point, 3)

        pygame.draw.circle(surface, CYAN, self.collision_point_right, 3)
        pygame.draw.circle(surface, CYAN, self.collision_point_left, 3)

        if self.off_track_timer > 0:
            bar_width = 40
            bar_height = 5
            bar_x = self.rect.centerx - bar_width // 2
            bar_y = self.rect.top - 15
            fill_width = (self.off_track_timer / self.max_off_track_time) * bar_width

            pygame.draw.rect(
                surface, (40, 45, 60), (bar_x, bar_y, bar_width, bar_height), 1
            )
            pygame.draw.rect(surface, RED, (bar_x, bar_y, fill_width, bar_height))

    def get_state(self):
        state = [min(radar[1] / 250, 1.0) for radar in self.radars]
        state.append(self.speed / self.max_speed)
        state.append(self.off_track_timer / self.max_off_track_time)
        return state


# Test Game class
class TestGame:
    def __init__(self, model_path="last_brain_gpu.pth"):
        # Load tracks
        self.track_pairs = load_tracks()
        if not self.track_pairs:
            print("No tracks found! Please add tracks to the tracks directory.")
            pygame.quit()
            sys.exit()

        self.current_track_index = 0
        self.track_path, self.overlay_path = self.track_pairs[self.current_track_index]

        # Load track images
        self.track_img = pygame.image.load(self.track_path).convert()
        self.track_img = pygame.transform.scale(
            self.track_img, (SCREEN_WIDTH, SCREEN_HEIGHT)
        )

        self.overlay_img = pygame.image.load(self.overlay_path).convert_alpha()
        self.overlay_img = pygame.transform.scale(
            self.overlay_img, (SCREEN_WIDTH, SCREEN_HEIGHT)
        )

        self.sand = create_sand(self.overlay_path)

        # Initialize car
        self.car = Car(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

        # Initialize agent (for testing only - no training)
        self.agent = DqnGPU(
            input_size=8,
            nb_action=3,
            gamma=0.99,
            lr=0.0001,
            memory_capacity=1000,  # Smaller memory for testing
        )

        # Load the trained model
        try:
            self.agent.load(model_path)
            print(f"Successfully loaded model: {model_path}")
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")
            print("Make sure the model file exists in the 'models' directory")
            pygame.quit()
            sys.exit()

        self.running = True
        self.paused = False
        self.score = 0
        self.episode = 0
        self.max_score = 0
        self.scores = []
        self.step_count = 0
        self.show_radars = True

    def next_track(self):
        self.current_track_index = (self.current_track_index + 1) % len(
            self.track_pairs
        )
        self.track_path, self.overlay_path = self.track_pairs[self.current_track_index]

        self.track_img = pygame.image.load(self.track_path).convert()
        self.track_img = pygame.transform.scale(
            self.track_img, (SCREEN_WIDTH, SCREEN_HEIGHT)
        )

        self.overlay_img = pygame.image.load(self.overlay_path).convert_alpha()
        self.overlay_img = pygame.transform.scale(
            self.overlay_img, (SCREEN_WIDTH, SCREEN_HEIGHT)
        )

        self.sand = create_sand(self.overlay_path)
        self.car.reset(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        print(f"Switched to track: {os.path.basename(self.track_path)}")

    def reset_episode(self):
        self.scores.append(self.score)
        self.max_score = max(self.max_score, self.score)
        print(f"Episode {self.episode} completed. Score: {self.score}")

        self.car.reset(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        self.score = 0
        self.episode += 1
        self.step_count = 0

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print(f"{'Paused' if self.paused else 'Resumed'}")
                elif event.key == pygame.K_r:
                    self.reset_episode()
                elif event.key == pygame.K_n:
                    self.next_track()
                elif event.key == pygame.K_v:
                    self.show_radars = not self.show_radars
                    print(f"Radar visualization: {'ON' if self.show_radars else 'OFF'}")

    def update(self):
        if self.paused:
            return

        # Update car
        self.car.update(self.sand)
        self.step_count += 1

        # Get current state
        state = self.car.get_state()

        # Get action from trained agent (no training)
        from agent import device  # Import device from agent module

        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)
        action = self.agent.select_action(
            state_tensor, training=False
        )  # No exploration

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
        if not self.car.alive or self.step_count > 10000:  # Max steps per episode
            self.reset_episode()

    def draw(self):
        # Draw track
        screen.fill(DARK_BLUE)
        screen.blit(self.track_img, (0, 0))

        # Draw car
        screen.blit(self.car.image, self.car.rect)

        # Draw radars (optional)
        if self.show_radars:
            self.car.draw_radars(screen)

        # Draw overlay
        screen.blit(self.overlay_img, (0, 0))

        # Draw stats
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        episode_text = font.render(f"Episode: {self.episode}", True, WHITE)
        max_score_text = font.render(f"Max Score: {self.max_score}", True, WHITE)

        status_text = font.render("TESTING MODE", True, GREEN)

        track_name = os.path.basename(self.track_path)
        track_text = small_font.render(f"Track: {track_name}", True, WHITE)

        # Control hints
        controls_text = small_font.render(
            "Controls: SPACE=Pause, R=Reset, N=Next Track, V=Toggle Radars, ESC=Exit",
            True,
            YELLOW,
        )

        # Position stats
        screen.blit(score_text, (20, 20))
        screen.blit(episode_text, (20, 50))
        screen.blit(max_score_text, (20, 80))
        screen.blit(status_text, (20, 110))
        screen.blit(track_text, (20, SCREEN_HEIGHT - 60))
        screen.blit(controls_text, (20, SCREEN_HEIGHT - 30))

        # FPS
        fps_text = small_font.render(f"FPS: {int(clock.get_fps())}", True, WHITE)
        screen.blit(fps_text, (SCREEN_WIDTH - 100, 20))

        pygame.display.flip()

    def run(self):
        print("=" * 60)
        print("TESTING TRAINED DQN MODEL")
        print("=" * 60)
        print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print(f"Loaded model with {self.agent.steps:,} training steps")
        print()
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  R - Reset episode")
        print("  N - Next track")
        print("  V - Toggle radar visualization")
        print("  ESC - Exit")
        print("=" * 60)

        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            clock.tick(FPS)

        # Print final statistics
        print("\nTesting session completed!")
        print(f"Episodes completed: {self.episode}")
        print(f"Best score: {self.max_score}")
        if self.scores:
            avg_score = np.mean(self.scores)
            print(f"Average score: {avg_score:.2f}")

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    # You can specify which model to load
    model_to_test = "last_brain_gpu.pth"  # Default model

    # Check command line arguments for custom model path
    if len(sys.argv) > 1:
        model_to_test = sys.argv[1]

    print(f"Testing model: {model_to_test}")

    try:
        test_game = TestGame(model_to_test)
        test_game.run()
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        pygame.quit()
        sys.exit()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
        pygame.quit()
        sys.exit()
