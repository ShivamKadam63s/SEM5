import pygame
import math
import time
import sys
import os
import random
from datetime import datetime
from collections import deque  # For BFS queue

# Define colors for visualization
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
PURPLE = (128, 0, 128)
YELLOW = (255, 255, 0)
DARK_GRAY = (40, 40, 40)
LIGHT_GRAY = (220, 220, 220)

# Grid settings
GRID_SIZE = 20
CELL_SIZE = 30
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE
INFO_PANEL_HEIGHT = 120
LEGEND_WIDTH = 250
TOTAL_WIDTH = WINDOW_WIDTH + LEGEND_WIDTH
TOTAL_HEIGHT = WINDOW_WIDTH + INFO_PANEL_HEIGHT

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((TOTAL_WIDTH, TOTAL_HEIGHT))
pygame.display.set_caption("BFS Pathfinding Demo (Randomized)")
clock = pygame.time.Clock()

# Font for displaying information
font = pygame.font.SysFont('Arial', 16)
title_font = pygame.font.SysFont('Arial', 24, bold=True)

# Create recordings directory if it doesn't exist
if not os.path.exists("recordings"):
    os.makedirs("recordings")

# Video recording variables
recording = True
frames = []
log_file = None

# Node class for BFS algorithm
class Node:
    def __init__(self, position, parent=None, depth=0):
        self.position = position  # (x, y)
        self.parent = parent
        self.depth = depth  # Distance from start (for BFS, equals steps)

    def __eq__(self, other):
        return self.position == other.position

# Function to get valid neighbors
def get_neighbors(grid, node):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, down, left, up
    neighbors = []
    for dx, dy in directions:
        nx, ny = node.position[0] + dx, node.position[1] + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[ny][nx] == 0:
            neighbors.append(Node((nx, ny), node, node.depth + 1))
    return neighbors

# Reconstruct path from goal to start
def reconstruct_path(current):
    path = []
    while current:
        path.append(current.position)
        current = current.parent
    return path[::-1]

# Print to both console and log file
def log_message(message):
    print(message)
    if log_file:
        log_file.write(message + "\n")

# Start recording and logging
def start_recording():
    global recording, frames, log_file
    recording = True
    frames = []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"recordings/bfs_random_log_{timestamp}.txt"
    log_file = open(log_filename, "w", encoding='utf-8')  # UTF-8 to avoid encoding errors
    
    log_message("BFS Pathfinding Demo (Randomized Neighbor Order)")
    log_message("=" * 50)
    log_message(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message("=" * 50)

# Stop recording and save video
def stop_recording():
    global recording, log_file
    recording = False
    
    if not frames:
        log_message("No frames to save.")
        return
        
    try:
        import cv2
        import numpy as np
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recordings/bfs_random_demo_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 10.0, (TOTAL_WIDTH, TOTAL_HEIGHT))
        
        for frame in frames:
            frame_rgb = pygame.surfarray.array3d(frame)
            frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        log_message(f"Recording saved as {filename}")
        
    except ImportError:
        log_message("OpenCV not installed. Cannot save video.")
        log_message("Install it with: pip install opencv-python")
    except Exception as e:
        log_message(f"Error saving video: {e}")
    
    if log_file:
        log_message("=" * 50)
        log_message(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_file.close()
        log_file = None

# Draw the grid and algorithm states
def draw_grid(screen, grid, queue, visited, current, path, start, goal, current_node_info=None):
    # Draw grid background
    screen.fill(WHITE)
    
    # Draw grid area
    grid_rect = pygame.Rect(0, 0, WINDOW_WIDTH, WINDOW_WIDTH)
    pygame.draw.rect(screen, WHITE, grid_rect)
    
    # Draw cells
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            
            if grid[y][x] == 1:  # Obstacle
                pygame.draw.rect(screen, BLACK, rect)
            elif (x, y) == start:  # Start
                pygame.draw.rect(screen, RED, rect)
            elif (x, y) == goal:  # Goal
                pygame.draw.rect(screen, GREEN, rect)
            elif (x, y) in visited:  # Visited
                pygame.draw.rect(screen, GRAY, rect)
            elif any(node.position == (x, y) for node in queue):  # In queue (frontier)
                pygame.draw.rect(screen, BLUE, rect)
            else:  # Empty
                pygame.draw.rect(screen, WHITE, rect)
                
            pygame.draw.rect(screen, BLACK, rect, 1)  # Grid lines
    
    # Draw current path
    if path:
        for pos in path:
            if pos != start and pos != goal:
                rect = pygame.Rect(pos[0] * CELL_SIZE, pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, PURPLE, rect)
                pygame.draw.rect(screen, BLACK, rect, 1)
    
    # Highlight current node
    if current:
        rect = pygame.Rect(current.position[0] * CELL_SIZE, current.position[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, YELLOW, rect)
        pygame.draw.rect(screen, BLACK, rect, 1)
    
    # Draw legend
    legend_rect = pygame.Rect(WINDOW_WIDTH, 0, LEGEND_WIDTH, WINDOW_WIDTH)
    pygame.draw.rect(screen, DARK_GRAY, legend_rect)
    
    legend_title = title_font.render("Legend", True, LIGHT_GRAY)
    screen.blit(legend_title, (WINDOW_WIDTH + 10, 20))
    
    legend_items = [
        (RED, "Enemy (Start)"),
        (GREEN, "Player (Goal)"),
        (BLUE, "Nodes in Queue"),
        (GRAY, "Visited Nodes"),
        (PURPLE, "Final Path"),
        (YELLOW, "Current Node"),
        (BLACK, "Walls/Obstacles")
    ]
    
    for i, (color, text) in enumerate(legend_items):
        y_pos = 60 + i * 35
        pygame.draw.rect(screen, color, (WINDOW_WIDTH + 20, y_pos, 20, 20))
        text_surf = font.render(text, True, LIGHT_GRAY)
        screen.blit(text_surf, (WINDOW_WIDTH + 50, y_pos))
    
    # Draw info panel
    info_panel = pygame.Rect(0, WINDOW_WIDTH, TOTAL_WIDTH, INFO_PANEL_HEIGHT)
    pygame.draw.rect(screen, DARK_GRAY, info_panel)
    
    queue_text = font.render(f"Queue: {len(queue)} nodes", True, LIGHT_GRAY)
    visited_text = font.render(f"Visited: {len(visited)} nodes", True, LIGHT_GRAY)
    path_text = font.render(f"Path Length: {len(path) if path else 0} nodes", True, LIGHT_GRAY)
    
    screen.blit(queue_text, (20, WINDOW_WIDTH + 20))
    screen.blit(visited_text, (20, WINDOW_WIDTH + 50))
    screen.blit(path_text, (20, WINDOW_WIDTH + 80))
    
    if current_node_info:
        current_text = font.render(f"Current Node: {current_node_info['position']}", True, LIGHT_GRAY)
        depth_text = font.render(f"Depth/Distance: {current_node_info['depth']}", True, LIGHT_GRAY)
        
        screen.blit(current_text, (TOTAL_WIDTH // 3, WINDOW_WIDTH + 20))
        screen.blit(depth_text, (TOTAL_WIDTH // 3, WINDOW_WIDTH + 50))
    
    status_text = font.render("Recording: Yes", True, LIGHT_GRAY)
    screen.blit(status_text, (TOTAL_WIDTH - 150, WINDOW_WIDTH + 20))

# Main BFS function with step-by-step visualization
def bfs(grid, start_pos, goal_pos, screen):
    start = Node(start_pos)
    goal = Node(goal_pos)

    queue = deque([start])  # 👈 FIFO Queue for BFS
    visited = set()
    path = []
    current_node_info = None

    # Initial state
    draw_grid(screen, grid, list(queue), visited, None, path, start_pos, goal_pos)
    if recording:
        frames.append(screen.copy())
    pygame.display.flip()

    while queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if recording:
                    stop_recording()
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if recording:
                        stop_recording()
                    pygame.quit()
                    sys.exit()

        # 👇 Dequeue from FRONT (BFS = First In, First Out)
        current = queue.popleft()
        
        if current.position in visited:
            continue
            
        visited.add(current.position)

        current_node_info = {
            'position': current.position,
            'depth': current.depth
        }

        current_path = reconstruct_path(current)
        draw_grid(screen, grid, list(queue), visited, current, current_path, start_pos, goal_pos, current_node_info)
        
        log_message("\n" + "="*50)
        log_message(f"Current node: {current.position} at depth {current.depth}")
        log_message(f"Queue size: {len(queue)}")
        log_message(f"Visited nodes: {len(visited)}")
        
        if recording:
            frames.append(screen.copy())
        
        pygame.display.flip()
        time.sleep(0.1)

        # Goal check
        if current.position == goal_pos:
            path = reconstruct_path(current)
            log_message("✅ Path found: " + str(path))
            draw_grid(screen, grid, list(queue), visited, None, path, start_pos, goal_pos, current_node_info)
            
            if recording:
                frames.append(screen.copy())
                
            pygame.display.flip()
            return path, list(queue), visited

        # Get and shuffle neighbors
        neighbors = get_neighbors(grid, current)
        random.shuffle(neighbors)  # Randomize exploration order
        
        for neighbor in neighbors:
            if neighbor.position not in visited:
                queue.append(neighbor)
                log_message(f" -> Adding neighbor {neighbor.position} to queue")

    log_message("❌ No path found")
    return None, list(queue), visited

# Create maze (identical to DFS/A*)
def create_maze():
    grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    
    # Border walls
    for i in range(GRID_SIZE):
        grid[0][i] = 1
        grid[GRID_SIZE-1][i] = 1
        grid[i][0] = 1
        grid[i][GRID_SIZE-1] = 1
    
    # Middle walls
    for i in range(5, 15):
        grid[5][i] = 1
        grid[15][i] = 1
        
    for i in range(5, 15):
        grid[i][5] = 1
        grid[i][15] = 1
        
    # Openings
    grid[5][10] = 0
    grid[15][10] = 0
    grid[10][5] = 0
    grid[10][15] = 0
    
    # Random obstacles
    obstacles = [
        (3, 3), (3, 4), (4, 3),
        (16, 16), (16, 17), (17, 16),
        (8, 8), (8, 9), (9, 8), (9, 9),
        (12, 12), (12, 13), (13, 12), (13, 13)
    ]
    
    for x, y in obstacles:
        grid[y][x] = 1
        
    return grid

# Main function
def main():
    start_recording()
    
    grid = create_maze()
    start_pos = (1, 1)
    goal_pos = (GRID_SIZE-2, GRID_SIZE-2)
    
    grid[start_pos[1]][start_pos[0]] = 0
    grid[goal_pos[1]][goal_pos[0]] = 0
    
    log_message(f"Start position: {start_pos}")
    log_message(f"Goal position: {goal_pos}")
    log_message("=" * 50)
    
    path, queue, visited = bfs(grid, start_pos, goal_pos, screen)
    
    draw_grid(screen, grid, queue, visited, None, path, start_pos, goal_pos)
    
    if recording:
        frames.append(screen.copy())
        
    pygame.display.flip()
    
    if recording:
        stop_recording()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
    pygame.quit()

if __name__ == "__main__":
    main()
