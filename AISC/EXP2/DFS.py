import pygame
import math
import time
import sys
import os
import random  # Added for randomization
from datetime import datetime
from collections import deque

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
GRID_SIZE = 20  # 20x20 grid
CELL_SIZE = 30  # Size of each cell in pixels
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE
INFO_PANEL_HEIGHT = 120  # Increased height for better spacing
LEGEND_WIDTH = 250
TOTAL_WIDTH = WINDOW_WIDTH + LEGEND_WIDTH
TOTAL_HEIGHT = WINDOW_WIDTH + INFO_PANEL_HEIGHT

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((TOTAL_WIDTH, TOTAL_HEIGHT))
pygame.display.set_caption("DFS Pathfinding Demo (Randomized)")
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

# Node class for DFS algorithm
class Node:
    def __init__(self, position, parent=None, depth=0):
        self.position = position  # (x, y)
        self.parent = parent
        self.depth = depth  # Depth in the search tree

    def __eq__(self, other):
        return self.position == other.position

# Function to get valid neighbors (up, down, left, right)
def get_neighbors(grid, node):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, down, left, up
    neighbors = []
    for dx, dy in directions:
        nx, ny = node.position[0] + dx, node.position[1] + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[ny][nx] == 0:  # Valid and not obstacle
            neighbors.append(Node((nx, ny), node, node.depth + 1))
    return neighbors

# Reconstruct path from goal to start
def reconstruct_path(current):
    path = []
    while current:
        path.append(current.position)
        current = current.parent
    return path[::-1]  # Reverse to get start to goal

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
    
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"recordings/dfs_random_log_{timestamp}.txt"
    log_file = open(log_filename, "w")
    
    log_message("DFS Pathfinding Demo (Randomized Neighbor Order)")
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
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recordings/dfs_random_demo_{timestamp}.mp4"
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 10.0, (TOTAL_WIDTH, TOTAL_HEIGHT))
        
        # Write frames to video
        for frame in frames:
            # Convert pygame surface to numpy array
            frame_rgb = pygame.surfarray.array3d(frame)
            frame_rgb = np.transpose(frame_rgb, (1, 0, 2))  # Fix dimensions
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        log_message(f"Recording saved as {filename}")
        
    except ImportError:
        log_message("OpenCV not installed. Cannot save video.")
        log_message("Install it with: pip install opencv-python")
    except Exception as e:
        log_message(f"Error saving video: {e}")
    
    # Close log file
    if log_file:
        log_message("=" * 50)
        log_message(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_file.close()
        log_file = None

# Draw the grid and algorithm states
def draw_grid(screen, grid, stack, visited, current, path, start, goal, current_node_info=None):
    # Draw grid background
    screen.fill(WHITE)
    
    # Draw grid area
    grid_rect = pygame.Rect(0, 0, WINDOW_WIDTH, WINDOW_WIDTH)
    pygame.draw.rect(screen, WHITE, grid_rect)
    
    # Draw cells
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            
            # Determine cell color based on its state
            if grid[y][x] == 1:  # Obstacle
                pygame.draw.rect(screen, BLACK, rect)
            elif (x, y) == start:  # Start position
                pygame.draw.rect(screen, RED, rect)
            elif (x, y) == goal:  # Goal position
                pygame.draw.rect(screen, GREEN, rect)
            elif (x, y) in visited:  # Visited nodes
                pygame.draw.rect(screen, GRAY, rect)
            elif any(node.position == (x, y) for node in stack):  # Nodes in stack
                pygame.draw.rect(screen, BLUE, rect)
            else:  # Empty cell
                pygame.draw.rect(screen, WHITE, rect)
                
            # Draw grid lines
            pygame.draw.rect(screen, BLACK, rect, 1)
    
    # Draw current path
    if path:
        for pos in path:
            if pos != start and pos != goal:  # Don't overwrite start and goal colors
                rect = pygame.Rect(pos[0] * CELL_SIZE, pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, PURPLE, rect)
                pygame.draw.rect(screen, BLACK, rect, 1)  # Redraw grid lines
    
    # Highlight current node
    if current:
        rect = pygame.Rect(current.position[0] * CELL_SIZE, current.position[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, YELLOW, rect)
        pygame.draw.rect(screen, BLACK, rect, 1)  # Redraw grid lines
    
    # Draw legend on the right side
    legend_rect = pygame.Rect(WINDOW_WIDTH, 0, LEGEND_WIDTH, WINDOW_WIDTH)
    pygame.draw.rect(screen, DARK_GRAY, legend_rect)
    
    # Draw legend title
    legend_title = title_font.render("Legend", True, LIGHT_GRAY)
    screen.blit(legend_title, (WINDOW_WIDTH + 10, 20))
    
    # Draw legend items with better spacing
    legend_items = [
        (RED, "Enemy (Start)"),
        (GREEN, "Player (Goal)"),
        (BLUE, "Nodes in Stack"),
        (GRAY, "Visited Nodes"),
        (PURPLE, "Final Path"),
        (YELLOW, "Current Node"),
        (BLACK, "Walls/Obstacles")
    ]
    
    for i, (color, text) in enumerate(legend_items):
        y_pos = 60 + i * 35  # Increased spacing
        pygame.draw.rect(screen, color, (WINDOW_WIDTH + 20, y_pos, 20, 20))
        text_surf = font.render(text, True, LIGHT_GRAY)
        screen.blit(text_surf, (WINDOW_WIDTH + 50, y_pos))
    
    # Draw information panel at the bottom
    info_panel = pygame.Rect(0, WINDOW_WIDTH, TOTAL_WIDTH, INFO_PANEL_HEIGHT)
    pygame.draw.rect(screen, DARK_GRAY, info_panel)
    
    # Draw pathfinding stats on the left with better spacing
    stack_text = font.render(f"Stack: {len(stack)} nodes", True, LIGHT_GRAY)
    visited_text = font.render(f"Visited: {len(visited)} nodes", True, LIGHT_GRAY)
    path_text = font.render(f"Path Length: {len(path) if path else 0} nodes", True, LIGHT_GRAY)
    
    screen.blit(stack_text, (20, WINDOW_WIDTH + 20))
    screen.blit(visited_text, (20, WINDOW_WIDTH + 50))
    screen.blit(path_text, (20, WINDOW_WIDTH + 80))
    
    # Draw current node info in the middle if available
    if current_node_info:
        current_text = font.render(f"Current Node: {current_node_info['position']}", True, LIGHT_GRAY)
        depth_text = font.render(f"Depth: {current_node_info['depth']}", True, LIGHT_GRAY)
        
        screen.blit(current_text, (TOTAL_WIDTH // 3, WINDOW_WIDTH + 20))
        screen.blit(depth_text, (TOTAL_WIDTH // 3, WINDOW_WIDTH + 50))
    
    # Draw recording status
    status_text = font.render("Recording: Yes", True, LIGHT_GRAY)
    screen.blit(status_text, (TOTAL_WIDTH - 150, WINDOW_WIDTH + 20))

# Main DFS function with step-by-step visualization and RANDOM neighbor order
def dfs(grid, start_pos, goal_pos, screen):
    start = Node(start_pos)
    goal = Node(goal_pos)

    stack = [start]  # Use list as stack (LIFO)
    visited = set()  # Keep track of visited positions
    path = []  # Final path
    current_node_info = None

    # Capture initial state
    draw_grid(screen, grid, stack, visited, None, path, start_pos, goal_pos)
    if recording:
        frames.append(screen.copy())
    pygame.display.flip()

    while stack:
        # Handle Pygame events to allow quitting
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

        # Get node from top of stack
        current = stack.pop()
        
        # Skip if already visited
        if current.position in visited:
            continue
            
        # Mark as visited
        visited.add(current.position)

        # Prepare current node info for display
        current_node_info = {
            'position': current.position,
            'depth': current.depth
        }

        # Visualize current state
        current_path = reconstruct_path(current)
        draw_grid(screen, grid, stack, visited, current, current_path, start_pos, goal_pos, current_node_info)
        
        # Log current state
        log_message("\n" + "="*50)
        log_message(f"Current node: {current.position} at depth {current.depth}")
        log_message(f"Stack size: {len(stack)}")
        log_message(f"Visited nodes: {len(visited)}")
        
        # Capture frame if recording
        if recording:
            frames.append(screen.copy())
        
        pygame.display.flip()
        time.sleep(0.1)  # Delay for visualization

        # If goal reached
        if current.position == goal_pos:
            path = reconstruct_path(current)
            log_message("Path found: " + str(path))
            # Keep the stack and visited sets visible
            draw_grid(screen, grid, stack, visited, None, path, start_pos, goal_pos, current_node_info)
            
            # Capture final frame if recording
            if recording:
                frames.append(screen.copy())
                
            pygame.display.flip()
            return path, stack, visited

        # Get valid neighbors
        neighbors = get_neighbors(grid, current)
        
        # RANDOMIZE the order of neighbors before adding to stack
        random.shuffle(neighbors)
        
        # Add unvisited neighbors to stack
        for neighbor in neighbors:
            if neighbor.position not in visited:
                stack.append(neighbor)
                # Optional: Log which neighbors we're adding
                log_message(f" -> Adding neighbor {neighbor.position} to stack")

    log_message("No path found")
    return None, stack, visited

# Create a more interesting maze (same as before)
def create_maze():
    # Initialize empty grid
    grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    
    # Add border walls
    for i in range(GRID_SIZE):
        grid[0][i] = 1
        grid[GRID_SIZE-1][i] = 1
        grid[i][0] = 1
        grid[i][GRID_SIZE-1] = 1
    
    # Add some obstacles in the middle
    for i in range(5, 15):
        grid[5][i] = 1
        grid[15][i] = 1
        
    for i in range(5, 15):
        grid[i][5] = 1
        grid[i][15] = 1
        
    # Add some openings
    grid[5][10] = 0
    grid[15][10] = 0
    grid[10][5] = 0
    grid[10][15] = 0
    
    # Add some random obstacles
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
    # Start recording and logging
    start_recording()
    
    # Create maze
    grid = create_maze()
    
    # Start (enemy) and goal (player) positions
    start_pos = (1, 1)
    goal_pos = (GRID_SIZE-2, GRID_SIZE-2)
    
    # Ensure start and goal are empty
    grid[start_pos[1]][start_pos[0]] = 0
    grid[goal_pos[1]][goal_pos[0]] = 0
    
    log_message(f"Start position: {start_pos}")
    log_message(f"Goal position: {goal_pos}")
    log_message("=" * 50)
    
    # Run DFS with randomized neighbor order
    path, stack, visited = dfs(grid, start_pos, goal_pos, screen)
    
    # Draw final result with stack and visited sets still visible
    draw_grid(screen, grid, stack, visited, None, path, start_pos, goal_pos)
    
    # Capture final frame
    if recording:
        frames.append(screen.copy())
        
    pygame.display.flip()
    
    # Stop recording
    if recording:
        stop_recording()
    
    # Wait for user to close window
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
