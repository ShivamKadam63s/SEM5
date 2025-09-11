import pygame
import heapq
import math
import time
import sys
import os
from datetime import datetime

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
pygame.display.set_caption("A* Pathfinding Demo")
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

# Node class for A* algorithm
class Node:
    def __init__(self, position, parent=None):
        self.position = position  # (x, y)
        self.parent = parent
        self.g = 0  # Cost from start to this node
        self.h = 0  # Heuristic to goal
        self.f = 0  # Total cost f = g + h

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f  # For heapq priority

# Function to get valid neighbors (up, down, left, right)
def get_neighbors(grid, node):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, down, left, up
    neighbors = []
    for dx, dy in directions:
        nx, ny = node.position[0] + dx, node.position[1] + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[ny][nx] == 0:  # Valid and not obstacle
            neighbors.append(Node((nx, ny)))
    return neighbors

# Heuristic function: Euclidean distance
def heuristic(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

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
    log_filename = f"recordings/astar_log_{timestamp}.txt"
    log_file = open(log_filename, "w")
    
    log_message("A* Pathfinding Demo")
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
        filename = f"recordings/astar_demo_{timestamp}.mp4"
        
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
def draw_grid(screen, grid, open_set, closed_set, current, path, start, goal, current_node_info=None):
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
            elif (x, y) in closed_set:  # Closed set
                pygame.draw.rect(screen, GRAY, rect)
            elif any(node.position == (x, y) for _, _, node in open_set):  # Open set
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
        (BLUE, "Nodes Being Considered"),
        (GRAY, "Evaluated Nodes"),
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
    open_set_text = font.render(f"Open Set: {len(open_set)} nodes", True, LIGHT_GRAY)
    closed_set_text = font.render(f"Closed Set: {len(closed_set)} nodes", True, LIGHT_GRAY)
    path_text = font.render(f"Path Length: {len(path) if path else 0} nodes", True, LIGHT_GRAY)
    
    screen.blit(open_set_text, (20, WINDOW_WIDTH + 20))
    screen.blit(closed_set_text, (20, WINDOW_WIDTH + 50))
    screen.blit(path_text, (20, WINDOW_WIDTH + 80))
    
    # Draw current node info in the middle if available
    if current_node_info:
        current_text = font.render(f"Current Node: {current_node_info['position']}", True, LIGHT_GRAY)
        g_text = font.render(f"g(n): {current_node_info['g']} (cost from start)", True, LIGHT_GRAY)
        h_text = font.render(f"h(n): {current_node_info['h']:.2f} (heuristic)", True, LIGHT_GRAY)
        f_text = font.render(f"f(n): {current_node_info['f']:.2f} (g + h)", True, LIGHT_GRAY)
        
        screen.blit(current_text, (TOTAL_WIDTH // 3, WINDOW_WIDTH + 20))
        screen.blit(g_text, (TOTAL_WIDTH // 3, WINDOW_WIDTH + 45))
        screen.blit(h_text, (TOTAL_WIDTH // 3, WINDOW_WIDTH + 70))
        screen.blit(f_text, (TOTAL_WIDTH // 3, WINDOW_WIDTH + 95))
    
    # Draw recording status
    status_text = font.render("Recording: Yes", True, LIGHT_GRAY)
    screen.blit(status_text, (TOTAL_WIDTH - 150, WINDOW_WIDTH + 20))

# Main A* function with step-by-step visualization
def a_star(grid, start_pos, goal_pos, screen):
    start = Node(start_pos)
    goal = Node(goal_pos)

    open_set = []  # Priority queue: (f, -g (for tie-breaking), node)
    heapq.heappush(open_set, (0, 0, start))
    open_set_hash = {start.position: start}  # For quick lookup
    closed_set = set()  # Positions in closed set

    path = []  # Final path
    current_node_info = None

    # Capture initial state
    draw_grid(screen, grid, open_set, closed_set, None, path, start_pos, goal_pos)
    if recording:
        # Create a copy of the screen surface for the frame
        frames.append(screen.copy())
    pygame.display.flip()

    while open_set:
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

        # Print the formula and values to console and log file
        log_message("\n" + "="*50)
        log_message("f(n) = g(n) + h(n)")
        log_message("Open set nodes:")
        for f, _, node in sorted(open_set):  # Sort for consistent printing
            log_message(f"Node {node.position}: g={node.g}, h={node.h:.2f}, f={node.f:.2f}")

        # Get node with lowest f
        current_f, _, current = heapq.heappop(open_set)
        del open_set_hash[current.position]
        log_message(f"Chosen node to expand next: {current.position}")

        # Add to closed set
        closed_set.add(current.position)

        # Prepare current node info for display
        current_node_info = {
            'position': current.position,
            'g': current.g,
            'h': current.h,
            'f': current.f
        }

        # Visualize current state (current path is back to start via parents)
        current_path = reconstruct_path(current)
        draw_grid(screen, grid, open_set, closed_set, current, current_path, start_pos, goal_pos, current_node_info)
        
        # Capture frame if recording
        if recording:
            # Create a copy of the screen surface for the frame
            frames.append(screen.copy())
        
        pygame.display.flip()
        time.sleep(0.1)  # Delay for visualization

        # If goal reached
        if current.position == goal_pos:
            path = reconstruct_path(current)
            log_message("Path found: " + str(path))
            # Keep the open and closed sets visible
            draw_grid(screen, grid, open_set, closed_set, None, path, start_pos, goal_pos, current_node_info)
            
            # Capture final frame if recording
            if recording:
                frames.append(screen.copy())
                
            pygame.display.flip()
            return path, open_set, closed_set

        # Expand neighbors
        for neighbor in get_neighbors(grid, current):
            if neighbor.position in closed_set:
                continue

            tentative_g = current.g + 1  # Cost per step = 1

            if neighbor.position not in open_set_hash or tentative_g < open_set_hash[neighbor.position].g:
                neighbor.parent = current
                neighbor.g = tentative_g
                neighbor.h = heuristic(neighbor.position, goal.position)
                neighbor.f = neighbor.g + neighbor.h

                if neighbor.position not in open_set_hash:
                    heapq.heappush(open_set, (neighbor.f, -neighbor.g, neighbor))  # -g for tie-breaking (prefer higher g)
                    open_set_hash[neighbor.position] = neighbor

    log_message("No path found")
    return None, open_set, closed_set

# Create a more interesting maze
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
    
    # Run A*
    path, open_set, closed_set = a_star(grid, start_pos, goal_pos, screen)
    
    # Draw final result with open and closed sets still visible
    draw_grid(screen, grid, open_set, closed_set, None, path, start_pos, goal_pos)
    
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
