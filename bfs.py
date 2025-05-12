from collections import deque
import time
from primMaze import display_maze
N, S, E, W = 1, 2, 4, 8
DX = {E: 1, W: -1, N: 0, S:0}
DY = {E: 0, W: 0, N: -1, S: 1}
OPPOSITE = {E: W, W:E, N:S, S:N}


def bfs_solve_maze(grid, width, height, start, end, mst, condicion):
    queue = deque()
    visited = set()
    parent = {}

    queue.append(start)
    visited.add(start)
    nodes_explored = 0

    while queue:
        current = queue.popleft()
        nodes_explored += 1

        if current == end:
            break

        x, y = current
        cell = grid[y][x]

        for dx, dy, direction in [ (0, -1, N), (0, 1, S), (-1, 0, W), (1, 0, E) ]:
            if cell & direction:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                if (0 <= nx < width and 0 <= ny < height and neighbor not in visited):
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)

    # Reconstruct path
    path = []
    current = end
    while current != start:
        path.append(current)
        current = parent.get(current)
        if current is None:
            print("No path found!")
            return [], visited
        if condicion:
            display_maze(width, height, grid, mst, visited, path)
            time.sleep(0.1)

        
    path.append(start)
    path.reverse()

    return path, visited
