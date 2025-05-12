import heapq
import time
from primMaze import display_maze

N, S, E, W = 1, 2, 4, 8
DX = {E: 1, W: -1, N: 0, S:0}
DY = {E: 0, W: 0, N: -1, S: 1}
OPPOSITE = {E: W, W:E, N:S, S:N}


def manhattan_distance(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def astar_solve_maze(grid, width, height, start, end, mst, condicion):
    heap = []
    visited = set()
    parent = {}
    g_cost = {start: 0}
    nodes_explored = 0

    heapq.heappush(heap, (manhattan_distance(start, end), 0, start))

    while heap:
        f_cost, current_g, current = heapq.heappop(heap)
        nodes_explored += 1

        if current == end:
            break

        if current in visited:
            continue
        visited.add(current)

        x, y = current
        cell = grid[y][x]

        for dx, dy, direction in [(0, -1, N), (0, 1, S), (-1, 0, W), (1, 0, E)]:
            if cell & direction:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)
                if 0 <= nx < width and 0 <= ny < height:
                    new_g = current_g + 1
                    if neighbor not in g_cost or new_g < g_cost[neighbor]:
                        g_cost[neighbor] = new_g
                        f = new_g + manhattan_distance(neighbor, end)
                        parent[neighbor] = current
                        heapq.heappush(heap, (f, new_g, neighbor))

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
