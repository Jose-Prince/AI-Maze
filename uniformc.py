import heapq


N, S, E, W = 1, 2, 4, 8
DX = {E: 1, W: -1, N: 0, S:0}
DY = {E: 0, W: 0, N: -1, S: 1}
OPPOSITE = {E: W, W:E, N:S, S:N}

def ucs_solve_maze(grid, width, height, start, end):
    heap = []
    visited = set()
    parent = {}
    cost = {start: 0}
    nodes_explored = 0

    heapq.heappush(heap, (0, start))

    while heap:
        current_cost, current = heapq.heappop(heap)
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
                    new_cost = current_cost + 1  
                    if neighbor not in cost or new_cost < cost[neighbor]:
                        cost[neighbor] = new_cost
                        parent[neighbor] = current
                        heapq.heappush(heap, (new_cost, neighbor))


    path = []
    current = end
    while current != start:
        path.append(current)
        current = parent.get(current)
        if current is None:
            print("No path found!")
            return [], visited

    path.append(start)
    path.reverse()

    return path, visited
