import time
import random
import minTree

N, S, E, W = 1, 2, 4, 8
DX = {E: 1, W: -1, N: 0, S:0}
DY = {E: 0, W: 0, N: -1, S: 1}
OPPOSITE = {E: W, W:E, N:S, S:N}

def display_maze(width, grid, start=None, end=None):
    print("\033[H", end="")
    print(" " + "_" * (2 * width - 1))
    for y, row in enumerate(grid):
        print("|", end="")
        for x, cell in enumerate(row):
            current = (x, y)

            if current == start:
                print("\033[42m", end="")
            elif current == end:
                print("\033[41m", end="")
            elif cell == 0:
                print("\033[47m", end="")

            print(" " if (cell & S) else "_", end="")

            if cell & E:
                neighbor = row[x + 1] if x + 1 < width else 0
                print(" " if ((cell | neighbor) & S) else "_", end="")
            else:
                print("|", end="")

            if current in [start, end] or cell == 0:
                print("\033[m", end="")
        print()

def executeKruskalAlgorithm(width, height):

    
    seed = random.randint(0, 0xFFFF_FFFF)
    random.seed(seed)


    mst = minTree.MinimunSpanningTree()

    grid = [[0 for _ in range(width)] for _ in range(height)]
    edges = []

    for y in range(height):
        for x in range(width):
            mst.make_set((x, y))
            if y > 0: edges.append((x, y, N))
            if x > 0: edges.append((x, y, W))

    random.shuffle(edges)

    print('\033[2J', end="")
    debug = 0
    while edges:
        debug = debug + 1
        x, y, direction = edges.pop()
        nx, ny = x + DX[direction], y + DY[direction]

        if not mst.connected((x, y), (nx, ny)):
            
            if debug % 50 == 0:
                display_maze(width, grid)
                time.sleep(0.1)

            mst.union((x, y), (nx, ny))
            grid[y][x] |= direction
            grid[ny][nx] |= OPPOSITE[direction]
    
            mst.add_edge((x, y), (nx, ny))

    display_maze(width, grid, start=mst.start, end=mst.end)
    return grid, mst

#executeKruskalAlgorithm()
