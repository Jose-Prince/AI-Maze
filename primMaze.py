import random
import time
import os

# Parámetros
width = 20
height = 20
seed = random.randint(0, 0xFFFF_FFFF)
random.seed(seed)

# Direcciones
N, S, E, W = 1, 2, 4, 8
IN = 16
FRONTIER = 32
OPPOSITE = { E: W, W: E, N: S, S: N }

# Grilla y frontera
grid = [[0 for _ in range(width)] for _ in range(height)]
frontier = []

def add_frontier(x, y):
    if 0 <= x < width and 0 <= y < height and grid[y][x] == 0:
        grid[y][x] |= FRONTIER
        frontier.append([x, y])

def mark(x, y):
    grid[y][x] |= IN
    add_frontier(x - 1, y)
    add_frontier(x + 1, y)
    add_frontier(x, y - 1)
    add_frontier(x, y + 1)

def neighbors(x, y):
    n = []
    if x > 0 and grid[y][x - 1] & IN != 0:
        n.append([x - 1, y])
    if x + 1 < width and grid[y][x + 1] & IN != 0:
        n.append([x + 1, y])
    if y > 0 and grid[y - 1][x] & IN != 0:
        n.append([x, y - 1])
    if y + 1 < height and grid[y + 1][x] & IN != 0:
        n.append([x, y + 1])
    return n

def direction(fx, fy, tx, ty):
    if fx < tx: return E
    if fx > tx: return W
    if fy < ty: return S
    if fy > ty: return N

def empty(cell):
    return cell == 0 or cell == FRONTIER

def display_maze():
    os.system("cls" if os.name == "nt" else "clear")
    print(" " + "_" * (width * 2 - 1))
    for y in range(height):
        line = "|"
        for x in range(width):
            cell = grid[y][x]
            if empty(cell) and y + 1 < height and empty(grid[y + 1][x]):
                line += " "
            else:
                line += " " if cell & S != 0 else "_"
            if empty(cell) and x + 1 < width and empty(grid[y][x + 1]):
                if y + 1 < height and (empty(grid[y + 1][x]) or empty(grid[y + 1][x + 1])):
                    line += " "
                else:
                    line += "_"
            elif cell & E != 0:
                if (cell | grid[y][x + 1]) & S != 0:
                    line += " "
                else:
                    line += "_"
            else:
                line += "|"
        print(line)

treeEdges = []

# Algoritmo de Prim
mark(random.randint(0, width - 1), random.randint(0, height - 1))
while frontier:
    x, y = frontier.pop(random.randint(0, len(frontier) - 1))
    n = neighbors(x, y)
    if not n:
        continue
    nx, ny = n[random.randint(0, len(n) - 1)]
    dir = direction(x, y, nx, ny)
    grid[y][x] |= dir
    if dir != None:
        grid[ny][nx] |= OPPOSITE[dir]

    treeEdges.append(((x, y), (nx, ny)))

    mark(x, y)
    display_maze()
    time.sleep(0.01)

# Mostrar laberinto final y semilla
display_maze()
print(f"python prim_maze.py {width} {height} {seed}")

