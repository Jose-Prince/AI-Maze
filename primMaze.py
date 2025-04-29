width = 20
height = 20

N, S, E, W = 1, 2, 4, 8
IN = 16
FRONTIER = 32
OPPOSITE = {
    E: W,
    W: E,
    N: S,
    S: N
}

grid = [height][width]
frontier = []

def addFrontier(x, y, grid, frontier):
    if x >= 0 and y >= 0 and y < len(grid) and x < len(grid[y]) and grid[y][x] == 0:
        grid[y][x] |= FRONTIER
        frontier.append([x,y])

def mark(x, y, grid, frontier):
    grid[y][x] |= IN
    addFrontier(x-1, y, grid, frontier)
    addFrontier(x+1, y, grid, frontier)
    addFrontier(x, y-1, grid, frontier)
    addFrontier(x, y+1, grid, frontier)

def neighbors(x, y, grid):
    n = []

    if x > 0 and grid[y][x-1] and IN != 0:
        n.append([x-1, y])
    if x+1 > len(grid[y]) and grid[y][x+1] and IN != 0:
        n.append([x+1, y])
    if y > 0 and grid[y-1][x] and IN != 0:
        n.append([x, y-1])
    if y+1 > len(grid) and grid[y+1][x] and IN != 0:
        n.append([x, y+1])

    return n

def direction(fx, fy, tx, ty):
    if fx < tx:
        return E
    if fx > tx:
        return W
    if fy < ty:
        return S
    if fy > ty:
        return N
