import sys
import primMaze
import kruskalMaze
import bfs, dfs, astar, uniformc

if __name__ == "__main__":
    width = height = option = 0
    if len(sys.argv) == 2:
        width = 60
        height = 80
        option = 1
    elif len(sys.argv) == 4:
        width = int(sys.argv[1])
        height = int(sys.argv[2])
        option = int(sys.argv[3])
    else:
        print("To use the maze you may use this command: python main.py {width} {height} {maze_type}")
        print("\nFor the maze type just insert 1 (Prim algorithm) or 2 (Kruskal algorithm)")
        sys.exit(1)
    
    if option == 1:
        grid, mst, width, height = primMaze.executePrimAlgorithm(width, height)

        mst.start = (0,0)
        mst.end = (79,59)
        startpoint = mst.start
        endpoint = mst.end 

        algorithm = int(input("Select the algorithm:\n1: BFS \n2: DFS \n3: COST UNIFROM SEARCH \n4: A* "))

        if algorithm == 1:
            path, visited = bfs.bfs_solve_maze(grid, width, height, startpoint, endpoint, mst, True)
        elif algorithm == 2:
            path, visited = dfs.dfs_solve_maze(grid, width, height, startpoint, endpoint, mst, True)
        elif algorithm == 3:
            path, visited = uniformc.ucs_solve_maze(grid, width, height, startpoint, endpoint, mst, True)
        elif algorithm == 4:
            path, visited = astar.astar_solve_maze(grid, width, height, startpoint, endpoint, mst, True)
        else:
            print("Set default BFS")
            path, visited = bfs.bfs_solve_maze(grid, width, height, startpoint, endpoint, True)

        
        primMaze.display_maze(width, height, grid, mst, visited, path)

        print(f"\nPath length: {len(path)}")
        print(f"Nodes explored: {len(visited)}")

    else:

        grid, mst= kruskalMaze.executeKruskalAlgorithm(width, height)
        startpoint = mst.start
        endpoint = mst.end

        algorithm = int(input("Select the algorithm:\n1: BFS \n2: DFS \n3: COST UNIFROM SEARCH \n4: A* "))

        if algorithm == 1:
            path, visited = bfs.bfs_solve_maze(grid, width, height, startpoint, endpoint, mst)
        elif algorithm == 2:
            path, visited = dfs.dfs_solve_maze(grid, width, height, startpoint, endpoint)
        elif algorithm == 3:
            path, visited = uniformc.ucs_solve_maze(grid, width, height, startpoint, endpoint)
        elif algorithm == 4:
            path, visited = astar.astar_solve_maze(grid, width, height, startpoint, endpoint)
        else:
            print("Set default BFS")
            path, visited = bfs.bfs_solve_maze(grid, width, height, startpoint, endpoint)
        
        primMaze.display_maze(width, height, grid, mst, visited, path)

        print(f"\nPath length: {len(path)}")
        print(f"Nodes explored: {len(visited)}")
