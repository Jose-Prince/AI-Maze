import sys
import primMaze
import kruskalMaze

if __name__ == "__main__":
    width = height = option = 0
    if len(sys.argv) == 2:
        width = 20
        height = 20
        option = 1
    elif len(sys.argv) == 4:
        ancho = int(sys.argv[1])
        alto = int(sys.argv[2])
        option = int(sys.argv[3])
    else:
        print("To use the maze you may use this command: python main.py {width} {height} {maze_type}")
        print("\nFor the maze type just insert 1 (Prim algorithm) or 2 (Kruskal algorithm)")
        sys.exit(1)
    
    if option == 1:
        primMaze.executePrimAlgorithm()
    else:
        kruskalMaze.executeKruskalAlgorithm()
