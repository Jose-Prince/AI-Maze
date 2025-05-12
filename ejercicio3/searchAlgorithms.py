import numpy as np
import matplotlib.pyplot as plt
import time
import heapq
from collections import deque
import pandas as pd
import random
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Constantes
HEIGHT = 45
WIDTH = 55
NUM_MAZES = 25
MIN_MANHATTAN_DISTANCE = 10

# Definición de colores
WALL_COLOR = 0  # Negro
PATH_COLOR = 1  # Blanco
START_COLOR = 2  # Verde
END_COLOR = 3  # Rojo
SOLUTION_COLOR = 4  # Azul
BFS_EXPLORED_COLOR = 5  # Celeste claro
DFS_EXPLORED_COLOR = 6  # Naranja claro
DIJKSTRA_EXPLORED_COLOR = 7  # Morado claro
ASTAR_EXPLORED_COLOR = 8  # Amarillo claro

# Mapa de colores para visualización
color_map = ListedColormap(['black', 'white', 'green', 'red', 'blue', 
                          'skyblue', 'orange', 'purple', 'yellow'])

class Maze:
    def __init__(self, height=HEIGHT, width=WIDTH, wall_probability=0.3):
        self.height = height
        self.width = width
        self.grid = np.ones((height, width), dtype=int)  # Inicialmente todo es camino (1)
        self.start = None
        self.end = None
        self.generate_maze(wall_probability)
        self.select_start_end_points()
    
    def generate_maze(self, wall_probability):
        # Generar laberinto aleatorio
        for i in range(self.height):
            for j in range(self.width):
                # Mantener bordes como paredes
                if i == 0 or i == self.height - 1 or j == 0 or j == self.width - 1:
                    self.grid[i, j] = WALL_COLOR
                # Para el resto, decidir aleatoriamente
                elif random.random() < wall_probability:
                    self.grid[i, j] = WALL_COLOR
    
    def is_valid_position(self, pos):
        i, j = pos
        return (0 <= i < self.height and 
                0 <= j < self.width and 
                self.grid[i, j] != WALL_COLOR)
    
    def select_start_end_points(self):
        valid_positions = []
        
        # Recopilar todas las posiciones válidas
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] != WALL_COLOR:
                    valid_positions.append((i, j))
        
        # Si no hay suficientes posiciones válidas, reintentar con menos paredes
        if len(valid_positions) < 2:
            self.generate_maze(wall_probability=0.2)
            self.select_start_end_points()
            return
        
        # Intentar encontrar puntos con distancia Manhattan >= MIN_MANHATTAN_DISTANCE
        max_attempts = 1000
        attempts = 0
        
        while attempts < max_attempts:
            start = random.choice(valid_positions)
            end = random.choice(valid_positions)
            
            # Calcular distancia Manhattan
            manhattan_dist = abs(start[0] - end[0]) + abs(start[1] - end[1])
            
            if manhattan_dist >= MIN_MANHATTAN_DISTANCE and start != end:
                self.start = start
                self.end = end
                
                # Marcar los puntos en el grid
                self.grid[start] = START_COLOR
                self.grid[end] = END_COLOR
                return
            
            attempts += 1
        
        # Si no se encontraron puntos adecuados después de muchos intentos,
        # seleccionar los dos más distantes entre sí
        max_dist = 0
        best_pair = (valid_positions[0], valid_positions[1])
        
        for start in valid_positions:
            for end in valid_positions:
                if start != end:
                    dist = abs(start[0] - end[0]) + abs(start[1] - end[1])
                    if dist > max_dist:
                        max_dist = dist
                        best_pair = (start, end)
        
        self.start = best_pair[0]
        self.end = best_pair[1]
        
        # Marcar los puntos en el grid
        self.grid[self.start] = START_COLOR
        self.grid[self.end] = END_COLOR
    
    def get_neighbors(self, pos):
        i, j = pos
        neighbors = []
        
        # Vecinos en 4 direcciones: arriba, derecha, abajo, izquierda
        for ni, nj in [(i-1, j), (i, j+1), (i+1, j), (i, j-1)]:
            if self.is_valid_position((ni, nj)):
                neighbors.append((ni, nj))
        
        return neighbors
    
    def visualize(self, solution_path=None, explored_nodes=None, algorithm_name=None):
        """Visualiza el laberinto con opción para mostrar la solución y nodos explorados"""
        plt.figure(figsize=(8, 7))
        
        # Hacer una copia del grid para no modificar el original
        visualization_grid = self.grid.copy()
        
        # Marcar nodos explorados si se proporcionan
        if explored_nodes:
            if algorithm_name == "BFS":
                explored_color = BFS_EXPLORED_COLOR
            elif algorithm_name == "DFS":
                explored_color = DFS_EXPLORED_COLOR
            elif algorithm_name == "Dijkstra":
                explored_color = DIJKSTRA_EXPLORED_COLOR
            elif algorithm_name == "A*":
                explored_color = ASTAR_EXPLORED_COLOR
            else:
                explored_color = BFS_EXPLORED_COLOR  # Color por defecto
            
            for node in explored_nodes:
                if node != self.start and node != self.end:
                    visualization_grid[node] = explored_color
        
        # Marcar la ruta de la solución si se proporciona
        if solution_path:
            for node in solution_path:
                if node != self.start and node != self.end:
                    visualization_grid[node] = SOLUTION_COLOR
        
        # Asegurar que start y end sean visibles
        visualization_grid[self.start] = START_COLOR
        visualization_grid[self.end] = END_COLOR
        
        # Crear leyenda
        legend_elements = [
            mpatches.Patch(color='black', label='Pared'),
            mpatches.Patch(color='white', label='Camino'),
            mpatches.Patch(color='green', label='Inicio'),
            mpatches.Patch(color='red', label='Fin'),
            mpatches.Patch(color='blue', label='Solución')
        ]
        
        if explored_nodes:
            if algorithm_name == "BFS":
                legend_elements.append(mpatches.Patch(color='skyblue', label='Explorado BFS'))
            elif algorithm_name == "DFS":
                legend_elements.append(mpatches.Patch(color='orange', label='Explorado DFS'))
            elif algorithm_name == "Dijkstra":
                legend_elements.append(mpatches.Patch(color='purple', label='Explorado Dijkstra'))
            elif algorithm_name == "A*":
                legend_elements.append(mpatches.Patch(color='yellow', label='Explorado A*'))
        
        plt.imshow(visualization_grid, cmap=color_map)
        plt.title(f"Laberinto {self.height}x{self.width}" +
                 (f" - {algorithm_name}" if algorithm_name else ""))
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

class SearchAlgorithms:
    @staticmethod
    def bfs(maze):
        """Breadth-First Search"""
        start_time = time.time()
        
        start = maze.start
        end = maze.end
        
        # Cola para BFS
        queue = deque([start])
        
        # Conjunto para rastrear nodos visitados
        visited = set([start])
        
        # Diccionario para rastrear padres para reconstrucción de ruta
        parent = {start: None}
        
        while queue:
            current = queue.popleft()
            
            # Si hemos llegado al objetivo
            if current == end:
                break
            
            # Explorar vecinos
            for neighbor in maze.get_neighbors(current):
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)
                    parent[neighbor] = current
        
        # Reconstruir ruta si se encontró una solución
        path = []
        if end in parent:
            current = end
            while current:
                path.append(current)
                current = parent[current]
            path.reverse()
        
        end_time = time.time()
        
        return {
            "algorithm": "BFS",
            "path": path,
            "path_length": len(path) - 1 if path else 0,  # -1 porque no contamos el nodo inicial
            "explored_nodes": list(visited),
            "num_explored": len(visited),
            "execution_time": end_time - start_time,
            "found_solution": len(path) > 0
        }
    
    @staticmethod
    def dfs(maze):
        """Depth-First Search"""
        start_time = time.time()
        
        start = maze.start
        end = maze.end
        
        # Pila para DFS (implementada con lista)
        stack = [start]
        
        # Conjunto para rastrear nodos visitados
        visited = set([start])
        
        # Diccionario para rastrear padres para reconstrucción de ruta
        parent = {start: None}
        
        while stack:
            current = stack.pop()
            
            # Si hemos llegado al objetivo
            if current == end:
                break
            
            # Explorar vecinos
            for neighbor in maze.get_neighbors(current):
                if neighbor not in visited:
                    stack.append(neighbor)
                    visited.add(neighbor)
                    parent[neighbor] = current
        
        # Reconstruir ruta si se encontró una solución
        path = []
        if end in parent:
            current = end
            while current:
                path.append(current)
                current = parent[current]
            path.reverse()
        
        end_time = time.time()
        
        return {
            "algorithm": "DFS",
            "path": path,
            "path_length": len(path) - 1 if path else 0,
            "explored_nodes": list(visited),
            "num_explored": len(visited),
            "execution_time": end_time - start_time,
            "found_solution": len(path) > 0
        }
    
    @staticmethod
    def dijkstra(maze):
        """Dijkstra's algorithm (Uniform Cost Search)"""
        start_time = time.time()
        
        start = maze.start
        end = maze.end
        
        # Cola de prioridad para Dijkstra
        priority_queue = [(0, start)]
        
        # Conjunto para rastrear nodos visitados
        visited = set()
        
        # Diccionario para rastrear distancias
        distances = {start: 0}
        
        # Diccionario para rastrear padres para reconstrucción de ruta
        parent = {start: None}
        
        while priority_queue:
            # Obtener nodo con menor costo
            current_distance, current = heapq.heappop(priority_queue)
            
            # Si ya procesamos este nodo, continuar
            if current in visited:
                continue
            
            visited.add(current)
            
            # Si hemos llegado al objetivo
            if current == end:
                break
            
            # Explorar vecinos
            for neighbor in maze.get_neighbors(current):
                # Costo uniforme de 1 para cada paso
                tentative_distance = distances[current] + 1
                
                # Si encontramos un camino más corto
                if neighbor not in distances or tentative_distance < distances[neighbor]:
                    distances[neighbor] = tentative_distance
                    heapq.heappush(priority_queue, (tentative_distance, neighbor))
                    parent[neighbor] = current
        
        # Reconstruir ruta si se encontró una solución
        path = []
        if end in parent:
            current = end
            while current:
                path.append(current)
                current = parent[current]
            path.reverse()
        
        end_time = time.time()
        
        return {
            "algorithm": "Dijkstra",
            "path": path,
            "path_length": len(path) - 1 if path else 0,
            "explored_nodes": list(visited),
            "num_explored": len(visited),
            "execution_time": end_time - start_time,
            "found_solution": len(path) > 0
        }
    
    @staticmethod
    def astar(maze):
        """A* Search"""
        start_time = time.time()
        
        start = maze.start
        end = maze.end
        
        # Función heurística: distancia Manhattan
        def heuristic(pos):
            return abs(pos[0] - end[0]) + abs(pos[1] - end[1])
        
        # Cola de prioridad para A*
        priority_queue = [(heuristic(start), 0, start)]  # (f, g, pos)
        
        # Conjunto para rastrear nodos visitados
        visited = set()
        
        # Diccionario para rastrear valores g (costo desde inicio)
        g_values = {start: 0}
        
        # Diccionario para rastrear valores f (g + heurística)
        f_values = {start: heuristic(start)}
        
        # Diccionario para rastrear padres para reconstrucción de ruta
        parent = {start: None}
        
        while priority_queue:
            # Obtener nodo con menor valor f
            current_f, current_g, current = heapq.heappop(priority_queue)
            
            # Si ya procesamos este nodo, continuar
            if current in visited:
                continue
            
            visited.add(current)
            
            # Si hemos llegado al objetivo
            if current == end:
                break
            
            # Explorar vecinos
            for neighbor in maze.get_neighbors(current):
                # Costo desde el inicio hasta el vecino a través del nodo actual
                tentative_g = current_g + 1
                
                # Si encontramos un camino más corto o no hemos visitado este nodo
                if neighbor not in g_values or tentative_g < g_values[neighbor]:
                    g_values[neighbor] = tentative_g
                    f_values[neighbor] = tentative_g + heuristic(neighbor)
                    heapq.heappush(priority_queue, (f_values[neighbor], tentative_g, neighbor))
                    parent[neighbor] = current
        
        # Reconstruir ruta si se encontró una solución
        path = []
        if end in parent:
            current = end
            while current:
                path.append(current)
                current = parent[current]
            path.reverse()
        
        end_time = time.time()
        
        return {
            "algorithm": "A*",
            "path": path,
            "path_length": len(path) - 1 if path else 0,
            "explored_nodes": list(visited),
            "num_explored": len(visited),
            "execution_time": end_time - start_time,
            "found_solution": len(path) > 0
        }

def run_experiments():
    # Lista para almacenar resultados
    all_results = []

    for maze_num in range(NUM_MAZES):
        print(f"Generando y analizando laberinto {maze_num + 1}/{NUM_MAZES}...")
        
        # Crear laberinto
        maze = Maze()
        
        # Ejecutar cada algoritmo
        bfs_result = SearchAlgorithms.bfs(maze)
        dfs_result = SearchAlgorithms.dfs(maze)
        dijkstra_result = SearchAlgorithms.dijkstra(maze)
        astar_result = SearchAlgorithms.astar(maze)
        
        # Almacenar resultados
        maze_results = {
            "maze_num": maze_num + 1,
            "BFS": bfs_result,
            "DFS": dfs_result,
            "Dijkstra": dijkstra_result,
            "A*": astar_result
        }
        all_results.append(maze_results)
        
        # Visualizar resultados del primer laberinto
        if maze_num == 0:
            # Visualizar el laberinto base
            maze.visualize()
            
            # Visualizar soluciones de cada algoritmo
            for algorithm_name, result in [("BFS", bfs_result), 
                                         ("DFS", dfs_result), 
                                         ("Dijkstra", dijkstra_result), 
                                         ("A*", astar_result)]:
                if result["found_solution"]:
                    maze.visualize(solution_path=result["path"], 
                                  explored_nodes=result["explored_nodes"],
                                  algorithm_name=algorithm_name)
                else:
                    print(f"El algoritmo {algorithm_name} no encontró solución en el laberinto {maze_num + 1}")
        
        # Crear tabla de comparación para este laberinto
        algorithms = ["BFS", "DFS", "Dijkstra", "A*"]
        comparison_data = []
        
        for algo in algorithms:
            result = maze_results[algo]
            comparison_data.append({
                "Algoritmo": algo,
                "Nodos Explorados": result["num_explored"],
                "Tiempo (s)": result["execution_time"],
                "Longitud Ruta": result["path_length"],
                "Encontró Solución": "Sí" if result["found_solution"] else "No"
            })
        
        # Convertir a DataFrame y ordenar por nodos explorados
        df = pd.DataFrame(comparison_data)
        print(f"\nComparación de algoritmos para el laberinto {maze_num + 1}:")
        print(df)
        print("\n")
    
    return all_results

def analyze_results(all_results):
    # Crear dataframes para análisis
    explored_nodes_data = []
    execution_time_data = []
    path_length_data = []
    
    for maze_result in all_results:
        maze_num = maze_result["maze_num"]
        
        for algo in ["BFS", "DFS", "Dijkstra", "A*"]:
            result = maze_result[algo]
            
            explored_nodes_data.append({
                "Laberinto": maze_num,
                "Algoritmo": algo,
                "Nodos Explorados": result["num_explored"]
            })
            
            execution_time_data.append({
                "Laberinto": maze_num,
                "Algoritmo": algo,
                "Tiempo (s)": result["execution_time"]
            })
            
            path_length_data.append({
                "Laberinto": maze_num,
                "Algoritmo": algo,
                "Longitud Ruta": result["path_length"] if result["found_solution"] else np.nan
            })
    
    # Convertir a DataFrames
    explored_df = pd.DataFrame(explored_nodes_data)
    time_df = pd.DataFrame(execution_time_data)
    path_df = pd.DataFrame(path_length_data)
    
    # Calcular promedios por algoritmo
    avg_explored = explored_df.groupby("Algoritmo")["Nodos Explorados"].mean().reset_index()
    avg_time = time_df.groupby("Algoritmo")["Tiempo (s)"].mean().reset_index()
    avg_path = path_df.groupby("Algoritmo")["Longitud Ruta"].mean().reset_index()
    
    # Fusionar resultados
    avg_results = pd.merge(avg_explored, avg_time, on="Algoritmo")
    avg_results = pd.merge(avg_results, avg_path, on="Algoritmo")
    
    # Calcular ranking de cada algoritmo para cada métrica
    avg_results["Ranking Nodos"] = avg_results["Nodos Explorados"].rank()
    avg_results["Ranking Tiempo"] = avg_results["Tiempo (s)"].rank()
    avg_results["Ranking Ruta"] = avg_results["Longitud Ruta"].rank()
    
    # Calcular ranking promedio
    avg_results["Ranking Promedio"] = (avg_results["Ranking Nodos"] + 
                                      avg_results["Ranking Tiempo"] + 
                                      avg_results["Ranking Ruta"]) / 3
    
    # Ordenar por ranking promedio
    avg_results = avg_results.sort_values("Ranking Promedio")
    
    return avg_results

def visualize_comparisons(all_results):
    # Extraer datos para visualización
    explored_data = []
    time_data = []
    path_data = []
    
    for maze_result in all_results:
        for algo in ["BFS", "DFS", "Dijkstra", "A*"]:
            result = maze_result[algo]
            
            explored_data.append({
                "Algoritmo": algo,
                "Nodos Explorados": result["num_explored"]
            })
            
            time_data.append({
                "Algoritmo": algo,
                "Tiempo (s)": result["execution_time"]
            })
            
            if result["found_solution"]:
                path_data.append({
                    "Algoritmo": algo,
                    "Longitud Ruta": result["path_length"]
                })
    
    # Convertir a DataFrames
    explored_df = pd.DataFrame(explored_data)
    time_df = pd.DataFrame(time_data)
    path_df = pd.DataFrame(path_data)
    
    # Visualizar nodos explorados
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    boxplot = plt.boxplot([explored_df[explored_df["Algoritmo"] == algo]["Nodos Explorados"] 
                          for algo in ["BFS", "DFS", "Dijkstra", "A*"]],
                         labels=["BFS", "DFS", "Dijkstra", "A*"])
    plt.title("Nodos Explorados por Algoritmo")
    plt.ylabel("Número de Nodos")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Visualizar tiempos de ejecución
    plt.subplot(1, 3, 2)
    boxplot = plt.boxplot([time_df[time_df["Algoritmo"] == algo]["Tiempo (s)"] 
                          for algo in ["BFS", "DFS", "Dijkstra", "A*"]],
                         labels=["BFS", "DFS", "Dijkstra", "A*"])
    plt.title("Tiempo de Ejecución por Algoritmo")
    plt.ylabel("Tiempo (s)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Visualizar longitudes de ruta
    plt.subplot(1, 3, 3)
    boxplot = plt.boxplot([path_df[path_df["Algoritmo"] == algo]["Longitud Ruta"] 
                          for algo in ["BFS", "DFS", "Dijkstra", "A*"]],
                         labels=["BFS", "DFS", "Dijkstra", "A*"])
    plt.title("Longitud de Ruta por Algoritmo")
    plt.ylabel("Longitud")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def main():
    print("Iniciando experimentos de comparación de algoritmos de búsqueda...")
    
    # Ejecutar experimentos
    all_results = run_experiments()
    
    # Analizar resultados
    avg_results = analyze_results(all_results)
    
    # Imprimir tabla resumen
    print("\n=== TABLA RESUMEN DE ALGORITMOS ===")
    print(avg_results[["Algoritmo", "Nodos Explorados", "Tiempo (s)", 
                     "Longitud Ruta", "Ranking Promedio"]])
    
    # Visualizar comparaciones
    visualize_comparisons(all_results)
    
    print("\nExperimentos completados.")

if __name__ == "__main__":
    main()