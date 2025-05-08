class MinimunSpanningTree:
    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.edges = []
        self.nodes = set()
        self.start = None
        self.end = None

    def make_set(self, cell):
        self.parent[cell] = cell
        self.rank[cell] = 0
        if self.start is None:
            self.start = cell
        self.end = cell
    
    def find(self, cell):
        if self.parent[cell] != cell:
            self.parent[cell] = self.find(self.parent[cell])
        return self.parent[cell]

    def union(self, cell1, cell2):
        root1 = self.find(cell1)
        root2 = self.find(cell2)
        if root1 == root2:
            return False
        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        else:
            self.parent[root2] = root1
            if self.rank[root1] == self.rank[root2]:
                self.rank[root1] += 1
        return True

    def connected(self, cell1, cell2):
        return self.find(cell1) == self.find(cell2)

    def add_edge(self, origin, destination):
        self.edges.append((origin, destination))
        self.nodes.update([origin, destination])

    def __repr__(self):
        s = f"Start: {self.start}, End: {self.end}\n"
        s += "\n".join(f"{a} -> {b}" for a, b in self.edges)
        return s
