import time
import random

def bfs(graph, start):
    # Initialization
    num_nodes = len(graph)
    distance = [float('inf')] * num_nodes
    visited = [False] * num_nodes
    queue = []

    distance[start] = 0
    visited[start] = True
    queue.append(start)

    while queue:
        u = queue.pop(0)  # dequeue
        for w in graph[u]:
            if not visited[w]:
                visited[w] = True
                distance[w] = distance[u] + 1
                queue.append(w)  # enqueue

    return distance

# Generate a random undirected graph as an adjacency list
def generate_graph(num_nodes, edge_prob):
    graph = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:
                graph[i].append(j)
                graph[j].append(i)
    return graph

if __name__ == "__main__":
    num_nodes = 1000
    edge_prob = 0.01
    start_node = 0

    graph = generate_graph(num_nodes, edge_prob)

    start_time = time.time()
    distances = bfs(graph, start_node)
    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"BFS from node {start_node} completed.")
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
