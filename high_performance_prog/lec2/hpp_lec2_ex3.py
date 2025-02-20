from collections import deque

# -------------------------------
# 1) Define the 4×4 "processors"
# -------------------------------
# We'll keep them in a 2D grid. For convenience, define the local values:
V = [3, 6, 1, 5, 
     7, 4, 2, 9,
     8, 3, 5, 1,
     4, 2, 6, 7]

def value_at(row, col):
    """Return the local value at processor (row, col)."""
    return V[row * 4 + col]

# -------------------------------
# 2) Helper: get valid neighbors
# -------------------------------
def neighbors(r, c):
    """Return the valid neighbors of (r,c) in a 4×4 mesh."""
    nbrs = []
    for (nr, nc) in [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]:
        if 0 <= nr < 4 and 0 <= nc < 4:
            nbrs.append((nr, nc))
    return nbrs

# -----------------------------------------
# 3) Build a BFS tree (for broadcast & reduction)
# -----------------------------------------
def build_bfs_tree(src=(0,0)):
    """
    Returns 'parent' dict storing the BFS tree from src.
    parent[node] = the node's parent in the BFS spanning tree
    """
    visited = set([src])
    parent = {src: None}  # root has no parent
    queue = deque([src])
    
    while queue:
        curr = queue.popleft()
        for nbr in neighbors(*curr):
            if nbr not in visited:
                visited.add(nbr)
                parent[nbr] = curr  # record BFS parent
                queue.append(nbr)
    
    return parent

parent_map = build_bfs_tree(src=(0,0))

# ------------------------------------------------
# 4) Simulate the one-to-all BROADCAST in levels
# ------------------------------------------------
# We'll keep track of "which nodes have the message" at each step.

has_message = set([(0,0)])     # initially, only (0,0) has the message
wave_front = [(0,0)]          # the "front" of the broadcast wave
steps_broadcast = 0

while wave_front:
    steps_broadcast += 1
    next_wave = []
    
    # Each processor in wave_front sends to neighbors
    for node in wave_front:
        for nbr in neighbors(*node):
            if nbr not in has_message:
                has_message.add(nbr)
                next_wave.append(nbr)
    
    wave_front = next_wave

# After this loop, all 16 processors have the message.
print(f"Broadcast completed in {steps_broadcast} 'hops' (store-and-forward steps).")

# ------------------------------------------------
# 5) Simulate the ALL-TO-ONE REDUCTION in levels
# ------------------------------------------------
# We'll gather sums from the leaves back to the root (0,0)
# using the BFS tree we found above.

# Step 5A: Each processor "knows" its own local sum initially.
partial_sum = {}
for r in range(4):
    for c in range(4):
        partial_sum[(r,c)] = value_at(r, c)

# Step 5B: We can do a bottom-up approach: 
#   - find the farthest BFS levels 
#   - at each level, each node sends its partial_sum to its parent
#   - the parent adds it in.

# Let's figure out BFS distances so we can process from farthest to closest
dist = {}
def bfs_distances(src=(0,0)):
    # Standard BFS to assign distance from src
    queue = deque([src])
    dist[src] = 0
    visited = {src}
    while queue:
        curr = queue.popleft()
        for nbr in neighbors(*curr):
            if nbr not in visited:
                visited.add(nbr)
                dist[nbr] = dist[curr] + 1
                queue.append(nbr)

bfs_distances((0,0))

# Now we know how far each node is from (0,0).
# The maximum distance = diameter = 6 for a corner in a 4×4, 
# so we'll do 6 "hops" in reverse to simulate the partial-sum wave.

steps_reduction = 0
max_dist = max(dist.values())
for d in range(max_dist, 0, -1):
    steps_reduction += 1
    # All nodes that are distance d will send their sum to their parent (distance d-1).
    for node in dist:
        if dist[node] == d:
            # send partial_sum[node] to its parent
            p = parent_map[node]
            partial_sum[p] += partial_sum[node]
            # node's sum is now "sent"; in a real concurrency scenario,
            # node might keep a local copy, but the key is it's been reduced upward.

print(f"Reduction completed in {steps_reduction} 'hops' (store-and-forward steps).")

# The final total is at (0,0):
print(f"Final SUM at (0,0) = {partial_sum[(0,0)]}.")

# Just for clarity, let's verify that sum matches Python's built-in sum:
print(f"Check with builtin sum: {sum(V)}.")
