from collections import defaultdict

# ==============================================
print("this is DFS algo")

def my_func(graph, src):
    node_list=[]
    dist = defaultdict(lambda: float("inf"))
    dist[src,src] = 0

    visitSet = set()
    def dfs(graph,node):
        visitSet.add(node)

        for v in graph[node]:
            if len(set(graph[node]).intersection(visitSet))>=1:
                 print ("there is a cycle")

            if v not in visitSet:
                dist[src,v] = min(dist[src,v], dist[src,node]+1)
                node_list.append(v)
                dfs(graph, v)
    dfs(graph, src)
    return node_list, dist

graph = {
    0: [1,2,3],
    1: [2,4],
    2: [3,5],
    3: [6,7],
    4: [],
    5: [],
    6:[7],
    7: []}

node_list, dist = my_func(graph, 0)
print(dist)
print(node_list)


