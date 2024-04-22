import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.nx_agraph import to_agraph
import random
import numpy as np
from collections import defaultdict

class GraphInstance:
    def __init__(self,edges=None,blockages=None,generate_graph=True,pos=None) :
        self.repr = defaultdict(lambda : [])
        for k,v in edges.items():
            self.repr[k[0]].append((k[1],v))
                                   
        self.costs = edges
        self.edges = list(edges.keys())
        self.pos = pos
        
        self.blockages = None
        if blockages != None:
            self.blockages = list(set(blockages))
        self.G = None
        
        self.nodes = set()
        for (i,j) in self.edges:
            self.nodes.add(i)
            self.nodes.add(j)
        self.nodes = list(self.nodes)
        
        if generate_graph:
            self.generate_graph()
        self.Time = 0
        self.mapp = {i:n for (i,n) in enumerate(self.nodes)}
        self.inv_map = {n:i for (i,n) in enumerate(self.nodes)}
    

        

    def generate_graph(self):
        self.G = self.getGraph()
        
    def getGraph(self):
        if self.G == None:
            self.G = nx.Graph()
            self.G.add_edges_from(self.edges)
        return self.G

                                   
    
    
    def is_connected(self):
        return self.kosaraju()

    @property
    def number_of_nodes(self):
        return len(self.G.nodes)
    
    def showGraph(self):
        A_graph = to_agraph(self.G)
        A_graph.layout('dot')
        if self.pos:
            nx.draw(self.G,self.pos,with_labels=True)
        else:
            nx.draw(self.G,self.pos,with_labels=True)


    
    def get_neighbours(self,s):
        """
        get neighbours of a node s in a set of undirected edges
        """

        return [v for (v,c) in self.repr[s]]
    

    def dfs(self,visited,node):  #function for dfs 
        if node not in visited:
            visited.append(node)
            for neighbour in self.get_neighbours(node):
                self.dfs(visited, neighbour)

    
     # DFS based function to find all bridges. It uses recursive
    # function bridgeUtil()
    def bridges(self):
        def bridgeUtil(u, visited, parent, low, disc):
 
            # Mark the current node as visited and print it
            i = self.inv_map[u]
            visited[i]= True

            # Initialize discovery time and low value
            disc[i] = self.Time
            low[i] = self.Time
            self.Time += 1

            #Recur for all the vertices adjacent to this vertex
            for v in self.get_neighbours(u):
                j = self.inv_map[v]
                # If v is not visited yet, then make it a child of u
                # in DFS tree and recur for it
                if visited[j] == False :
                    parent[j] = u
                    bridgeUtil(v, visited, parent, low, disc)

                    # Check if the subtree rooted with v has a connection to
                    # one of the ancestors of u
                    low[i] = min(low[i], low[j])


                    ''' If the lowest vertex reachable from subtree
                    under v is below u in DFS tree, then u-v is
                    a bridge'''
                    if low[j] > disc[i]:
                        bridges.append((u,v))


                elif v != parent[i]: # Update low value of u for parent function calls.
                    low[i] = min(low[i], disc[j])

        # Mark all the vertices as not visited and Initialize parent and visited, 
        # and ap(articulation point) arrays
        time = 0
        bridges = []
        visited = [False] * (self.number_of_nodes)
        disc = [float("Inf")] * (self.number_of_nodes)
        low = [float("Inf")] * (self.number_of_nodes)
        parent = [-1] * (self.number_of_nodes)
 
        # Call the recursive helper function to find bridges
        # in DFS tree rooted with vertex 'i'
        for i in self.nodes:
            if visited[self.inv_map[i]] == False:
                bridgeUtil(i, visited, parent, low, disc)
       
    
    def save_dat_format(self):
        filename = f"v_{self.number_of_nodes}_a{len(self.edges)}_r0_b{len(self.blockages) if self.blockages != None else 0}.dat"
                                    
        with open(filename,'w') as f:
            f.write(f"INSTANCE_NAME v_{self.number_of_nodes}_a{len(self.edges)}_r0_b{len(self.blockages) if self.blockages != None else 0}\n")
            f.write(f"NB_VERTICES {self.number_of_nodes}\n")
            f.write(f"NB_ARCS {len(self.edges)}\n")
            f.write(f"NB_BLOCKAGES {len(self.blockages) if self.blockages != None else 0}\n\n\n")

            f.write("VERTICES\n")
            for ((i,j),_) in self.costs.items():
                f.write(f"{i} {j} \n")
            
            f.write("\n\nARCS\n")
            for ((i,j),c) in self.costs.items():
                f.write(f"{i} {j} {c}\n")
                
            f.write("\n\n")
            f.write("REQUESTS\n\n")
            f.write("BLOCKAGES\n\n")
            for (i,j) in self.blockages:
                f.write(f"{i} {j}\n")
                
          
            f.write("\nEND\n")
            
    
            
        
       

class GridInstance(GraphInstance):
    def __init__(self,n,generate_graph=True):
        super().__init__(edges=GridInstance.create_grid_instance_edges(n),generate_graph=generate_graph,pos={n*(i-1)+j:(j,n-i) for i in range(1,n+1) for j in range(1,n+1)})
        self.n = n
        self.costs = {e:1 for e in self.edges}
    
    
    def getGraph(self):
        if self.G == None:
            self.G = nx.Graph()
            self.G.add_edges_from(self.edges)
        return self.G
    
    @property
    def number_of_nodes(self):
        return self.n**2
    @classmethod
    def create_grid_instance_edges(cls,n):
        """
        Generate edges for a nxn grid instance
        """
        edges = {}
        for i in range(1,n+1):
            for j in range(1,n+1):
                if i < n:
                    k = (i-1)*n+j
                    l = i*n+j
                    edges[(k,l)] = edges[(l,k)] = 1
                if j < n:
                    k = (i-1)*n+j
                    l = (i-1)*n+j+1
                    edges[(k,l)] = edges[(l,k)] = 1
                    
        return edges

        
    @classmethod
    def create_instance_with_blockages(cls,n,n_blockages):
        blockages = []
        gI = GridInstance(n,True)
        edges = gI.edges
        nodes = gI.nodes
        gI.blocks = generate_blockages(nodes,edges,n_blockages)
    
        
        return gI
    
    
    

class SNOPCompactModel:
    def __init__(self,instance,name="SNOP Compact",pos=None,env=None,construct=True):
        self.instance = instance
        if env:
            self.model = gp.Model(name=name,env = env)
        else:
            self.model = gp.Model(name=name)
        self.x , self.f = {} , {}
        if construct:
            self.construct_model()
    
        self.pos = pos
    def __enter__(self):
        return self
    
    def __exit__(self,type,value,traceback):
        pass
    def construct_model(self):
        
        #Construct all possible directed edges
        A = []
        A_costs = {}

        for edge in self.instance.edges:
            A.append(edge)
            A.append((edge[1],edge[0]))

            A_costs[edge] = self.instance.costs[edge]
            A_costs[(edge[1],edge[0])] = self.instance.costs[edge]
        
        N = self.instance.nodes
        # Construct variables
        for edge in A:
            self.x[edge[0],edge[1]] = self.model.addVar(name="x_%s%s"%(edge[0],edge[1]),vtype=GRB.BINARY)
            self.x[edge[1],edge[0]] = self.model.addVar(name="x_%s%s"%(edge[1],edge[0]),vtype=GRB.BINARY)

            for s in N:
                self.f[s,edge[0],edge[1]] = self.model.addVar(0,GRB.INFINITY,name="f^%s_%s%s"%(s,edge[0],edge[1]),vtype=GRB.CONTINUOUS)
                self.f[s,edge[1],edge[0]] = self.model.addVar(0,GRB.INFINITY,name="f^%s_%s%s"%(s,edge[1],edge[0]),vtype=GRB.CONTINUOUS)

        self.model.update()
        
        

        #Construct objective
        objective = gp.quicksum(A_costs[i,j]*self.f[s,i,j] for (i,j) in A for s in N)
        self.model.setObjective(objective,sense=GRB.MINIMIZE)
        
        #Construct constraints
        
        for (i,j) in self.instance.edges:
            self.model.addConstr(self.x[i,j] + self.x[j,i] == 1)

        for s in N:
            self.model.addConstr(sum(self.f[s,s,i] for i in self.instance.get_neighbours(s)) == self.instance.number_of_nodes - 1)
            for i in N:
                if i != s :
                    neighbours = self.instance.get_neighbours(i)
                    self.model.addConstr(sum(self.f[s,j,i] for j in neighbours) - sum(self.f[s,i,j] for j in neighbours) == 1)
            for (i,j) in A:
                self.model.addConstr(-self.f[s,i,j]+(self.instance.number_of_nodes-1)*self.x[i,j] >= 0)

        #blockages
        if self.instance.blockages:
            for (i,j) in self.instance.blockages:
                self.model.addConstr(self.x[i,j] == 0)
            
    def save_model(self,savename):
        self.model.write(savename)
    
    def solve(self,callback=None,itLimit=None,timeLimit=None,show=True):
        if itLimit:
            self.model.Params.IterationLimit = itLimit
        if timeLimit:
            self.model.Params.timeLimit = timeLimit
        self.model.optimize(callback)
    
        if self.model.status == GRB.OPTIMAL:
            # for now, we're dealing only with optimal solutions
            diedges = []
            for v in self.model.getVars():
                if v.varname[0] == "x" and v.x > 0:
                    for (i,j) in self.instance.edges:
                        if v.varname == "x_%s%s"%(i,j):
                            diedges.append((i,j))
                            break
                        elif v.varname == "x_%s%s"%(j,i):
                            diedges.append((j,i))
                            break
            if show:
                Gp = nx.MultiDiGraph()
                Gp.add_edges_from(diedges)

                if self.instance.pos:
                    nx.draw(Gp,self.instance.pos,with_labels=True)
                else:
                    nx.draw(Gp,with_labels=True)
            return self.model.getVars()
        
        return None
    
    

def RDFS(nodes,edges):
    #1 -> white 
    #0 -> black
    #-1 -> gray
    def adjw(s):
        """
        get neighbours of a node s in a set of undirected edges
        """
        neighbours = []
        for (i,j) in edges:
            if i == s:
                neighbours.append(j)
            elif j == s:
                neighbours.append(i)

        return sorted(neighbours,key=lambda x : w[inv_map[x]])
    
    def DFS_visit(i,f):
        color[inv_map[i]] = -1
        
        for j in adjw(i):
            if color[inv_map[j]] == 1:
                A.append((i,j))
                DFS_visit(j,i)
            elif j != f and color[inv_map[j]] != 0:
                A.append((i,j))
        color[inv_map[i]] = 0
        
    mapp = {i:n for (i,n) in enumerate(nodes)}
    inv_map = {n:i for (i,n) in enumerate(nodes)}
    
    color = [1 for _ in range(len(nodes))]
    A = []
    w = list(range(0,len(nodes)))
    random.shuffle(w)
    
    r = mapp[np.argmin(w)]
    
    DFS_visit(r,-1)
    
    
    return A



def dfs(visited, edges, node):  #function for dfs 
    if node not in visited:
        visited.append(node)
        for neighbour in get_neighbours_out(edges,node):
            dfs(visited, edges, neighbour)
            
            
def kosaraju(nodes,edges):
    visited = []
    dfs(visited,edges,nodes[0])
    if len(visited) == len(nodes):
        reverse_edges = [(j,i) for (i,j) in edges]
        visited = []
        dfs(visited,reverse_edges,nodes[0])
    return len(visited) == len(nodes)


def get_blockages(nodes,edges,n_blockages):
    n = 0
    temp_edges = edges[:]
    blocks = []
    while n != n_blockages:
        r = random.choice(temp_edges)
        temp_edges.remove(r)
        if kosaraju(nodes,temp_edges):
            blocks.append(r)
            n += 1
        else:
            temp_edges.append(r)
    return blocks


def read_instance(filepath):

    arcs = {}
    vertices = []
    blockages = []
    name = ""
    requests  = []

    lecture = 0

    with open(filepath,"r") as f:
        for line in f:
            line = line.strip()
            if line != '':
                line = line.split(" ")
                if  lecture == 0:
                    if line[0] == "INSTANCE_NAME":
                        name = line[1].strip()
                    if line[0].strip() == "VERTICES":
                        lecture = 1
                elif lecture == 1:
                    if line[0] != "ARCS":
                        vertices.append(int(line[0]))
                    else:
                        lecture = 2
                elif lecture==2:
                    if line[0] != "REQUESTS":
                        arcs[int(line[0]),int(line[1])] = float(line[2])
                    else:
                        lecture = 3
                elif lecture == 3:
                    if line[0] != "BLOCKAGES":
                        requests.append(line)
                    else:
                        lecture = 4
                elif lecture == 4:
                    if line[0] != "END":
                        blockages.append((int(line[0]),int(line[1])))

    return vertices,arcs,requests,blockages

def generate_blockages(self,nodes,edges,n_blockages):
        self.blockages = []
        n = 0
        temp_edges = edges + [(j,i) for (i,j) in edges]
        blocks = []
        it = 0
        while n != n_blockages and it < 100000:
            r = random.choice(temp_edges)
            temp_edges.remove(r)
            if kosaraju(nodes,temp_edges):
                blocks.append(r)
                n += 1
            else:
                temp_edges.append(r)
            it += 1
        return blocks
           
