# INTRODUCTION

Graphs represent connections between objects: Internet, maps, social networks, configuration space (positions connected by motion), etc.

_Note: all examples are written in JavaScript._ 

#### Terminology

+ each node is called a **vertex**
+ each line connecting two vertices $$(u, v)$$ is called an **edge**, **incident** on the two vertices
+ two connected vertices are **adjacent** or **neighbors**
+ the number of edges incident on a vertex is its **degree**
+ two undirectly connected vertices are linked via one or several **paths of $$x$$ edges** 
+ a path going from a particular vertex to itself is a **cycle**
+ an edge connecting a vertex to itself is a **loop**
+ edges can be weighted, giving a **weighted graph**
+ edges can be directed, giving a **directed graph**

<br>
The vertex set is usually noted $$V$$ and the edge set noted $$E$$.

As an example, a **map** can be thought of as a graph of weighted edges, or **weighted graph**: the cities are represented a vertices, roads as edges  & distances as weights. The shortest route between two locations is the path of minimum weigth. **One-way roads** can be represented as **directed edges**.

Another example of directed graphs are lists of interdependant tasks, PERT charts, dependencies between software libraries, etc.

#### Density

The ratio between edges and vertices $$|E|/|V|$$ is the graph **density**. 

+ a graph is **dense** _(left)_ when $$|E| \simeq |V|^2$$, ie when a large fraction of all vertices pairs are connected by edges
+ a graph  is **sparce** _(right)_  when $$|E| \simeq |V|$$, ie when each vertex only has a few edges

![Example](/modules/articles/assets/img/dense_sparce.png)

#### Representations

Vertices are often identified by indexes from $$0$$ to $$|V|-1$$. Edges can be represented in several ways:

+ **edge list**: array of $$|E|$$ arrays. Two vertex numbers, plus the weight if needed
+ **adjacency matrix**: matrix of size $$|V| \times |V|$$. 0s and 1s when edge $$(i,j)$$ exists
+ **adjacency list**: array of $$|V|$$ arrays: at index $$i$$, all the vertices adjacent to vertex $$i$$

Let's look at an example:

![Example](/modules/articles/assets/img/graph.png)

##### Edge List

$$(A,B),(A,C),(A,D),(C,D)$$

##### Adjacency Matrix

$$ \begin{matrix} 
     & A & B & C & D \\
	A & 0 & 1 & 1 & 1 \\
    B & 1 & 0 & 0 & 0 \\
    C & 1 & 0 & 0 & 1 \\
    D & 1 & 0 & 1 & 0
\end{matrix}$$

_Note: for weighted graphs, the null value can be used to indicate absent edges. For undirected graphs,  the matrix is symmetric as all edges are both ways._

##### Adjacency List

$$ (A: B,C,D) ,(B: A), (C: A,D),(D: A,C)$$




#### Runtime

The main inconvenient of edge lists is the loop over $$|E|$$ to search for existing edges. Matrices takes a lot of space, even if the graph is sparce (few edges), and the search of adjacent vertices requires a loop over $$|V|$$. In adjacent lists, the search for adjacent vertices takes at worst $$\Theta(d)$$, where $$d$$ is the vertex degree.

| Representation | Space | Search for edge  | List all edges | List Neighbours | 
|:-------------:|:--------------:|:--------------:|:--------------:|:--------------:|
| Adjacent Matrix    |    $$V^2$$      | $$1$$  | $$V^2$$ | $$V$$ | 
| Edge List                 |    $$E$$    |  $$E$$  | $$E$$ | $$E$$ |
| Adjacent List         |    $$V+2E$$      |  $$d$$ | $$E$$ | $$d$$ |

_Note: $$\Theta$$ is omitted, $$|V|$$ and $$|E|$$ are replaced by $$V$$ and $$E$$. For instance, $$V^2$$ means $$\Theta(|V|^2)$$._

Adjacent Lists for undirected graphs contain $$2 \times |E|$$ elements: each edge $$(i, j)$$ appears in the $$i$$ and $$j$$ lists. For directed graphs, adjacent lists only have [E| elements.

The runtime of $$O(|V|)$$ and $$O(|E|)$$ can be compared by using the graph **density**.




# DEPTH-FIRST SEARCH

#### Principle

Depth-First Search **traverses an entire graph**, exploring as far as possible along each branch before backtracking (hence its name).

It is used to identify:

+ **reachability**: is there a path from one vertex to another
+ **connectivity**: find **connected components**, ie. groups of vertices that are reachable from one another

A few applications: finding routes, solving puzzles with only one solution _(mazes, etc.)_.


#### Algorithm

The algorithm traverses the graph recursively, going from nodes to connected nodes. Here is an example with three connected components _(click/tap to toggle the animation)_ :

<div pw-gif staticImg="/modules/articles/assets/img/dfs00.png" gifImg="/modules/articles/assets/img/dfs.gif"></div>

```js
// the adjacency list is stored in 'graph'
var graph = [[1,2,3],[0],[0,3],[0,2],[5],[4],[7,8],[6,8],[6,7]];

// we track the connected component of each vertex
var dfsCCnum = [];
// we mark each vertex as not visited yet
var dfsVisited = [];
for ( var i = 0; i < graph.length; i++ ) { dfsVisited[i] = false; }

// 'explore' traverses all vertices connected together
function explore( v ) {
  // we mark the vertex as visited to prevent infinite loops 
  dfsVisited[v] = true;
  // we clock the execution of DFS - tic
  previsit( v );
  // we identify the connected comp
  dfsCCnum[v] = cc;
  // we recursively check all connected vertices
  var vNeighbors = graph[v];
  for ( var i = 0; i < vNeighbors.length; i++ ) {
    w = vNeighbors[i];
    if ( !dfsVisited[w] ) explore( w );
  }
  // we clock the execution of DFS - tac
  postvisit( v );
}

// DFS traverses the entire graph
var cc = 0;
function DFS( graph ) {
  for ( var v = 0; v < graph.length; v++ ) {
    if ( !dfsVisited[v] ) {
      // new connected component
      cc++; 
      // recursively traverse the entire connected component
      explore( v );
    }
  }
  return dfsCCnum;
}

// clock 
var pre = [], post = [], clock = 1;
function previsit( v ) {
  pre[v] = clock;
  clock++;
}
function postvisit( v ) {
  post[v] = clock;
  clock++;
}

// returns [1, 1, 1, 1, 2, 2, 3, 3, 3]
console.log( DFS( graph ) );
```

_Note: we have an example of closure of `DFS` & `explore` over `cc`. Both functions have access to the global scope where they were created. Moving the `cc` declaration inside `DFS` would lead to a `ReferenceError` from `explore`. _

#### Pre & Post Visit

For two distinct vertices $$u$$ and $$v$$, `[pre(u),post(u)]` and `[pre(v),post(v)]` can either be:

+ **nested**, if they are in the same connected component
+ **distinct**, otherwise

#### Runtime

Each vertex is explored exactly once. For each vertex, all neighbors are checked (ie. all edges of the vertex). 

_Note: for undirected graphs, each edge is counted twice: A neighbor of B means B neighbor of A._

The **total runtime** is $$O(|V|+|E|)$$:

+ $$O(1)$$ for each vertex
+ $$O(1)$$ for each edge
+ $$|V|$$ vertices
+ $$2\ |E|$$ edges




# DIRECTED GRAPHS

#### Terminology

In a directed graph:

+ edges **leave** and **enter** vertices
+ the number of edges  leaving & entering  a vertex are its **in  & out degrees**
+ a **source** has no incoming edge
+ a **sink** has no outgoing edges


#### DAG & Topological Sort

A directed graph without cycles is called a **Directed Acyclic Graphs**, or DAG. In the following figure, only A is a DAG:

![Example](/modules/articles/assets/img/DAG.png)

A directed graph can be linearly ordered only and only if it is a DAG. We can use  the reverse postorder of a DFS: once the branches leaving a vertex have been explored and ordered, the vertex becomes the last one of the remaining unordered graph. 

_Note: there can be several ways to order a graph._

Here is the DFS in action for the example A above, starting from source nodes:

<div pw-gif staticImg="/modules/articles/assets/img/toposort.png" gifImg="/modules/articles/assets/img/toposort.gif"></div>

```js
// the adjacency list is stored in 'graph'
// it starts at index 1, hence the empty array at pos 0
var graph = [[],[7],[1,3],[1,8],[5,6,3],[7],[8],[],[7]];

// we store the postorders 
var postorders = [];

// we mark each vertex as not visited yet
var dfsVisited = [];
for ( var i = 0; i < graph.length; i++ ) { dfsVisited[i] = false; }

// 'explore' traverses all vertices connected together
function explore( v ) {
  dfsVisited[v] = true;
  var vNeighbors = graph[v];
  for ( var i = 0; i < vNeighbors.length; i++ ) {
    w = vNeighbors[i];
    if ( !dfsVisited[w] ) explore( w );
  }
  // we record the postorder once all the
  // vertex branches are explored & ordered
  postorders.push( v );
}

// DFS traverses the entire graph, starting at both source vertices
// a more generic fn would explore all vertices
function DFS( graph ) {
  explore( 4 );
  explore( 2 );
  return postorders.reverse();
}

// returns [2, 4, 3, 1, 6, 8, 5, 7]
console.log( DFS( graph ) );
```


#### Strongly connected components

In a directed graph, groups of vertices that are reachable from one another are called  **strongly connected components** _(left)_. A **metagraph** shows connections between strongly connected components _(right)_.

![Example](/modules/articles/assets/img/strongCC.png)

_Note: a metagraph is always a DAG, as each cycle is represented as a single vertex._




# BREATH-FIRST SEARCH

#### Principle

Breadth-first search, also known as BFS, finds the **shortest path** (shortest number of edges) from a given vertex to all other vertices:

1. Start by marking the goal vertex with the number 0
2. After marking all the relevant vertices with the number $$k$$, mark with the number $$k+1$$ all vertices that are one step away and have not yet been marked
3. Find a path to the goal by choosing a sequence of vertices with always decreasing numbers: the **shortest-path tree**

_Note: the shortest-path tree contains no cycle._

Here is an illustration with a maze problem:

<div pw-gif staticImg="/modules/articles/assets/img/maze_algo00.png" gifImg="/modules/articles/assets/img/maze_algo.gif"></div>


#### Algorithm

The queue ensures that all vertices located at distance $$k$$ are processed before moving to distance $$k+1$$. 

```js
// Queue object prototype - not mandatory, but
// it makes the algorithm easier to read
var Queue = function() { this.items = []; };
Queue.prototype.enqueue = function(obj) { this.items.push(obj); };
Queue.prototype.dequeue = function() { return this.items.shift(); };
Queue.prototype.isEmpty = function() { return this.items.length === 0; };

// Breadth-first search - graph as adjacent list
var doBFS = function( graph, source ) {
  
  // init the bfs params for the source vertex
  var bfsInfo = {};
  bfsInfo[source] = {
    predecessor: null,
    distance: 0
  };
  
  // the queue is used to track which vertices to process next
  var queue = new Queue();
  queue.enqueue(source);

  // Traverse the graph
  var vertex, vNeighbors, newVertex;
  while (!queue.isEmpty()) {
    // remove first vertex from the queue & process it
    vertex = queue.dequeue(); 
    // loop over the adjacent vertices
    vNeighbors = graph[vertex];
    for (var i=0; i < vNeighbors.length; i++) { 
      newVertex = vNeighbors[i];
      //save new info if not already passed
      if (!bfsInfo[newVertex]) {
        bfsInfo[newVertex] = {
          predecessor: vertex,
          distance: bfsInfo[vertex].distance+1
        };
        // add newVertex to queue, to process it later
        queue.enqueue(newVertex); 
      }
    }      
  }
  
  return bfsInfo;
  
};

// get shortest path
var getShortestPath = function( bfsInfo, source, dest ) {
  var path = [dest];
  while ( dest !== source ) {
    path.push( bfsInfo[dest].predecessor );
    dest = bfsInfo[dest].predecessor;
  }
  return path.reverse();
};

// example (see graph picture below)
var graph = [[1,3],[0,2,4],[1,5],[0,6],[1,5],[2,4,7,8],[3,7],[5,6,8],[5,7]],
    source = 4,
    bfsInfo = doBFS( graph, source );

// return [4, 5, 7, 6]
var dest = 6;
console.log( getShortestPath( bfsInfo, source, dest ) );
```


![Example](/modules/articles/assets/img/bfs.png)


#### Runtime

It is the same as Depth-First Search.

Each vertex is enqueued at most once. For each vertex, all neighbors are checked (ie. all edges of the vertex). 

_Note: for undirected graphs, each edge is counted twice: A neighbor of B means B neighbor of A._

The **total runtime** is $$O(|V|+|E|)$$:

+ $$O(1)$$ for each vertex
+ $$O(1)$$ for each edge
+ $$|V|$$ vertices
+ $$2\ |E|$$ edges




# DIJKSTRA

#### Principle

The Dijkstra's Algorithm finds the **shortest path** (smallest sum of weights) from a given vertex to all other vertices, in a weighted graph:

1. Start by marking the goal vertex with the number 0
2. Assign distance to goal to +Inf, for all vertices except 0
3. Find the min distance $$d_{min}$$ between a vertex $$u$$ and the goal. We start at vertex 0, for which $$d_{min} = 0$$
4. Calculate the _tentative_ distance from the goal to each of $$u$$ neighbors, with a path going through $$u$$: $$d_{min}$$ plus the weighted edge
5. If the tentative distance is shorter than the previous shortest one, save it 
6. Go to the vertex with the smallest tentative distance: it is its $$d_{min}$$
7. Loop to 3 until all vertices have been explored

_Note: this algorithm does not work with negative weigths. It supposes that if $$(u,w)$$ is longer than $$(u,v)$$, then $$(u,w,v)$$ is longer too (implied in step 6)._

Example:

<div pw-gif staticImg="/modules/articles/assets/img/dijkstra.png" gifImg="/modules/articles/assets/img/dijkstra.gif"></div>
</br>


#### Algorithm

```js
// Queue object prototype - not mandatory, but
// it makes the algorithm easier to read
var Queue = function() { this.items = []; };
Queue.prototype.isEmpty = function() { return this.items.length === 0; };
Queue.prototype.enqueue = function(obj) { this.items.push(obj); };
// remove from queue the vertex with smallest distance to the goal
Queue.prototype.dequeueMin = function(info) { 
  var distMin = Infinity, 
      posMin = -1;
  for (var i=0; i <  this.items.length; i++) {
    if ( info[this.items[i]] && info[this.items[i]].distance < distMin ) {
      vertexMin = i;
      distMin = info[this.items[i]].distance;
    }
  }
  return this.items.splice(vertexMin,1)[0]; 
};

// Dijkstra - graph as adjacent list, from 0 to |V|-1
var dijkstra = function( graph, source ) {
  
  // init the params for the source vertex
  var info = {};
  info[source] = {
    predecessor: null,
    distance: 0
  };
  
  // all vertices are initially in the queue
  var queue = new Queue();
  for (var i=0; i < graph.length; i++) queue.enqueue(i);

  // Traverse the graph
  var vertex, vNeighbors, newVertex, edgeWeight;
  while (!queue.isEmpty()) {
    // remove vertex closest to goal from the queue & process it
    vertex = queue.dequeueMin(info); 
    // loop over the adjacent vertices
    vNeighbors = graph[vertex];
    for (var i=0; i < vNeighbors.length; i++) { 
      newVertex = vNeighbors[i][0];
      edgeWeight = vNeighbors[i][1];
      tentativeDist =  info[vertex].distance + edgeWeight;
      //save new info if not already passed or shorter
      if (!info[newVertex] || info[newVertex].distance > tentativeDist) {
        info[newVertex] = {
          predecessor: vertex,
          distance: tentativeDist
        };
      }
    }      
  }
  
  return info;
  
};

// get shortest path
var getShortestPath = function( info, source, dest ) {
  var path = [dest];
  while ( dest !== source ) {
    path.push( info[dest].predecessor );
    dest = info[dest].predecessor;
  }
  return path.reverse();
};

// example (see graph picture below)
var graph = [
  [[1,3],[4,10]],
  [[2,3],[4,8],[5,5]],
  [[3,2],[4,3],[5,1]],
  [],
  [[1,2],[5,5]],
  [[3,0]]
],
source = 0,
info = dijkstra( graph, source );

// return [0, 1, 2, 5, 3]
var dest = 3;
console.log( getShortestPath( info, source, dest ) );
```

![Example](/modules/articles/assets/img/dijkstraNum.png)


#### Runtime

The runtime can be split in three parts:

1. create initial queue with all the vertices
2. for each iteration of the `while` loop, dequeue the best vertex
3. for each iteration of the `vNeighbors` loop, update distance

So the runtime is:

$$T$$ ( makeQueue ) + $$|V| \times T$$ ( dequeueMin ) + $$|E| \times T$$ ( updateDist )

If the queue is an array, as in the code above, the runtime is $$O(|V| + |V|^2 + |E|) = O(|V|^2)$$.

