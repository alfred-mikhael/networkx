"""Functions for computing sparsifiers of graphs."""

import math

import networkx as nx
from networkx.utils import not_implemented_for, py_random_state

__all__ = ["spanner"]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@py_random_state(3)
@nx._dispatchable(edge_attrs="weight", returns_graph=True)
def spanner(G, stretch, weight=None, seed=None):
    """Returns a spanner of the given graph with the given stretch.

    A spanner of a graph G = (V, E) with stretch t is a subgraph
    H = (V, E_S) such that E_S is a subset of E and the distance between
    any pair of nodes in H is at most t times the distance between the
    nodes in G.

    Parameters
    ----------
    G : NetworkX graph
        An undirected simple graph.

    stretch : float
        The stretch of the spanner.

    weight : object
        The edge attribute to use as distance.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    NetworkX graph
        A spanner of the given graph with the given stretch.

    Raises
    ------
    ValueError
        If a stretch less than 1 is given.

    Notes
    -----
    This function implements the spanner algorithm by Baswana and Sen,
    see [1].

    This algorithm is a randomized las vegas algorithm: The expected
    running time is O(km) where k = (stretch + 1) // 2 and m is the
    number of edges in G. The returned graph is always a spanner of the
    given graph with the specified stretch. For weighted graphs the
    number of edges in the spanner is O(k * n^(1 + 1 / k)) where k is
    defined as above and n is the number of nodes in G. For unweighted
    graphs the number of edges is O(n^(1 + 1 / k) + kn).

    References
    ----------
    [1] S. Baswana, S. Sen. A Simple and Linear Time Randomized
    Algorithm for Computing Sparse Spanners in Weighted Graphs.
    Random Struct. Algorithms 30(4): 532-563 (2007).
    """
    if stretch < 1:
        raise ValueError("stretch must be at least 1")

    k = (stretch + 1) // 2

    # initialize spanner H with empty edge set
    H = nx.empty_graph()
    H.add_nodes_from(G.nodes)

    # phase 1: forming the clusters
    # the residual graph has V' from the paper as its node set
    # and E' from the paper as its edge set
    residual_graph = _setup_residual_graph(G, weight)
    # clustering is a dictionary that maps nodes in a cluster to the
    # cluster center
    clustering = {v: v for v in G.nodes}
    sample_prob = math.pow(G.number_of_nodes(), -1 / k)
    size_limit = 2 * math.pow(G.number_of_nodes(), 1 + 1 / k)

    i = 0
    while i < k - 1:
        # step 1: sample centers
        sampled_centers = set()
        for center in set(clustering.values()):
            if seed.random() < sample_prob:
                sampled_centers.add(center)

        # combined loop for steps 2 and 3
        edges_to_add = set()
        edges_to_remove = set()
        new_clustering = {}
        for v in residual_graph.nodes:
            if clustering[v] in sampled_centers:
                continue

            # step 2: find neighboring (sampled) clusters and
            # lightest edges to them
            lightest_edge_neighbor, lightest_edge_weight = _lightest_edge_dicts(
                residual_graph, clustering, v
            )
            neighboring_sampled_centers = (
                set(lightest_edge_weight.keys()) & sampled_centers
            )

            # step 3: add edges to spanner
            if not neighboring_sampled_centers:
                # connect to each neighboring center via lightest edge
                for neighbor in lightest_edge_neighbor.values():
                    edges_to_add.add((v, neighbor))
                # remove all incident edges
                for neighbor in residual_graph.adj[v]:
                    edges_to_remove.add((v, neighbor))

            else:  # there is a neighboring sampled center
                closest_center = min(
                    neighboring_sampled_centers, key=lightest_edge_weight.get
                )
                closest_center_weight = lightest_edge_weight[closest_center]
                closest_center_neighbor = lightest_edge_neighbor[closest_center]

                edges_to_add.add((v, closest_center_neighbor))
                new_clustering[v] = closest_center

                # connect to centers with edge weight less than
                # closest_center_weight
                for center, edge_weight in lightest_edge_weight.items():
                    if edge_weight < closest_center_weight:
                        neighbor = lightest_edge_neighbor[center]
                        edges_to_add.add((v, neighbor))

                # remove edges to centers with edge weight less than
                # closest_center_weight
                for neighbor in residual_graph.adj[v]:
                    nbr_cluster = clustering[neighbor]
                    nbr_weight = lightest_edge_weight[nbr_cluster]
                    if (
                        nbr_cluster == closest_center
                        or nbr_weight < closest_center_weight
                    ):
                        edges_to_remove.add((v, neighbor))

        # check whether iteration added too many edges to spanner,
        # if so repeat
        if len(edges_to_add) > size_limit:
            # an iteration is repeated O(1) times on expectation
            continue

        # iteration succeeded
        i = i + 1

        # actually add edges to spanner
        for u, v in edges_to_add:
            _add_edge_to_spanner(H, residual_graph, u, v, weight)

        # actually delete edges from residual graph
        residual_graph.remove_edges_from(edges_to_remove)

        # copy old clustering data to new_clustering
        for node, center in clustering.items():
            if center in sampled_centers:
                new_clustering[node] = center
        clustering = new_clustering

        # step 4: remove intra-cluster edges
        for u in residual_graph.nodes:
            for v in list(residual_graph.adj[u]):
                if clustering[u] == clustering[v]:
                    residual_graph.remove_edge(u, v)

        # update residual graph node set
        for v in list(residual_graph.nodes):
            if v not in clustering:
                residual_graph.remove_node(v)

    # phase 2: vertex-cluster joining
    for v in residual_graph.nodes:
        lightest_edge_neighbor, _ = _lightest_edge_dicts(residual_graph, clustering, v)
        for neighbor in lightest_edge_neighbor.values():
            _add_edge_to_spanner(H, residual_graph, v, neighbor, weight)

    return H


def _setup_residual_graph(G, weight):
    """Setup residual graph as a copy of G with unique edges weights.

    The node set of the residual graph corresponds to the set V' from
    the Baswana-Sen paper and the edge set corresponds to the set E'
    from the paper.

    This function associates distinct weights to the edges of the
    residual graph (even for unweighted input graphs), as required by
    the algorithm.

    Parameters
    ----------
    G : NetworkX graph
        An undirected simple graph.

    weight : object
        The edge attribute to use as distance.

    Returns
    -------
    NetworkX graph
        The residual graph used for the Baswana-Sen algorithm.
    """
    residual_graph = G.copy()

    # establish unique edge weights, even for unweighted graphs
    for u, v in G.edges():
        if not weight:
            residual_graph[u][v]["weight"] = (id(u), id(v))
        else:
            residual_graph[u][v]["weight"] = (G[u][v][weight], id(u), id(v))

    return residual_graph


def _lightest_edge_dicts(residual_graph, clustering, node):
    """Find the lightest edge to each cluster.

    Searches for the minimum-weight edge to each cluster adjacent to
    the given node.

    Parameters
    ----------
    residual_graph : NetworkX graph
        The residual graph used by the Baswana-Sen algorithm.

    clustering : dictionary
        The current clustering of the nodes.

    node : node
        The node from which the search originates.

    Returns
    -------
    lightest_edge_neighbor, lightest_edge_weight : dictionary, dictionary
        lightest_edge_neighbor is a dictionary that maps a center C to
        a node v in the corresponding cluster such that the edge from
        the given node to v is the lightest edge from the given node to
        any node in cluster. lightest_edge_weight maps a center C to the
        weight of the aforementioned edge.

    Notes
    -----
    If a cluster has no node that is adjacent to the given node in the
    residual graph then the center of the cluster is not a key in the
    returned dictionaries.
    """
    lightest_edge_neighbor = {}
    lightest_edge_weight = {}
    for neighbor in residual_graph.adj[node]:
        nbr_center = clustering[neighbor]
        weight = residual_graph[node][neighbor]["weight"]
        if (
            nbr_center not in lightest_edge_weight
            or weight < lightest_edge_weight[nbr_center]
        ):
            lightest_edge_neighbor[nbr_center] = neighbor
            lightest_edge_weight[nbr_center] = weight
    return lightest_edge_neighbor, lightest_edge_weight


def _add_edge_to_spanner(H, residual_graph, u, v, weight):
    """Add the edge {u, v} to the spanner H and take weight from
    the residual graph.

    Parameters
    ----------
    H : NetworkX graph
        The spanner under construction.

    residual_graph : NetworkX graph
        The residual graph used by the Baswana-Sen algorithm. The weight
        for the edge is taken from this graph.

    u : node
        One endpoint of the edge.

    v : node
        The other endpoint of the edge.

    weight : object
        The edge attribute to use as distance.
    """
    H.add_edge(u, v)
    if weight:
        H[u][v][weight] = residual_graph[u][v]["weight"][0]


def _nagamochi_ibaraki_certificate(G, k):
    """Computes the Nagamochi-Ibaraki k-sparse spanning forest in `G`. This is
    a maximal spanning forest `F` with at most `k * (n-1)` edges, such that
    `F` contains all edges of `G` with edge connectivity less than or equal to
    `k`.

    Parameters
    ----------
    G : NetworkX Graph

    k : int
        The edge connectivity threshold such that all edges of `G` with edge
        connectivity less than or equal to `k` will be in the returned forest.

    Returns
    -------
    F : collection of edges
        A spanning forest with at most `k * (n-1)` edges, such that all edges
        of `G` which cross a cut of value at most `k` are included in `F`.
    """
    m = G.size()

    node_buckets = {i: set() for i in range(m)}
    node_buckets[0] = set(G)
    node_labels = {v: 0 for v in G}
    largest_label = 0

    trees = [set()] * m
    scanned_edges = set()

    if G.is_multigraph():
        # need edges with multiplicity
        def edge_dict(u):
            return G.edges(u, keys=True)
    else:

        def edge_dict(u):
            return G.edges(u)

    while node_buckets[largest_label] > 0:
        u = node_buckets[largest_label].pop()
        for e in edge_dict(u):
            if e in scanned_edges:
                continue
            v = e[1]
            trees[node_labels[v] + 1].add(e)
            # increment label and move to new bucket
            node_buckets[node_labels[v]].remove(v)
            node_buckets[node_labels[v] + 1].add(v)
            node_labels[v] += 1

    return set.union(trees[i] for i in range(k))


def _contract_all_but(G, E):
    """Contracts all edges of `G` except for those in `E`, and returns the
    contracted graph.

    Parameters
    ----------
    G : NetworkX Graph

    E : collection
        Collection of edges to avoid contracting

    Returns
    -------
    NetworkX Graph
        A graph with all edges of `G` contracted, except for those in `E`.
    """
    H = G.__class__()
    G.remove_edges_from(E)
    components = list(nx.connected_components(G))
    num_components = len(components)
    node_images = {v: i for i in range(num_components) for v in components[i]}
    # restore G
    G.add_edges_from(E)
    H.add_nodes_from(i for i in range(num_components))
    for e in E:
        # can't unpack as u, v = e because there may or may not be a key,
        # depending on whether G is a multigraph or not
        u = e[0]
        v = e[1]
        H.add_edge(node_images(u), node_images(v))
    return H


def _sparse_partition(G, k):
    """Compute a `k`-sparse partition of `G`. A `k`-sparse partition is a set
    of edges `E` such that all edges with connectivity at most `k` are in `E`
    and if `G - E` has `r` connected components, then `E` has at most
    `2 * r * (n-1)` edges.
    """
    if G.size() <= 2 * k * (G.number_of_nodes() - 1):
        return G.edges(keys=True) if G.is_multigraph() else G.edges()

    to_remove = _nagamochi_ibaraki_certificate(G, k)
    H = _contract_all_but(G, to_remove)
    return _sparse_partition(H, k)


def _weak_edges(G, k):
    """Compute all edges in `G` which have strong connectivity no more than
    `2k`.
    """
    n = G.number_of_nodes()
    weak_edges = set()
    for _ in range(math.ceil(math.log2(n))):
        E = _sparse_partition(G, 2 * k)
        # exit early if there are no more edges to remove
        if len(E) == 0:
            break
        G.remove_edges_from(E)
        weak_edges.update(E)
    # restore G
    G.add_edges_from(weak_edges)
    return weak_edges


def _estimate_strong_connectivities(G):
    """Estimates the strong connectivity of each edge in `G` up to a factor
    of 2, such that for any edge e with strong connectivity k, the estimate
    for edge e is between k/2 and k.
    """
    m = G.size()
    edge_connectivities = {e: 1 for e in G.edges()}
    if G.is_multigraph():
        edge_connectivities = {e: 1 for e in G.edges(keys=True)}
    removed = set()
    connectivity = 1

    def estimation(C, k):
        E = _weak_edges(C, 2 * k)
        G.remove_edges_from(E)
        removed.update(E)
        for e in E:
            edge_connectivities[e] = k

    while connectivity <= m:
        for c in nx.connected_components(G):
            if len(c) > 1:
                estimation(G.subgraph(c), connectivity)
        connectivity *= 2

    G.add_edges_from(removed)
    return edge_connectivities


@not_implemented_for("directed")
@py_random_state(3)
def cut_sparsifier(G, epsilon, seed=None):
    r"""Returns a $1 \pm \epsilon$ cut sparsifier of a graph `G` with n nodes,
    such that at most the returned graph has at most `30 n ln n / epsilon ** 2`
    edge with probability at least `1 - 1/n`.

    Parameters
    ----------
    G : NetworkX Graph

    epsilon : float
        The quality of the cut approximator. If a cut in `G` has value `k`,
        then the same cut in the returned graph will have value between
        `(1-epsilon) * k` and `(1 + epsilon) * k`.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    H : NetworkX Graph or MultiGraph
        A weighted graph or multigraph which is a sparse $1 \pm \epsilon$
        cut-approximator of `G`.

    Raises
    ------
    NetworkXError
        If the input graph has less than `30 n ln n / epsilon ** 2` edges.

    ValueError
        If `epsilon` is not positive.
    """
    n = G.number_of_nodes()

    if G.size() < 30 * n * math.log(n) / epsilon**2:
        raise nx.NetworkXError("The input graph is already sparse.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    rho = 15 * math.log(n) / epsilon**2
    connectivities = _estimate_strong_connectivities(G)

    H = G.__class__()
    H.add_nodes_from(G)
    edge_iter = G.edges(keys=True) if G.is_multigraph() else G.edges()
    for e in edge_iter:
        p = min(1, rho / connectivities[e])
        u = e[0]
        v = e[1]
        if seed.random() < p:
            H.add_edge(u, v, weight=connectivities[e])
    return H
