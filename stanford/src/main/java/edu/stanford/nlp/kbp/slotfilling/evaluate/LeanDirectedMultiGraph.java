package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.graph.Graph;
import edu.stanford.nlp.util.CollectionUtils;
import edu.stanford.nlp.util.Generics;

import java.util.*;

/**
 * Allows the addition of special "leaf" nodes that do not take up extra space
 */
public class LeanDirectedMultiGraph<V,E> implements Graph<V, E> /* Serializable */{
  protected Map<V, Map<V, List<E>>> outgoingEdges = Generics.newHashMap();

  public LeanDirectedMultiGraph() {
  }

  public LeanDirectedMultiGraph(LeanDirectedMultiGraph<V, E> graph) {
    this.outgoingEdges = graph.outgoingEdges;
  }
  // I don't save a incomingEdges to save on memory :-/

  /**
   * Be careful hashing these. They are mutable objects, and changing the object
   * will throw off the hash code, messing up your hash table
   */
  public int hashCode() {
    return outgoingEdges.hashCode();
  }

  @SuppressWarnings("unchecked")
  public boolean equals(Object that) {
    if (that == this)
      return true;
    if (!(that instanceof LeanDirectedMultiGraph))
      return false;
    return outgoingEdges.equals(((LeanDirectedMultiGraph) that).outgoingEdges);
  }

  /**
   * For adding a zero degree vertex
   *
   * @param v
   */
  public boolean addVertex(V v) {
    if (outgoingEdges.containsKey(v))
      return false;
    outgoingEdges.put(v, Generics.<V, List<E>>newHashMap());
    return true;
  }

  public boolean addLeaf(V v) {
    return addVertex(v);
//    if (outgoingEdges.containsKey(v))
//      return false;
//    outgoingEdges.put(v, Collections.EMPTY_MAP); // This node doesn't have any outgoing edges.
//    return true;
  }

  /**
   * adds vertices (if not already in the graph) and the edge between them
   *
   * @param source
   * @param dest
   * @param data
   */
  public void add(V source, V dest, E data) {
    addVertex(source);
    addVertex(dest);

    Map<V, List<E>> outgoingMap = outgoingEdges.get(source);
    List<E> outgoingList = outgoingMap.get(dest);
    if (outgoingList == null) {
      outgoingList = new ArrayList<E>(1); // Be conservative with memory please
      outgoingMap.put(dest, outgoingList);
    }
    outgoingList.add(data);
  }

  public boolean removeEdges(V source, V dest) {
    if (!outgoingEdges.containsKey(source)) {
      return false;
    }
    if (!outgoingEdges.get(source).containsKey(dest)) {
      return false;
    }
    outgoingEdges.get(source).remove(dest);
    return true;
  }

  public boolean removeEdge(V source, V dest, E data) {
    if (!outgoingEdges.containsKey(source)) {
      return false;
    }
    if (!outgoingEdges.get(source).containsKey(dest)) {
      return false;
    }
    boolean foundOut = outgoingEdges.get(source).get(dest).remove(data);
    // TODO: cut down the number of .get calls
    if (outgoingEdges.get(source).get(dest).size() == 0) {
      outgoingEdges.get(source).remove(dest);
    }
    return foundOut;
  }

  /**
   * remove a vertex (and its edges) from the graph.
   *
   * @param vertex
   * @return true if successfully removes the node
   */
  public boolean removeVertex(V vertex) {
    if (!outgoingEdges.containsKey(vertex)) {
      return false;
    }
    outgoingEdges.remove(vertex);
    return true;
  }

  public boolean removeVertices(Collection<V> vertices) {
    boolean changed = false;
    for (V v : vertices) {
      if (removeVertex(v)) {
        changed = true;
      }
    }
    return changed;
  }

  public int getNumVertices() {
    return outgoingEdges.size();
  }

  public List<E> getOutgoingEdges(V v) {
    return CollectionUtils.flatten(outgoingEdges.get(v).values());
  }

  @Override
  public List<E> getIncomingEdges(V v) {
    throw new RuntimeException("Not implemented");
  }

  public int getNumEdges() {
    int count = 0;
    for (Map.Entry<V, Map<V, List<E>>> sourceEntry : outgoingEdges.entrySet()) {
      for (Map.Entry<V, List<E>> destEntry : sourceEntry.getValue().entrySet()) {
        count += destEntry.getValue().size();
      }
    }
    return count;
  }

  /*
   * WARNING: This can be an expensive computation.
   */
  public Set<V> getParents(V vertex) {
    if( !outgoingEdges.containsKey(vertex) ) return null; // AAH :-/

    Set<V> parents = new HashSet<V>();
    for( Map.Entry<V,Map<V,List<E>>> edge : outgoingEdges.entrySet() ) {
      if( edge.getValue().containsKey(vertex) )
        parents.add( edge.getKey() );
    }
    return Collections.unmodifiableSet(parents);
  }

  public Set<V> getChildren(V vertex) {
    Map<V, List<E>> childMap = outgoingEdges.get(vertex);
    if (childMap == null)
      return null;
    return Collections.unmodifiableSet(childMap.keySet());
  }

  /**
   * Gets both parents and children nodes
   * WARNING: Don't use this please. I won't like you if you do.
   * @param v
   */
  public Set<V> getNeighbors(V v) {
    // TODO: pity we have to copy the sets... is there a combination set?
    Set<V> children = getChildren(v);
    Set<V> parents = getParents(v);

    if (children == null && parents == null)
      return null;
    Set<V> neighbors = Generics.newHashSet();
    neighbors.addAll(children);
    neighbors.addAll(parents);
    return neighbors;
  }

  /**
   * clears the graph, removes all edges and nodes
   */
  public void clear() {
    outgoingEdges.clear();
  }

  public boolean containsVertex(V v) {
    return outgoingEdges.containsKey(v);
  }

  /**
   * only checks if there is an edge from source to dest. To check if it is
   * connected in either direction, use isNeighbor
   *
   * @param source
   * @param dest
   */
  public boolean isEdge(V source, V dest) {
    Map<V, List<E>> childrenMap = outgoingEdges.get(source);
    if (childrenMap == null || childrenMap.isEmpty())
      return false;
    List<E> edges = childrenMap.get(dest);
    if (edges == null || edges.isEmpty())
      return false;
    return edges.size() > 0;
  }

  public boolean isNeighbor(V source, V dest) {
    return isEdge(source, dest) || isEdge(dest, source);
  }

  public Set<V> getAllVertices() {
    return Collections.unmodifiableSet(outgoingEdges.keySet());
  }

  public List<E> getAllEdges() {
    List<E> edges = new ArrayList<E>();
    for (Map<V, List<E>> e : outgoingEdges.values()) {
      for (List<E> ee : e.values()) {
        edges.addAll(ee);
      }
    }
    return edges;
  }

  /**
   * False if there are any vertices in the graph, true otherwise. Does not care
   * about the number of edges.
   */
  public boolean isEmpty() {
    return outgoingEdges.isEmpty();
  }

  /**
   * Deletes nodes with zero incoming and zero outgoing edges
   * WARNING: Likely slow :-/
   */
  public void removeZeroDegreeNodes() {
    Map<V,Integer> references = new HashMap<V,Integer>();
    for (V vertex : outgoingEdges.keySet()) {
      for( V vertex_ : outgoingEdges.get(vertex).keySet() ) {
        if( !references.containsKey(vertex_) ) references.put( vertex_, 0 );
        references.put(vertex_, references.get(vertex_) + 1 );
      }
    }
  }


  public List<E> getEdges(V source, V dest) {
    Map<V, List<E>> childrenMap = outgoingEdges.get(source);
    if (childrenMap == null) {
      return Collections.emptyList();
    }
    List<E> edges = childrenMap.get(dest);
    if (edges == null) {
      return Collections.emptyList();
    }
    return Collections.unmodifiableList(edges);
  }

  public int getInDegree(V vertex) {
    if (!containsVertex(vertex)) {
      throw new IllegalArgumentException("Unknown vertex " + vertex);
    }
    int result = 0;
    for( Map.Entry<V,Map<V,List<E>>> edge : outgoingEdges.entrySet() ) {
      if( edge.getValue().containsKey(vertex) ) result++;
    }
    return result;
  }

  public int getOutDegree(V vertex) {
    int result = 0;
    Map<V, List<E>> outgoing = outgoingEdges.get(vertex);
    if (outgoing == null) {
      throw new IllegalArgumentException("Unknown vertex " + vertex);
    }
    for (List<E> edges : outgoing.values()) {
      result += edges.size();
    }
    return result;
  }

  @Override
  public List<Set<V>> getConnectedComponents() {
    throw new RuntimeException("Not implemented");
  }

  public Iterator<E> outgoingEdgeIterator(final V vertex) {
    return new EdgeIterator<V, E>(outgoingEdges, vertex);
  }

  public Iterable<E> outgoingEdgeIterable(final V vertex) {
    return new Iterable<E>() {
      public Iterator<E> iterator() {
        return new EdgeIterator<V, E>(outgoingEdges, vertex);
      }
    };
  }

  public Iterator<E> edgeIterator() {
    return new EdgeIterator<V, E>(this);
  }

  public Iterable<E> edgeIterable() {
    return new Iterable<E>() {
      public Iterator<E> iterator() {
        return new EdgeIterator<V, E>(LeanDirectedMultiGraph.this);
      }
    };
  }

  static class EdgeIterator<V, E> implements Iterator<E> {
    private Iterator<Map<V, List<E>>> vertexIterator;
    private Iterator<List<E>> connectionIterator;
    private Iterator<E> edgeIterator;

    E next;

    public EdgeIterator(LeanDirectedMultiGraph<V, E> graph) {
      vertexIterator = graph.outgoingEdges.values().iterator();
      primeNext();
    }

    public EdgeIterator(Map<V, Map<V, List<E>>> source, V startVertex) {
      Map<V, List<E>> neighbors = source.get(startVertex);
      if (neighbors == null) {
        return;
      }
      connectionIterator = neighbors.values().iterator();
      primeNext();
    }

    public boolean hasNext() {
      return next != null;
    }

    public E next() {
      if (next == null) {
        throw new NoSuchElementException("Graph edge iterator exhausted.");
      }
      E value = next;
      primeNext();
      return value;
    }

    private void primeNext() {
      while (true) {
        if (edgeIterator != null && edgeIterator.hasNext()) {
          next = edgeIterator.next();
          break;
        }

        if (connectionIterator != null && connectionIterator.hasNext()) {
          edgeIterator = connectionIterator.next().iterator();
          continue;
        }

        if (vertexIterator != null && vertexIterator.hasNext()) {
          connectionIterator = vertexIterator.next().values().iterator();
          continue;
        }

        next = null;
        break;
      }
    }

    public void remove() {
      throw new UnsupportedOperationException();
    }
  }

  @Override
  public String toString() {
    StringBuilder s = new StringBuilder();
    s.append("{\n");
    s.append("Vertices:\n");
    for (V vertex : outgoingEdges.keySet()) {
      s.append("  ").append(vertex).append('\n');
    }
    s.append("Edges:\n");
    for (V source : outgoingEdges.keySet()) {
      for (V dest : outgoingEdges.get(source).keySet()) {
        for (E edge : outgoingEdges.get(source).get(dest)) {
          s.append("  ").append(source).append(" -> ").append(dest).append(" : ").append(edge).append('\n');
        }
      }
    }
    s.append('}');
    return s.toString();
  }
}
