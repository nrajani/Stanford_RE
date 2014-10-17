package edu.stanford.nlp.kbp.slotfilling.common;

import edu.stanford.nlp.graph.DirectedMultiGraph;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.*;

import java.lang.reflect.Array;
import java.util.*;
import java.util.PriorityQueue;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * Utilities for common collection types
 */
@SuppressWarnings("UnusedDeclaration")
public class CollectionUtils {

  public static <T1,T2> Pair<List<T1>,List<T2>> unzip(List<Pair<T1,T2>> lst) {
    int elems = lst.size();
    List<T1> lst1 = new ArrayList<T1>(elems);
    List<T2> lst2 = new ArrayList<T2>(elems);

    for( Pair<T1,T2> element : lst ) {
      lst1.add(element.first);
      lst2.add(element.second);
    }
    return new Pair<List<T1>,List<T2>>(lst1, lst2);
  }
  public static <T1,T2> List<Pair<T1,T2>> zip(List<T1> lst1, List<T2> lst2 ) {
    int elems = (lst1.size() < lst2.size()) ? lst1.size() : lst2.size();
    List<Pair<T1,T2>> lst = new ArrayList<Pair<T1,T2>>(elems);

    for(int i = 0; i < elems; i++) {
      lst.add( new Pair<T1,T2>( lst1.get(i), lst2.get(i) ) );
    }
    return lst;
  }
  public static <T1> Maybe<T1> find( Collection<T1> lst, Function<T1,Boolean> filter) {
    for( T1 elem : lst )
      if(filter.apply(elem)) return Maybe.Just(elem);
    return Maybe.Nothing();
  }
  public static <T1> boolean exists( Collection<T1> lst, Function<T1,Boolean> filter) {
    for( T1 elem : lst )
      if(filter.apply(elem)) return true;
    return false;
  }
  public static <T1> boolean forall( Collection<T1> lst, Function<T1,Boolean> filter) {
    for( T1 elem : lst )
      if(!filter.apply(elem)) return false;
    return true;
  }

  public static <T1,T2> List<T2> map( Iterable<T1> lst, Function<T1,T2> mapper) {
    List<T2> lst_ = new ArrayList<T2>();
    for( T1 elem : lst )
      lst_.add( mapper.apply(elem) );
    return lst_;
  }

  @SuppressWarnings("unchecked")
  public static <T1,T2> T2[] map( T1[] lst, Function<T1,T2> mapper) {
    List<T2> lst_ = new ArrayList<T2>(lst.length);
    for( T1 elem : lst ) {
      lst_.add( mapper.apply(elem) );
    }
    if (lst.length == 0) { return (T2[]) new Object[0]; }
    T2[] rtn = (T2[]) Array.newInstance(lst_.get(0).getClass(), lst_.size());
    return lst_.toArray(rtn);
  }

  public static <T1,T2> List<T2> lazyMap( final List<T1> lst, final Function<T1,T2> mapper) {
    return new AbstractList<T2>() {
      @Override
      public T2 get(int index) {
        return mapper.apply(lst.get(index));
      }
      @Override
      public int size() {
        return lst.size();
      }
    };
  }

  public static <T1,T2> Collection<T2> lazyMap( final Collection<T1> lst, final Function<T1,T2> mapper) {
    return new AbstractCollection<T2>() {
      @Override
      public Iterator<T2> iterator() {
        return new Iterator<T2>() {
          Iterator<T1> impl = lst.iterator();
          @Override
          public boolean hasNext() { return impl.hasNext(); }
          @Override
          public T2 next() { return mapper.apply(impl.next()); }
          @Override
          public void remove() { impl.remove(); }
        };
      }
      @Override
      public int size() {
        return lst.size();
      }
    };
  }

  public static <T1,T2> IterableIterator<T2> mapIgnoreNull( final Iterator<T1> lst, final Function<T1,T2> mapper) {
    return new IterableIterator<T2>(new Iterator<T2>(){
      private T2 next = null;
      @Override
      public boolean hasNext() {
        if (next == null) {
          while (lst.hasNext() && next == null) {
            T1 nextIn = lst.next();
            if (nextIn != null) { next = mapper.apply(nextIn); }
          }
          return next != null;
        } else {
          return true;
        }
      }
      @Override
      public T2 next() {
        if (!hasNext()) { throw new NoSuchElementException(); }
        T2 rtn = next;
        assert rtn != null;
        next = null;
        return rtn;
      }
      @Override
      public void remove() {
        throw new UnsupportedOperationException();
      }
    });
  }

  public static <T1,T2> IterableIterator<T2> flatMapIgnoreNull( final Iterator<T1> input, final Function<T1,Iterator<T2>> mapper) {
    return new IterableIterator<T2>(new Iterator<T2>(){
      private Iterator<T2> iter = null;
      private T2 next = null;

      private boolean hasNextInIter() {
        if (iter == null) { return false; }
        if (next == null) {
          while (iter.hasNext() && next == null) {
            next = iter.next();
          }
          return next != null;
        } else {
          return true;
        }
      }

      @Override
      public boolean hasNext() {
        while (!hasNextInIter()) {
          if (!input.hasNext()) { return false; }
          iter = mapper.apply(input.next());
        }
        return iter != null;
      }
      @Override
      public T2 next() {
        if (!hasNext()) { throw new NoSuchElementException(); }
        T2 toReturn = next;
        next = null;
        return toReturn;
      }
      @Override
      public void remove() {
        throw new UnsupportedOperationException();
      }
    });
  }



  public static <T1,T2> List<T2> concat( List<T1> lst, Function<T1,List<T2>> mapper) {
    List<T2> lst_ = new ArrayList<T2>(lst.size());
    for( T1 elem : lst )
      lst_.addAll(mapper.apply(elem));
    return lst_;
  }
  public static <T1,T2,T3> Map<T2,List<T3>> collect( List<T1> lst, Function<T1,Pair<T2,T3>> mapper) {
    Map<T2,List<T3>> map = new HashMap<T2,List<T3>>();
    for( T1 elem : lst ) {
      Pair<T2,T3> pair = mapper.apply(elem);
      if( !map.containsKey(pair.first) )
        map.put(pair.first, new ArrayList<T3>() );
      map.get(pair.first).add(pair.second);
    }
    return map;
  }
  public static <T1,T2,T3> Map<T2,Set<T3>> collectDistinct( List<T1> lst, Function<T1,Pair<T2,T3>> mapper) {
    Map<T2,Set<T3>> map = new HashMap<T2,Set<T3>>();
    for( T1 elem : lst ) {
      Pair<T2,T3> pair = mapper.apply(elem);
      if( !map.containsKey(pair.first) )
        map.put(pair.first, new HashSet<T3>() );
      map.get(pair.first).add(pair.second);
    }
    return map;
  }

  public static <T1> List<T1> filter(List<T1> lst, Function<T1,Boolean> filter) {
    List<T1> output = new ArrayList<T1>();
    for(T1 elem : lst) {
      if (filter.apply(elem)) output.add(elem);
    }
    return output;
  }

  public static <T1> Maybe<T1> find( Collection<T1> lst, T1 thing, Function<Pair<T1,T1>,Boolean> comparator ) {
    for( T1 elem : lst ) {
      if( comparator.apply( Pair.makePair(elem, thing) ) ) return Maybe.Just( thing );
    }
    return Maybe.Nothing();
  }

  public static <T1> Maybe<T1> overlap( Collection<T1> lst, Collection<T1> lst_ ) {
    for( T1 elem : lst ) {
      if( lst_.contains( elem ) ) return Maybe.Just(elem);
    }
    return Maybe.Nothing();
  }

  public static <T1> List<T1> allOverlaps( Collection<T1> lst, Collection<T1> lst_ ) {
    List<T1> overlaps = new ArrayList<T1>();
    for( T1 elem : lst ) {
      if( lst_.contains( elem ) ) overlaps.add( elem );
    }
    return overlaps;
  }

  /**
   * Partition on the keys returned by mapper
   */
  public static <T1,T2> Map<T2,List<T1>> partition( List<T1> lst, Function<T1,T2> mapper) {
    Map<T2,List<T1>> map = new HashMap<T2,List<T1>>();
    for( T1 elem : lst ) {
      T2 key = mapper.apply(elem);
      if( !map.containsKey(key) )
        map.put(key, new ArrayList<T1>() );
      map.get(key).add(elem);
    }
    return map;
  }

  // Removes all elements of lst2 from lst1
  public static <T1> void difference( List<T1> lst1, List<T1> lst2 ) {
    for(T1 elem : lst2) {
        lst1.remove(elem);
    }
  }

  /**
   * Create a sub list with just these indices
   */
  public static <T1> List<T1> subList( List<T1> lst, Collection<Integer> indices ) {
    List<T1> sublst = new ArrayList<T1>();
    for(Integer idx : indices)
      sublst.add(lst.get(idx));
    return sublst;
  }

  // -- Graph functions

  /**
   * Map each edge of the graph, retaining vertices
   */
  public static <T1,T2,T3> DirectedMultiGraph<T1,T3> mapEdges( DirectedMultiGraph<T1,T2> graph, Function<Triple<T1,T1,T2>,T3> mapper ) {
      DirectedMultiGraph<T1,T3>  graph_ = new DirectedMultiGraph<T1,T3>();
      for( T1 head : graph.getAllVertices() ) {
          for( T1 tail : graph.getChildren(head) ) {
              for( T2 edge : graph.getEdges(head, tail) ) {
                  graph_.add(head, tail, mapper.apply(Triple.makeTriple(head, tail, edge)));
              }
          }
      }
      return graph_;
  }

  /**
   * Map the set of edges between vertices of a graph, retaining vertices
   */
  public static <T1,T2,T3> DirectedMultiGraph<T1,T3> mapEdgeSets( DirectedMultiGraph<T1,T2> graph, Function<Triple<T1,T1,List<T2>>,List<T3>> mapper ) {
    DirectedMultiGraph<T1,T3>  graph_ = new DirectedMultiGraph<T1,T3>();
    for( T1 head : graph.getAllVertices() ) {
      for( T1 tail : graph.getChildren(head) ) {
        for( T3 edge_ : mapper.apply(Triple.makeTriple(head, tail, graph.getEdges(head, tail)))) {
          graph_.add(head, tail, edge_);
        }
      }
    }
    return graph_;
  }

  public static <T1,T2,T3,T4> DirectedMultiGraph<T3,T4> map( DirectedMultiGraph<T1,T2> graph, Function<Triple<T1,T1,List<T2>>,Triple<T3,T3,List<T4>> > mapper ) {
    DirectedMultiGraph<T3,T4>  graph_ = new DirectedMultiGraph<T3,T4>();
    for( T1 head : graph.getAllVertices() ) {
      for( T1 tail : graph.getChildren(head) ) {
        Triple<T3,T3,List<T4>> triple = mapper.apply(Triple.makeTriple(head, tail, graph.getEdges(head, tail)));
        for( T4 edge : triple.third ) {
          graph_.add(triple.first, triple.second, edge);
        }
      }
    }
    return graph_;
  }

  /**
   * Similar to map, but allows you to return a list, adding many edges for each edge between two vertices
   */
  public static <T1,T2,T3> DirectedMultiGraph<T1,T3> collectEdges( DirectedMultiGraph<T1,T2> graph, Function<Triple<T1,T1,T2>,List<T3>> mapper ) {
    DirectedMultiGraph<T1,T3>  graph_ = new DirectedMultiGraph<T1,T3>();
    for( T1 head : graph.getAllVertices() ) {
      for( T1 tail : graph.getChildren(head) ) {
        for( T2 edge : graph.getEdges(head, tail) ) {
          for( T3 edge_ : mapper.apply(Triple.makeTriple(head, tail, edge))) {
            graph_.add(head, tail, edge_);
          }
        }
      }
    }
    return graph_;
  }


  public static <T1,T2> List<Pair<T1,T1>> vertexPairs( DirectedMultiGraph<T1,T2> graph ) {
    List<Pair<T1,T1>> pairs = new ArrayList<Pair<T1,T1>>();
    for( T1 head : graph.getAllVertices() ) {
      for( T1 tail : graph.getChildren(head) ) {
        pairs.add( Pair.makePair(head, tail) );
      }
    }
    return pairs;
  }

  public static <T1,T2> List<Triple<T1,T1,List<T2>>> groupedEdges( DirectedMultiGraph<T1,T2> graph ) {
    List<Triple<T1,T1,List<T2>>> edges = new ArrayList<Triple<T1,T1,List<T2>>>();
    for( T1 head : graph.getAllVertices() ) {
      for( T1 tail : graph.getChildren(head) ) {
        edges.add( Triple.makeTriple(head, tail, graph.getEdges(head, tail)) );
      }
    }
    return edges;
  }

  public static <T> boolean equalOrBothUndefined(Maybe<T> x, Maybe<T> y) {
    if (x.isDefined() && y.isDefined()) {
      if (x.get() == null && y.get() == null) { assert false; return true; }
      else if (x.get() == null || y.get() == null) { assert false; return false; }
      else {
        if (Double.class.isAssignableFrom(x.get().getClass()) &&
            Double.class.isAssignableFrom(y.get().getClass())) {
          return Math.abs(((Double) x.get()) - ((Double) y.get())) < 1e-5;
        } else {
          return x.get().equals(y.get());
        }
      }
    } else return !x.isDefined() && !y.isDefined();
  }

  public static <T> boolean equalIfBothDefined(Maybe<T> x, Maybe<T> y) {
    if (x.isDefined() && y.isDefined()) {
      if (x.get() == null && y.get() == null) { assert false; return true; }
      else if (x.get() == null || y.get() == null) { assert false; return false; }
      else {
        if (Double.class.isAssignableFrom(x.get().getClass()) &&
            Double.class.isAssignableFrom(y.get().getClass())) {
          return Math.abs(((Double) x.get()) - ((Double) y.get())) < 1e-5;
        } else {
          return x.get().equals(y.get());
        }
      }
    } else {
      return true;
    }
  }

  private static <E> PriorityQueue<Triple<E, Double, Integer>> createFrontier(int size) {
    return new PriorityQueue<Triple<E, Double, Integer>>(size,
        new Comparator<Triple<E, Double, Integer>>(){
          @Override
          public int compare(Triple<E, Double, Integer> o1, Triple<E, Double, Integer> o2) {
            if (o1.second > o2.second) return -1;
            if (o1.second < o2.second) return 1;
            else return 0;
          }
        });
  }

  public static <E> IterableIterator<Pair<E, Double>> interleave(final IterableIterator<Pair<E, Double>>[] elements) {
    // -- Handle state
    final Set<E> queued = new HashSet<E>();
    final PriorityQueue<Triple<E, Double, Integer>> frontier = createFrontier(elements.length);

    // -- Set Up Search
    PriorityQueue<Triple<E, Double, Integer>> immediateFrontier = createFrontier(elements.length);
    // Load first results
    for (int queryIndex = 0; queryIndex < elements.length; ++queryIndex) {
      if (elements[queryIndex].hasNext()) {
        Pair<E, Double> next = elements[queryIndex].next();
        immediateFrontier.add(Triple.makeTriple(next.first, next.second, queryIndex));
      }
    }
    // Queue first results
    for (Triple<E, Double, Integer> term : immediateFrontier) {
      if (!queued.contains(term.first)) {
        frontier.add(term);
        queued.add(term.first);
      }
    }

    // -- Run [lazy] Search
    return new IterableIterator<Pair<E, Double>>(new Iterator<Pair<E, Double>>() {
      @Override
      public boolean hasNext() {
        return !frontier.isEmpty();
      }
      @Override
      public Pair<E, Double> next() {
        // Get next element
        Triple<E, Double, Integer> next = frontier.poll();
        int queryIndex = next.third;
        // Update queue
        boolean haveQueued = false;
        while (!haveQueued && elements[queryIndex].hasNext()) {
          Pair<E, Double> toQueue = elements[queryIndex].next();
          if (!queued.contains(toQueue.first)) {
            frontier.add(Triple.makeTriple(toQueue.first, toQueue.second, queryIndex));
            queued.add(toQueue.first);
            haveQueued = true;
          }
        }
        // Return
        return Pair.makePair(next.first, next.second);
      }
      @Override
      public void remove() {
        throw new UnsupportedOperationException();
      }
    });

  }

  public static int[] seq(int n) {
    int[] rtn = new int[n];
    for (int i = 0; i < n; ++i) { rtn[i] = i; }
    return rtn;
  }

  public static <E> void shuffleInPlace(E[] elems, Random rand) {
    for(int j = elems.length - 1; j > 0; j --){
      int randIndex = rand.nextInt(j+1);
      E tmp = elems[randIndex];
      elems[randIndex] = elems[j];
      elems[j] = tmp;
    }
  }

  /**
   * Creates an iterator from a function which continually produces Maybe's.
   * Perhaps somewhat counterintuitively (but fitting the use case for its creation),
   * the semantics of an element coming out of the factory are:
   *
   * -> Just(E): iterator returns E
   * -> Nothing: iterator skips over this element. ***it doesn't stop on Nothing!***
   * -> null:    iterator stops
   *
   * @param factory The function which creates Maybe's
   * @param <E> The return type
   * @return An iterator over non-Nothing Maybe's returned by the factory
   */
  @SuppressWarnings("UnusedDeclaration")
  public static <E> Iterator<E> iteratorFromMaybeFactory(final Factory<Maybe<E>> factory) {
    return new Iterator<E>() {

      private Maybe<E> nextElement = Maybe.Nothing();

      @Override
      public boolean hasNext() {
        if (nextElement.isDefined()) { return true; }
        while (nextElement != null && !nextElement.isDefined()) {
          nextElement = factory.create();
        }
        return nextElement != null;
      }

      @Override
      public E next() {
        if (!hasNext()) { throw new NoSuchElementException(); }
        E element = nextElement.get();
        nextElement = Maybe.Nothing();
        return element;
      }

      @Override
      public void remove() {
        throw new UnsupportedOperationException();
      }
    };
  }

  public static <E> Iterator<E> iteratorFromMaybeIterableFactory(final Factory<Maybe<Iterable<E>>> factory) {
    return new Iterator<E>() {

      private Maybe<Iterator<E>> nextIterator = Maybe.Nothing();
      private E nextElement;

      @Override
      public boolean hasNext() {
        // Already found the next element
        if (nextElement != null) { return true; }
        // Can get the next element from the iterator
        if (nextIterator.isDefined() && nextIterator.get().hasNext()) {
          nextElement = nextIterator.get().next();
          return true;
        }
        // Update iterator state
        if (nextIterator.isDefined() && !nextIterator.get().hasNext()) {
          nextIterator = Maybe.Nothing();
        }
        // Get a new iterator
        while (nextIterator != null && !nextIterator.isDefined()) {
          Maybe<Iterable<E>> next = factory.create();
          nextIterator = next == null ? null : next.isDefined() ? Maybe.Just(next.get().iterator()) : Maybe.<Iterator<E>>Nothing();
        }
        // End of the line
        if (nextIterator == null) { return false; }
        // Else try again with a new iterator
        return hasNext();  // stack depth should be the number of Nothing iterators returned.
      }

      @Override
      public E next() {
        if (!hasNext()) { throw new NoSuchElementException(); }
        E element = nextElement;
        nextElement = null;
        return element;
      }

      @Override
      public void remove() {
        throw new UnsupportedOperationException();
      }
    };
  }

  public interface EdgeRewriter<V,E> {
    public boolean sameEdge(E victorEdge, E candidateEdge);
    public boolean isValidOutgoingEdge(V victor, E edge);
    public E mergeEdges(E edge1, E edge2);
    public E rewrite(V pivot, V newValue, E edge);
  }

  /**
   * Merges the loser into the victor in the graph
   */
  public static<V,E> void mergeVertices( DirectedMultiGraph<V,E> graph, V victor, V loser,
                                         final EdgeRewriter<V,E> edgeRewriter) {
    if( victor.equals( loser ) ) return;

    startTrack("Merging vertices: " + victor + " <- " + loser );

    int originalVictorSize = graph.getChildren(victor).size();
    int originalLoserSize = graph.getChildren(loser).size();

    debug(String.format("Victor: %s starts with %d children", victor, originalVictorSize));
    debug(String.format("Loser: %s starts with %d children", loser, originalLoserSize));

    //V updatedVictor = edgeRewriter.mergeVertices(victor, loser);

    // 1. Copy over all edges.
    for( V child : new ArrayList<V>(graph.getChildren(victor)) ) {
      if( child.equals(loser) ) continue; // Don't add self-loops
      List<E> victorEdges = new ArrayList<E>(graph.getEdges( victor, child ));
      graph.removeEdges(victor, child);

      List<E> loserEdges = new ArrayList<E>(graph.getEdges( loser, child ));
      graph.removeEdges(loser, child);
      for( final E victorEdge : victorEdges ) {
        Maybe<E> loserEdge = CollectionUtils.find(loserEdges, new Function<E, Boolean>() {
          @Override
          public Boolean apply(E in) { return edgeRewriter.sameEdge(victorEdge, in); } });
        if(loserEdge.isDefined()) {
          loserEdges.remove(loserEdge.get());
          graph.add( victor, child, edgeRewriter.mergeEdges(victorEdge, loserEdge.get()) );
          // Merge
        } else {
          // Add to graph
          graph.add( victor, child, victorEdge );
        }
      }
      for( E loserEdge : loserEdges ) {
        E outgoingEdge = edgeRewriter.rewrite(child, victor, loserEdge);
        if (edgeRewriter.isValidOutgoingEdge(victor, outgoingEdge)) {
          graph.add(victor, child, outgoingEdge);
        }
      }
    }
    // Move the remaining edges of the loser
    for( V child : new ArrayList<V>(graph.getChildren( loser ) ) ) {
      if( child.equals(victor) ) continue;
      List<E> loserEdges = new ArrayList<E>(graph.getEdges( loser, child ));
      for( E loserEdge : loserEdges ) {
        E outgoingEdge = edgeRewriter.rewrite(child, victor, loserEdge);
        if (edgeRewriter.isValidOutgoingEdge(victor, outgoingEdge)) {
          graph.add(victor, child, outgoingEdge);
        }
      }
    }

    // Repeat except for parent nodes
    for( V parent : new ArrayList<V>(graph.getParents(victor)) ) {
      if( parent.equals(loser) ) continue; // Don't add self-loops
      List<E> victorEdges = new ArrayList<E>(graph.getEdges( parent, victor ));
      graph.removeEdges(parent, victor);

      List<E> loserEdges = new ArrayList<E>(graph.getEdges( parent, loser));
      graph.removeEdges(parent, loser);
      for( final E victorEdge : victorEdges ) {
        Maybe<E> loserEdge = CollectionUtils.find(loserEdges, new Function<E, Boolean>() {
          @Override
          public Boolean apply(E in) { return edgeRewriter.sameEdge(victorEdge, in); } });
        if(loserEdge.isDefined()) {
          loserEdges.remove(loserEdge.get());
          graph.add( parent, victor, edgeRewriter.mergeEdges(victorEdge, loserEdge.get()) );
          // Merge
        } else {
          // Add to graph
          graph.add( parent, victor, victorEdge );
        }
      }
      for( E loserEdge : loserEdges ) {
        E outgoingEdge = edgeRewriter.rewrite(parent, victor, loserEdge);
        if (edgeRewriter.isValidOutgoingEdge(parent, outgoingEdge)) {
          graph.add(parent, victor, outgoingEdge);
        }
      }
    }
    // 
    for( V parent : new ArrayList<V>(graph.getParents( loser ) ) ) {
      if( parent.equals(victor) ) continue;
      List<E> loserEdges = new ArrayList<E>(graph.getEdges( parent, loser ));
      for( E loserEdge : loserEdges ) {
        E outgoingEdge = edgeRewriter.rewrite(parent, victor, loserEdge);
        if (edgeRewriter.isValidOutgoingEdge(parent, outgoingEdge)) {
          graph.add(parent, victor, outgoingEdge);
        }
      }
    }

    // Now, finally delete the old vertex
    graph.removeVertex( loser );

    debug(String.format("Victor: %s ends with %d children", victor, graph.getChildren(victor).size()));
//    assert graph.getChildren(victor).size() >= Math.min(originalVictorSize, originalLoserSize);  // note(gabor): not true, if edges are filtered.
    assert graph.getChildren(victor).size() <= originalVictorSize + originalLoserSize;

    endTrack("Merging vertices: " + victor + " <- " + loser );
  }

  /**
   * Group elements in a list together by equivalence classes
   */
  public static<V> List<Set<V>> groupByEquivalence( Collection<V> lst,
      Function<Pair<V,V>,Boolean> comparator ) {

    // Create a map of items and things they are equivalent to
    Map<V, List<V>> equivalenceMatching = new HashMap<V,List<V>>();
    startTrack("Making equivalence lists");
    for( V elem : lst ) {
      equivalenceMatching.put( elem, new ArrayList<V>() );
      for( V elem_ : lst ) {
        if( elem.equals( elem_ ) ) continue; // don't add yourself
        if( comparator.apply( Pair.makePair( elem, elem_ ) ) ) {
          debug( String.format("Merging %s and %s", elem, elem_ ));
          equivalenceMatching.get( elem ).add( elem_ );
        }
      }
    }
    endTrack("Making equivalence lists");
    return groupByEquivalence(lst, equivalenceMatching);
  }

  public static <V,T> Map<T,List<V>> groupBy(List<V> lst, Function<V,T> selector) {
    Map<T,List<V>> groupedList = new HashMap<T,List<V>>();
    for( V elem : lst ) {
      T sel = selector.apply( elem );
      if( !groupedList.containsKey(sel) ) {
        groupedList.put( sel, new ArrayList<V>() );
      }
      groupedList.get(sel).add(elem);
    }

    return groupedList;
  }

  /**
   * Split a collection according to a selector.
   * @param lst The list to filter
   * @param selector A function determining whether the elements of the list should go into the first or second component.
   * @param <V> The type of the list.
   * @return A pair of lists, where the first contains elements for which selector is true, and the second contains elements for which it's false.
   */
  public static <V> Pair<List<V>,List<V>> split(final List<V> lst, Function<V,Boolean> selector) {
    List<V> lst1 = new ArrayList<V>();
    List<V> lst2 = new ArrayList<V>();
    for( V elem : lst ) {
      if( selector.apply( elem ) ) lst1.add(elem);
      else lst2.add(elem);
    }

    return Pair.makePair(lst1, lst2);
  }

  public static <V> void removeDuplicates (List<V> lst) {
    for(ListIterator<V>  it = lst.listIterator(); it.hasNext();) {
      int idx = it.nextIndex();
      V elem = it.next();
      if( idx < lst.size() - 1 &&
          lst.subList(idx+1,lst.size()-1).contains(elem) ) it.remove();
    }
  }

  /**
   * Group elements in a list together by equivalence classes
   */
  public static<V> List<Set<V>> groupByRankedEquivalence( Collection<V> lst,
                                                    Function<Pair<V,V>,Double> comparator ) {

    // Create a map of items and things they are equivalent to
    Map<V, List<V>> equivalenceMatching = new HashMap<V,List<V>>();
    startTrack("Making equivalence lists");
    for( V elem : lst ) {
      equivalenceMatching.put( elem, new ArrayList<V>() );
      ClassicCounter<V> candidates = new ClassicCounter<V>();
      for( V elem_ : lst ) {
        if( elem.equals( elem_ ) ) continue; // don't add yourself

        double score = comparator.apply( Pair.makePair( elem, elem_ ) );
        if( score == Double.POSITIVE_INFINITY )
          equivalenceMatching.get(elem).add(elem_);
        else
          candidates.setCount(elem_, score);
      }
      if( Counters.max(candidates) > 0.0 ) {
        debug("Best match for " + elem + " was " + Counters.argmax(candidates) + " : " + Counters.max(candidates) );
        V elem_ = Counters.argmax(candidates);
        debug(String.format("Merging %s and %s", elem, elem_));
        equivalenceMatching.get( elem ).add( elem_ );
      }
    }
    endTrack("Making equivalence lists");

    return groupByEquivalence(lst, equivalenceMatching);
  }

  public static<V> List<Set<V>> groupByEquivalence( Collection<V> lst,
                                                    Map<V, List<V>> equivalenceMatching ) {
    List<Set<V>> equivalenceClasses = new ArrayList<Set<V>>();

    startTrack("Flattening into equivalence classes");
    while( equivalenceMatching.keySet().iterator().hasNext() ) {
      // Get some item
      V item = equivalenceMatching.keySet().iterator().next();
      Set<V> equivalenceClass = new HashSet<V>();
      equivalenceClass.add(item);

      // Now merge everything that is equivalent to it.
      startTrack("Flattening entities equivalent to " + item );
      Queue<V> equivalentEntities = new LinkedList<V>( equivalenceMatching.get(item) );
      while( equivalentEntities.size() > 0 ) {
        V entity = equivalentEntities.poll();
        // Ignore if you've already merged this entity
        if( !equivalenceMatching.containsKey( entity ) ) continue;

        // Otherwise add to the equivalence class and queue of things to
        // be processed.
        equivalenceClass.addAll( equivalenceMatching.get( entity ) );
        equivalentEntities.addAll( equivalenceMatching.get( entity ) );
        equivalenceMatching.remove( entity );
      }
      // Finally remove this item
      equivalenceMatching.remove( item );
      endTrack("Flattening entities equivalent to " + item );
      equivalenceClasses.add( equivalenceClass );
    }
    endTrack("Flattening into equivalence classes");

    return equivalenceClasses;
  }

  public static<V> boolean all(Collection<V> lst, Function<V,Boolean> fn ) {
    for( V elem : lst )
      if( !fn.apply(elem) ) return false;
    return true;
  }

  public static<V> boolean any(Collection<V> lst, Function<V,Boolean> fn ) {
    for( V elem : lst )
      if( fn.apply(elem) ) return true;
    return false;
  }

}
