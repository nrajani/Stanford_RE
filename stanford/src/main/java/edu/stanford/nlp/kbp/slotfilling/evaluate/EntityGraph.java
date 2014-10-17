package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.graph.DirectedMultiGraph;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.util.Function;
import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.Pair;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

import java.util.*;

/**
 * Specialized graph between entities and slot fills
 * TODO(arun): This graph is really somewhat of a hack in terms of how I extract out
 * entities and types from it. It could really be improved.
 */
public class EntityGraph extends DirectedMultiGraph<KBPEntity, KBPSlotFill> {
  public EntityGraph() {
    super();
  }

  public EntityGraph(DirectedMultiGraph<KBPEntity,KBPSlotFill> graph) {
    super(graph);
  }

  @Override
  public boolean addVertex(KBPEntity entity) {
      return super.addVertex( entity );
  }


  /**
   * When adding a new edge, do two things:
   *  i. find canonical entities if possible.
   *  ii. if the relation between the two exists, merge the edge
   * @param source The source of the edge
   * @param destination The destination of the edge
   * @param edge The slot fill corresponding to this edge, representing the relation and provenance
   *             for the source -> destination edge.
   */
  @Override
  public void add(KBPEntity source, KBPEntity destination, KBPSlotFill edge) {
    // Error checks
    if (!edge.key.entityName.equals(source.name)) {
      throw new IllegalArgumentException("Source does not match edge's source. source=" + source + " edge=" + edge);
    }
    if (!edge.key.entityType.equals(source.type)) {
      throw new IllegalArgumentException("Source type does not match edge's source type. source=" + source + " edge=" + edge);
    }
    if (!edge.key.slotValue.equals(destination.name)) {
      throw new IllegalArgumentException("Destination does not match edge's destionation. destination=" + source + " edge=" + edge);
    }
    if (Utils.assertionsEnabled() && !edge.key.getEntity().equals(source)) {
      throw new AssertionError("Entity is not start of edge. source=" + source + " edge.entity=" + edge.key.getEntity());
    }

    // Now check if the slot type already exists; if so, then merge it.
    List<KBPSlotFill> mergableEdges = new ArrayList<KBPSlotFill>();
    for( KBPSlotFill edge_ : getEdges(source, destination) ) {
      if( edge_.key.relationName.equals(edge.key.relationName) ) mergableEdges.add(edge);
    }
    for( KBPSlotFill edge_ : mergableEdges ) {
      // Firstly remove the edge.
      removeEdge(source, destination, edge_);
      // Then merge this with that.
      edge = mergeEdges( edge, edge_ );
    }
    //noinspection AssertWithSideEffects
    assert( edge.key.getEntity().equals(source) );
    assert( edge.key.slotValue.equals(destination.name) );
    super.add( source, destination, edge );

    assert containsVertex(source);
    assert containsVertex(destination);
  }
  public void add(KBPSlotFill edge) {
    add( edge.key.getEntity(), edge.key.getSlotEntity().orCrash(), edge );
  }

  /**
   * Merge edge2 into edge1 - edge1 has been canonicalized, so always keep it.
   */
  private KBPSlotFill mergeEdges( KBPSlotFill edge1, KBPSlotFill edge2 ) {
    // Noisy or the scores and update the provenance
    Maybe<Double> newScore = Utils.noisyOr(edge1.score, edge2.score);
    // Choose best provenance
    Maybe<KBPRelationProvenance> bestProvenance;
    if( edge1.provenance.isDefined() && edge2.provenance.isDefined() )
      bestProvenance =  edge1.provenance.get().score.getOrElse(0.0) > edge2.provenance.get().score.getOrElse(0.0)
        ? edge1.provenance
        : edge2.provenance;
    else
      bestProvenance =  edge1.provenance.orElse(edge2.provenance);

    return KBPNew.from(edge1).score(newScore).provenance(bestProvenance).KBPSlotFill();
  }

  public Iterable<KBPSlotFill> relation(final KBPEntity entity, final String reln) {
    final EntityGraph graph = this;
    return new Iterable<KBPSlotFill>() {
      @Override
      public Iterator<KBPSlotFill> iterator() {
        return new RelationTypeIterator(graph, entity, reln);
      }
    };
  }
  public Iterable<KBPSlotFill> relation(final KBPEntity entity, final RelationType reln) {
    return relation(entity, reln.canonicalName);
  }
  public Iterable<KBPSlotFill> relation(final String reln) {
    final EntityGraph graph = this;
    return new Iterable<KBPSlotFill>() {
      @Override
      public Iterator<KBPSlotFill> iterator() {
        return new RelationTypeIterator(graph, reln);
      }
    };
  }

  public boolean isValidGraph() {
    for(KBPEntity head : getAllVertices() ) {
      for(KBPEntity tail : getAllVertices() )  {
        for(KBPSlotFill fill : getEdges(head, tail) ) {
          KBPEntity head_ = fill.key.getEntity();
          KBPEntity tail_ = fill.key.getSlotEntity().orCrash();

          if( !(head_.equals(head) && tail_.equals(tail) ) ) {
            err(String.format("Types do not match!: %s vs %s and %s vs %s", head, head_, tail, tail_ ) );
            if( !( head_.name.equals(head.name) && tail_.name.equals(tail.name) ) ) {
              return false;
//            } else {
//              warn(String.format("Types do not match!: %s vs %s and %s vs %s", head, head_, tail, tail_ ) );
            }
          }
        }
      }
    }
    return true;
  }

  public void addRelations(final Iterable<KBPSlotFill> newFills ) {
    for(KBPSlotFill fill : newFills ) {
      KBPEntity head = fill.key.getEntity();
      KBPEntity tail = fill.key.getSlotEntity().orCrash();
      add( head, tail, fill );
    }
  }

  public NERTag guessType( KBPEntity node ) {
    for( KBPSlotFill fill : getIncomingEdges(node) ) {
      if( fill.key.slotType.isDefined() ) return fill.key.slotType.get();
    }
    return node.type;
  }

  public Maybe<KBPEntity> findEntity( String name ) {
    for(KBPEntity entity : getAllVertices() ) {
      if( entity.name.equals( name ) ) return Maybe.Just(entity);
    }
    return Maybe.Nothing();
  }
  public Maybe<KBPEntity> findEntity( KBPEntity entity ) {
    return getAllVertices().contains( entity ) 
      ? Maybe.Just(entity)
      : findEntity(entity.name);
  }

  public Maybe<KBPEntity> findEntity( KBPEntity head, String name ) {
    for(KBPEntity entity : getChildren(head) ) {
      if( entity.name.equals( name ) ) return Maybe.Just(entity);
    }
    return Maybe.Nothing();
  }

  /**
   * Get the connected component of a graph from a given root.
   * This will run a search from the root and keep everything that has a path to the root.
   * @param root The entity to start the search from.
   * @return A set consisting of the entities in this connected component.
   */
  public Set<KBPEntity> getConnectedComponent(KBPEntity root) {
    Set<KBPEntity> component = new IdentityHashSet<KBPEntity>();
    component.add(root);
    dfs(component, root);
    return new HashSet<KBPEntity>(component);
  }

  private void dfs( Set<KBPEntity> component, KBPEntity node) {
    for( KBPEntity child : getChildren(node) ) {
      if( !component.contains(child) ) {
        component.add(child);
        dfs(component, child);
      }
    }
  }

  /**
   * Throws out all vertices not in the main component.
   * @param mainComponent The set of entities to keep
   */
  public void restrictGraph(Set<KBPEntity> mainComponent) {
    List<KBPEntity> complement = new ArrayList<KBPEntity>();
    for( KBPEntity node : getAllVertices() )
      if( !mainComponent.contains(node) ) complement.add(node);
    removeVertices(complement);
  }

  public static class RelationTypeIterator implements Iterator<KBPSlotFill> {
    String relationName;
    Iterator<KBPSlotFill> it;
    KBPSlotFill peek; // Ugh

    RelationTypeIterator(EntityGraph graph, KBPEntity pivot, String relationName) {
      this.relationName = relationName;
      it = graph.getOutgoingEdges(pivot).iterator();
      forwardToMatch();
    }
    RelationTypeIterator(EntityGraph graph, String relationName) {
      this.relationName = relationName;
      it = graph.getAllEdges().iterator();
      forwardToMatch();
    }

    public void forwardToMatch() {
      peek = null;
      while(it.hasNext()) {
        peek = it.next();
        if(peek.key.relationName.equals(relationName)) break;
      }
    }

    @Override
    public boolean hasNext() {
      return it.hasNext();
    }

    @Override
    public KBPSlotFill next() {
      KBPSlotFill fill = peek;
      forwardToMatch();
      return fill;
    }

    @Override
    public void remove() {
      throw new IllegalArgumentException("Can't remove");
    }
  }

  public static EntityGraph transformEdges( EntityGraph graph, Function<Pair<KBPEntity,List<KBPSlotFill>>,List<KBPSlotFill>> mapper ) {
    EntityGraph graph_ = new EntityGraph();
    for( KBPEntity head : graph.getAllVertices() ) {
      graph_.addVertex(head);
      for( KBPSlotFill edge_ : mapper.apply(Pair.makePair(head, graph.getOutgoingEdges(head) ) ) ) {
        assert( edge_.key.getEntity().equals(head) );
        KBPEntity tail = edge_.key.getSlotEntity().orCrash();
        graph_.add(head, tail, edge_);
      }
    }
    return graph_;
  }


}
