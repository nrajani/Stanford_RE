package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.graph.DirectedMultiGraph;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Function;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;
import edu.stanford.nlp.util.logging.Redwood;

import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * Implements a number of consistency steps
 */
public class GraphConsistencyPostProcessors {

  protected static final Redwood.RedwoodChannels logger = Redwood.channels("GraphConsistency");

  public static enum MergeStrategy {
    MAX,
    MAX_PLUS,
    MEAN,
    NOISY_OR;

    public static MergeStrategy fromString(String s) {
        if (s.equals("max"))
          return MergeStrategy.MAX;
        else if (s.equals("max+"))
          return MergeStrategy.MAX_PLUS;
        else if (s.equals("max_plus"))
          return MergeStrategy.MAX_PLUS;
        else if (s.equals("mean"))
          return MergeStrategy.MEAN;
        else if (s.equals("noisy_or"))
          return MergeStrategy.NOISY_OR;
        else
          throw new RuntimeException("ERROR: Unknown merge strategy: " + s);
    }
  }


  /**
   * This processor collapses entities in a graph which name match
   * fairly accurately _and_ share some parent/child in the graph.
   */
  public static class EntityMergingPostProcessor extends GraphConsistencyPostProcessor {

    @Override
    public EntityGraph postProcess(EntityGraph graph, GoldResponseSet checklist) {
      startTrack("Collapse entities");

      // For each entity create a list of entities it is equivalent to.
      // This may involve potentially expensive entity linking calls, so try to minimize
      // the number of calls made.
      // (variables)
      Map<KBPEntity,Set<KBPEntity>> equivalenceClasses = new IdentityHashMap<KBPEntity,Set<KBPEntity>>();
      List<KBPEntity> entities = new ArrayList<KBPEntity>(graph.getAllVertices());
      for (KBPEntity entity1 : entities) { equivalenceClasses.put(entity1, new HashSet<KBPEntity>()); equivalenceClasses.get(entity1).add(entity1); }
      boolean[] deadEntities = new boolean[entities.size()];
      // (collect "priority" for certain entities)
      Counter<KBPEntity> priorities = new ClassicCounter<KBPEntity>();
      for (KBPEntity vertex : graph.getAllVertices()) {
        if (vertex instanceof KBPOfficialEntity) {
          for (KBPSlotFill fill : graph.getOutgoingEdges(vertex)) {
            for (KBPEntity slotEntity : fill.key.getSlotEntity()) {
              priorities.incrementCount(slotEntity, fill.score.getOrElse(0.0));
            }
          }
        }
      }
      // (create entity contexts)
      List<EntityContext> contexts = new ArrayList<EntityContext>();
      for (KBPEntity entity1 : entities) {
        contexts.add(new EntityContext(entity1, graph.getOutgoingEdges(entity1)));
      }
      // (do matching)
      for (int first = 0; first < entities.size(); ++first) {
        if (deadEntities[first]) { continue; }
        for (int second = first + 1; second < entities.size(); ++second) {
          if (deadEntities[second]) { continue; }
          if (Props.KBP_ENTITYLINKER.sameEntity(contexts.get(first), contexts.get(second))) {
            equivalenceClasses.get(entities.get(first)).addAll(equivalenceClasses.get(entities.get(second)));
            deadEntities[second] = true;
          }
        }
      }
      // (remove dead equivalence classes)
      for (int i = 0; i < deadEntities.length; ++i) {
        if (deadEntities[i]) { equivalenceClasses.remove(entities.get(i)); }
      }
      log( equivalenceClasses.size() + " equivalence classes found." );

      // Merge each of these lists one by one.
      for( Set<KBPEntity> eqClass : equivalenceClasses.values() ) {
        // Get the representative mention
        KBPEntity representative = getRepresentativeEntity( eqClass, priorities );
        logger.channels(DBG).logf("Using %s as the representative for %d entities", representative, eqClass.size());
        // HACK: get the provenance from the sentence that gives these two entities a common edge
        Map<KBPEntity, KBPRelationProvenance> provenanceMap = getProvenanceMap(graph, eqClass, representative);

        for( KBPEntity entity : eqClass ) {
          if( entity.equals( representative ) ) { continue; }  // Ignore yourself
          CollectionUtils.mergeVertices( graph, representative, entity, edgeRewriter );
        }
        // Now add all of those entities as alternative names!
        if( Props.TEST_GRAPH_ALTNAMES_DO && representative.type.isEntityType() ) {
          for( KBPEntity entity : eqClass ) {
            if( entity.equals( representative ) ) { continue; }  // Ignore yourself
            if (!Utils.isValidAlternateName(representative.name, entity.name)) { continue; }  // too close to be an alternate name
            KBPSlotFill toAdd = KBPNew.from(representative).slotValue(entity.name).slotType(entity.type)
                .rel((representative.type.equals(NERTag.ORGANIZATION)) ?
                    RelationType.ORG_ALTERNATE_NAMES.canonicalName :
                    RelationType.PER_ALTERNATE_NAMES.canonicalName)
                .provenance(provenanceMap.get(entity))
                .score(Double.POSITIVE_INFINITY).KBPSlotFill();
            graph.add(representative, entity, toAdd);
          }
        }
      }

      endTrack("Collapse entities");

      return graph;
    }

    /**
     * Try to find provanence linking the entity to the representative
     */
    protected Maybe<KBPRelationProvenance> findProvenance(DirectedMultiGraph<KBPEntity, KBPSlotFill> graph, KBPEntity entity, KBPEntity representative) {
      // Try to see if there is a common slot with the representative
      Set<KBPEntity> childrenOfEntity = graph.getChildren(entity);
      Set<KBPEntity> childrenOfRepresentative = graph.getChildren(representative);
      if(childrenOfEntity != null && childrenOfRepresentative != null) {
        for( KBPEntity common: CollectionUtils.allOverlaps(childrenOfEntity, childrenOfRepresentative) ) {
          for( KBPSlotFill fill : graph.getEdges(entity, common) )
            if( fill.provenance.isDefined() ) return fill.provenance;
        }
      }
      Set<KBPEntity> parentsOfEntity = graph.getParents(entity);
      Set<KBPEntity> parentsOfRepresentative = graph.getParents(representative);
      if(parentsOfEntity != null && parentsOfRepresentative != null) {
        for( KBPEntity common: CollectionUtils.allOverlaps(parentsOfEntity, parentsOfRepresentative) ) {
          for( KBPSlotFill fill : graph.getEdges(common,entity) )
            if( fill.provenance.isDefined() ) return fill.provenance;
        }
      }

      return Maybe.Nothing();
    }

    /**
     * Get a map of all the provenances you can possibly get
     */
    protected Map<KBPEntity, KBPRelationProvenance> getProvenanceMap(DirectedMultiGraph<KBPEntity, KBPSlotFill> graph, Set<KBPEntity> eqClass, KBPEntity representative) {
      Map<KBPEntity, KBPRelationProvenance> provenanceMap = new HashMap<KBPEntity, KBPRelationProvenance>();
      for( KBPEntity entity : eqClass ) {
        if( entity.equals(representative) ) continue;
        Maybe<KBPRelationProvenance> provanence = findProvenance(graph, entity, representative);
        if( provanence.isDefined() ) {
          provenanceMap.put(entity, provanence.get());
          continue;
        }
        // Still have work to do; try every other guy
        for( KBPEntity entity_ : eqClass ) {
          if( entity.equals( entity_ ) || representative.equals(entity_) ) continue;
          provanence = findProvenance(graph, entity, entity_);
          if( provanence.isDefined() ) {
            provenanceMap.put(entity, provanence.get());
            //noinspection UnnecessaryContinue
            continue;
          }
        }
      }

      return provenanceMap;
    }


    public static CollectionUtils.EdgeRewriter<KBPEntity,KBPSlotFill> edgeRewriter = new CollectionUtils.EdgeRewriter<KBPEntity, KBPSlotFill>() {
      @Override
      public boolean sameEdge(KBPSlotFill victorEdge, KBPSlotFill candidateEdge) {
        return victorEdge.key.relationName.equals( candidateEdge.key.relationName );
      }

      @Override
      public boolean isValidOutgoingEdge(KBPEntity victor, KBPSlotFill edge) {
        for (RelationType rel : edge.key.tryKbpRelation()) {
          if (rel.isAlternateName() && edge.key.getSlotEntity().equalsOrElse(victor, false)) { return false; }
        }
        return true;
      }

      @Override
      public KBPSlotFill mergeEdges(KBPSlotFill edge1, KBPSlotFill edge2) {
        // Don't worry about entity rewrites

        double newScore;
        switch( Props.TEST_GRAPH_MERGE_STRATEGY ) {
          case MAX:
            newScore = Math.max( edge1.score.getOrElse(0.0), edge2.score.getOrElse(0.0) );
            break;
          case MAX_PLUS:
            newScore = Math.max( edge1.score.getOrElse(0.0), edge2.score.getOrElse(0.0) ) + 1.0;
            break;
          case MEAN:
            // this isn't exactly a fair mean, but meh
            newScore = (edge1.score.getOrElse(0.0) + edge2.score.getOrElse(0.0))/2.0;
            break;
          case NOISY_OR:
            // this isn't exactly a fair mean, but meh
            newScore = 1.0 - (1.0 - edge1.score.getOrElse(0.0)) * (1.0 - edge2.score.getOrElse(0.0));
            if( !(0.0 <= newScore && newScore <= 1.0) ) { // It had better be a valid probability
              log( "Could not do noisy or - relation weights are not a probability" );
              debug( edge1 + " + " + edge2 );
              newScore = Math.max(edge1.score.getOrElse(0.0), edge2.score.getOrElse(0.0));
            }
            break;
          default:
            throw new RuntimeException("Invalid merge strategy");
        }
        return KBPNew.from(edge1).score( newScore ).KBPSlotFill();
      }

      @Override
      public KBPSlotFill rewrite(KBPEntity pivot, KBPEntity newValue, KBPSlotFill edge) {
        //noinspection AssertWithSideEffects
        assert( edge.key.getEntity().equals(pivot) || edge.key.getSlotEntity().equalsOrElse(pivot, false) );
        if( edge.key.getEntity().equals(pivot) ) {
          return KBPNew.from(edge).slotValue(newValue.name).slotType(edge.key.slotType.getOrElse(newValue.type)).KBPSlotFill();
        } else {
          return KBPNew.from(edge).entity(newValue).KBPSlotFill();
        }
      }
    };

    /**
     * Get a representative entity for an equivalence class. The representative should ideally have the following properties:
     * <ul>
     *      <li> Have a query id (official name; don't touch it). </li>
     *      <li> Have the highest "priority." This is conventionally determined by the sum incoming score to this entity from an official entity.</li>
     *      <li> Have the best defined type (more useful than name). </li>
     *      <li> Have the longest possible name (hopefully that'll be canonical). </li>
     * </ul>
     * @param entities - Equivalence class of entities
     * @return - The representative element matching the above properties.
     */
    protected KBPEntity getRepresentativeEntity( Collection<KBPEntity> entities, Counter<KBPEntity> priorities ) {
      assert( entities.size() > 0 );
      Iterator<KBPEntity> it = entities.iterator();

      KBPEntity representative = it.next();
      while( it.hasNext() ) {
        KBPEntity entity = it.next();
        if (entity instanceof KBPOfficialEntity) {
          KBPOfficialEntity candidate = (KBPOfficialEntity) entity;
          if(representative instanceof  KBPOfficialEntity) {
            KBPOfficialEntity representative_ = (KBPOfficialEntity) representative;
            // Choose the one with the query id
            if( candidate.queryId.isDefined() && ! representative_.queryId.isDefined() ) {
              representative = entity;
            // Choose the one with the id
            } else  if( candidate.id.isDefined() && ! representative_.id.isDefined() ) {
              representative = entity;
            // Choose the one with the higher priority
            } else if (priorities.getCount(candidate) > priorities.getCount(representative_)) {
              representative = entity;
            // Choose the one with the better type
            } else  if( candidate.type.isEntityType() && ! representative_.type.isEntityType() ) {
              representative = entity;
            // Choose the one with the longer name
            } else if( candidate.name.length() > representative_.name.length() ) {
              representative = entity;
            }
          } else {
            // Always choose the official entity
            representative = entity;
          }
        } else {
          // Always choose the official entity
          if(representative instanceof KBPOfficialEntity) { continue; }
          // Chose the one with the higher priority
          if (priorities.getCount(entity) > priorities.getCount(representative)) {
            representative = entity;
          } else if (priorities.getCount(entity) < priorities.getCount(representative)) {
            //noinspection UnnecessaryContinue
            continue;
          } else {
            // Choose the one with the better type
            if( entity.type.isEntityType() && !representative.type.isEntityType() ) {
              representative = entity;
              // Choose the one with the longer name
            } else if (entity.name.length() > representative.name.length()) {
              representative = entity;
            }
          }
        }
      }

      return representative;
    }
  }


  /**
   * Complete the missing relations for transitive relations.
   * Particularly, follow the trail of alternate names
   * TODO(gabor) perhaps this should fall out of Markov Logic?
   */
  public static class TransitiveRelationPostProcessor extends GraphConsistencyPostProcessor {
    public static final Collection<RelationType> transitiveRelations = Arrays.asList(
        RelationType.ORG_ALTERNATE_NAMES,
        RelationType.PER_ALTERNATE_NAMES
    );

    @Override
    public EntityGraph postProcess(EntityGraph graph, GoldResponseSet checklist) {
      for (KBPEntity node1 : graph.getAllVertices()) {
        for (KBPSlotFill edge1to2 : graph.getOutgoingEdges(node1)) {
          if (!edge1to2.key.hasKBPRelation()) { continue; }  // ignore unofficial relations
          if (!transitiveRelations.contains(edge1to2.key.kbpRelation())) { continue; }
          for (KBPEntity node2 : edge1to2.key.getSlotEntity()) {
            for (KBPSlotFill edge2to3 : graph.getOutgoingEdges(node2)) {
              if (!edge2to3.key.hasKBPRelation()) { continue; }  // ignore unofficial relations
              if (edge1to2.key.kbpRelation() != edge2to3.key.kbpRelation() ||
                  !transitiveRelations.contains(edge2to3.key.kbpRelation())) { continue; }
              for (KBPEntity node3 : edge2to3.key.getSlotEntity()) {
                graph.add(node1, node3, KBPNew.from(node1).slotValue(node3).rel(edge1to2.key.kbpRelation())
                    .provenance(edge2to3.provenance.orElse(edge1to2.provenance))
                    .score(Math.min(edge1to2.score.getOrElse(1.0), edge2to3.score.getOrElse(1.0))).KBPSlotFill()
                );
              }
            }
          }
        }
      }
      return graph;
    }
  }




  /**
   * Add edges for all sorts of symmetric relations
   */
  // TODO(arun): Rename to bijective
  public static class SymmetricFunctionRewritePostProcessor extends GraphConsistencyPostProcessor {

    public static final Map<RelationType,RelationType> symmetricPairs;
    static {
      // Populated from the KBP specification inverse slots table
      symmetricPairs = new HashMap<RelationType,RelationType>();
      symmetricPairs.put( RelationType.PER_CHILDREN, RelationType.PER_PARENTS );
      symmetricPairs.put( RelationType.PER_PARENTS, RelationType.PER_CHILDREN );
      symmetricPairs.put( RelationType.PER_OTHER_FAMILY, RelationType.PER_OTHER_FAMILY );
      symmetricPairs.put( RelationType.PER_SIBLINGS, RelationType.PER_SIBLINGS );
      symmetricPairs.put( RelationType.PER_SPOUSE, RelationType.PER_SPOUSE );
      // Not really true
      // symmetricPairs.put( RelationType.PER_EMPLOYEE_OF, RelationType.ORG_TOP_MEMBERS_SLASH_EMPLOYEES );
      // symmetricPairs.put( RelationType.PER_MEMBER_OF, RelationType.ORG_TOP_MEMBERS_SLASH_EMPLOYEES );
      symmetricPairs.put( RelationType.ORG_TOP_MEMBERS_SLASH_EMPLOYEES, RelationType.PER_EMPLOYEE_OF );
      //symmetricPairs.put( RelationType.PER_EMPLOYEE_OF, RelationType.PER_MEMBER_OF );
      //symmetricPairs.put( RelationType.PER_MEMBER_OF, RelationType.PER_EMPLOYEE_OF );
      symmetricPairs.put( RelationType.ORG_PARENTS, RelationType.ORG_SUBSIDIARIES );
      symmetricPairs.put( RelationType.ORG_SUBSIDIARIES, RelationType.ORG_PARENTS );
    }

    @Override
    public EntityGraph postProcess(EntityGraph graph, GoldResponseSet checklist) {
      startTrack("Symmetrize relations");

      // Go through each edge in the graph, and if the symmetric version exists, leave it be,
      // else add it with the same weight and justification
      for( Triple<KBPEntity,KBPEntity,List<KBPSlotFill>> edge : CollectionUtils.groupedEdges(graph) ) {
        KBPEntity head = edge.first;
        KBPEntity tail = edge.second;

        // Keep track of this list separately (else face the wrath of a concurrent modification error
        List<KBPSlotFill> additionalSlots = new ArrayList<KBPSlotFill>();

        for( KBPSlotFill fill : edge.third ) {
          for( RelationType reln : fill.key.tryKbpRelation() ) {
            if( symmetricPairs.containsKey(reln) ) {
              final RelationType reln_ = symmetricPairs.get(reln);
              // Check if the symmetric relation already holds
              if( !CollectionUtils.find( graph.getOutgoingEdges(tail), new Function<KBPSlotFill, Boolean>() {
                @Override
                public Boolean apply(KBPSlotFill in) {
                  return in.key.tryKbpRelation().equalsOrElse( reln_, false );
                }
              }).isDefined() ) {
                // No? Let's add it
                additionalSlots.add(KBPNew.from(tail).slotValue(head.name).slotType(head.type).rel(reln_).provenance(fill.provenance).score(fill.score).KBPSlotFill());
              }
            }
          }
        }
        for( KBPSlotFill newEdge : additionalSlots )
          graph.add(tail, head, newEdge);
      }

      endTrack("Symmetrize relations");

      return graph;
    }
  }



  /**
   * Run a unary post processor on every element of the graph
   */
  public static class UnaryConsistencyPostProcessor extends GraphConsistencyPostProcessor {
    public final Maybe<KBPOfficialEntity> queryEntity;
    public final SlotfillPostProcessor unaryProcessor;

    public UnaryConsistencyPostProcessor(KBPOfficialEntity queryEntity, SlotfillPostProcessor unaryProcessor) {
      this.unaryProcessor = unaryProcessor;
      this.queryEntity = Maybe.Just(queryEntity);
    }

    public UnaryConsistencyPostProcessor(SlotfillPostProcessor unaryProcessor) {
      this.unaryProcessor = unaryProcessor;
      this.queryEntity = Maybe.Nothing();
    }

    @Override
    public EntityGraph postProcess(EntityGraph graph, final GoldResponseSet goldResponses) {
      return EntityGraph.transformEdges(graph, new Function<Pair<KBPEntity, List<KBPSlotFill>>, List<KBPSlotFill>>() {
        @Override
        public List<KBPSlotFill> apply(Pair<KBPEntity, List<KBPSlotFill>> in) {
          KBPEntity entity = in.first;
          List<KBPSlotFill> relations = in.second;

          // Register raw slot fill predictions for debugging output
          if (queryEntity.equalsOrElse(in.first, false)) { for (KBPSlotFill fill : relations) { goldResponses.registerResponse(fill); } }

          // Run consistency only on KBP relations.
          Pair<List<KBPSlotFill>, List<KBPSlotFill>> groupedRelations = CollectionUtils.split(relations, new Function<KBPSlotFill, Boolean>() {
            @Override
            public Boolean apply(KBPSlotFill in) {
              return in.key.hasKBPRelation();
            }
          });
          List<KBPSlotFill> officialRelations = groupedRelations.first;
          List<KBPSlotFill> unofficialRelations = groupedRelations.second;
          List<KBPSlotFill> cleanRelations = queryEntity.equalsOrElse(entity, false)
              ? (Props.TEST_CONSISTENCY_DO ? unaryProcessor.postProcess(entity, officialRelations, goldResponses) : officialRelations)
              : (Props.TEST_CONSISTENCY_DO ? unaryProcessor.postProcess(entity, officialRelations) : officialRelations);
          cleanRelations.addAll(unofficialRelations);

          logger.debug("" + cleanRelations.size() + " slot fills remain after consistency (pass 1)");
          return cleanRelations;
        }
      });

    }
  }

}
