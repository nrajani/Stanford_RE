package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.graph.DirectedGraph;
import edu.stanford.nlp.kbp.slotfilling.classify.HeuristicRelationExtractor;
import edu.stanford.nlp.kbp.slotfilling.classify.RelationClassifier;
import edu.stanford.nlp.kbp.slotfilling.common.CollectionUtils;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.kbp.slotfilling.ir.PostIRAnnotator;
import edu.stanford.nlp.kbp.slotfilling.process.EntityMentionAnnotator;
import edu.stanford.nlp.kbp.slotfilling.process.KBPProcess;
import edu.stanford.nlp.kbp.slotfilling.process.KBPProcess.AnnotateMode;
import edu.stanford.nlp.kbp.slotfilling.process.RelationMentionAnnotator;
import edu.stanford.nlp.kbp.slotfilling.process.SlotMentionAnnotator;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.AnnotationPipeline;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.util.logging.Redwood;

import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * An extension of the slot filler that uses some graph inference
 */
public class InferentialSlotFiller extends SimpleSlotFiller {

  protected static final Redwood.RedwoodChannels logger = Redwood.channels("Infer");
  protected GraphInferenceEngine graphInferenceEngine;

  public InferentialSlotFiller(Properties props,
                               KBPIR ir,
                               KBPProcess process,
                               RelationClassifier classify,
                               GoldResponseSet goldResponses
  ) {
    super( props, ir, process, classify, goldResponses );
    if( Props.TEST_GRAPH_INFERENCE_RULES.exists() ) {
      graphInferenceEngine = new GraphInferenceEngine( irComponent,
          GraphInferenceEngine.loadFromFile(Props.TEST_GRAPH_INFERENCE_RULES, Props.TEST_GRAPH_INFERENCE_RULES_CUTOFF ) );
    } else {
      logger.warn("Graph inference rules do not exist at path " + Props.TEST_GRAPH_INFERENCE_RULES );
    }
  }

  @Override
  public List<KBPSlotFill> fillSlots(final KBPOfficialEntity queryEntity) {
    startTrack("Annotating " + queryEntity);
    System.gc();
    logger.log( "Memory usage: " + Utils.getMemoryUsage() );

    // -- Raw Classification
    startTrack("Raw Classification");
    // -- Construct Graph
    // Read slot candidates from the index
    // Each of these tuples effectively represents (e_1, e_2, [datums]) where e_1 is the query entity.
    EntityGraph relationGraph = extractRelationGraph(queryEntity, Props.TEST_SENTENCES_PER_ENTITY, Props.TEST_GRAPH_DEPTH, Maybe.<Function<String, Boolean>>Nothing());
    assert relationGraph.containsVertex(queryEntity);
    assert relationGraph.isValidGraph();
    printRelations(relationGraph);
    logger.log( relationGraph.getOutDegree(queryEntity) + " relations in the graph" );
    endTrack("Raw Classification");


    // -- Run a first pass of consistency
    startTrack("Consistency Pass I");
    // Go about clearing each of the edges.
    relationGraph = new GraphConsistencyPostProcessors.UnaryConsistencyPostProcessor(queryEntity, SlotfillPostProcessor.unary(irComponent)).postProcess(relationGraph, goldResponses);
    assert relationGraph.isValidGraph();
    printRelations(relationGraph);
    logger.log( relationGraph.getOutDegree(queryEntity) + " relations in the graph" );
    endTrack("Consistency Pass I");

    // -- Run Simple Inference
    startTrack("Inference");
    // Merge vertices on the graph
    if( Props.TEST_GRAPH_MERGE_DO ) {
      relationGraph = new GraphConsistencyPostProcessors.EntityMergingPostProcessor().postProcess(relationGraph, goldResponses);
      assert relationGraph.containsVertex(queryEntity);
      assert relationGraph.isValidGraph();
    }
    // Prune out everything but the core component
    forceTrack("Pruning graph");
    relationGraph.restrictGraph(relationGraph.getConnectedComponent(queryEntity));
    assert relationGraph.containsVertex(queryEntity);
    //printGraph( queryEntity.name, relationGraph );
    logger.log( "Memory usage: " + Utils.getMemoryUsage() );
    printRelations(relationGraph);
    logger.log( relationGraph.getOutDegree(queryEntity) + " relations in the graph" );
    endTrack("Pruning graph");

    // Compute transitive completion of relations
    if (Props.TEST_GRAPH_TRANSITIVE_DO) {
      relationGraph = new GraphConsistencyPostProcessors.TransitiveRelationPostProcessor().postProcess(relationGraph, goldResponses);
      assert relationGraph.isValidGraph();
    }

    // Symmetric function rewrite by default
    if (Props.TEST_GRAPH_SYMMETERIZE_DO) {
      relationGraph = new GraphConsistencyPostProcessors.SymmetricFunctionRewritePostProcessor().postProcess(relationGraph, goldResponses);
      assert relationGraph.isValidGraph();
    }

    // -- Add Inference Rules
    if( Props.TEST_GRAPH_INFERENCE_DO ) {
      startTrack("Rules Inference");
      relationGraph = graphInferenceEngine.apply(relationGraph, queryEntity);
      assert relationGraph.isValidGraph();
      endTrack("Rules Inference");
    }
    assert relationGraph.isValidGraph();
    logger.log( "Memory usage: " + Utils.getMemoryUsage() );
    printRelations(relationGraph);
    logger.log( relationGraph.getOutDegree(queryEntity) + " relations in the graph" );
    endTrack("Inference");

    // -- Run final consistency checks
    startTrack("Consistency Pass II");
    // From now on, it's no more graph, all just the query entity
    List<KBPSlotFill> cleanRelations;

    if(!relationGraph.containsVertex(queryEntity)) {
      logger.err("Couldn't find query entity in graph!");
      cleanRelations = new ArrayList<KBPSlotFill>();
    } else {
      cleanRelations =
        CollectionUtils.filter(relationGraph.getOutgoingEdges(queryEntity), new Function<KBPSlotFill, Boolean>() {
          @Override
          public Boolean apply(KBPSlotFill in) {
            return in.key.hasKBPRelation();
          }
        });
    }
    //noinspection UnusedAssignment
    relationGraph = null; // Be nice to GC
    logger.log("" + cleanRelations.size() + " slot fills remain at end of inference");

    // Re-run step1 consistency
    cleanRelations
        = Props.TEST_CONSISTENCY_DO ? SlotfillPostProcessor.unary(irComponent).postProcess(queryEntity, cleanRelations, goldResponses) : cleanRelations;
    // Find missing provenances
    List<KBPSlotFill> withProvenance;
    withProvenance = new ArrayList<KBPSlotFill>(cleanRelations.size());
    for (KBPSlotFill fill : cleanRelations) {
      KBPSlotFill augmented = KBPNew.from(fill).provenance(findBestProvenance(queryEntity, fill)).KBPSlotFill();
      if (augmented.provenance.isDefined() && (!Props.TEST_PROVENANCE_DO || augmented.provenance.get().isOfficial())) {
        withProvenance.add(augmented);
      } else {
        goldResponses.discardNoProvenance(fill);
      }
    }
    logger.log("" + withProvenance.size() + " slot fills remain after provenance");

    // Run consistency pass 2
    List<KBPSlotFill> consistentRelations = this.finalConsistencyAndProvenancePass(queryEntity, withProvenance, goldResponses);
    endTrack("Consistency Pass II");

    printRelations(queryEntity, consistentRelations);
    logger.log("Memory usage: " + Utils.getMemoryUsage());

    // -- Print Judgements
    logger.prettyLog(goldResponses.loggableForEntity(queryEntity, Maybe.Just(irComponent)));

    logger.log( "Memory usage: " + Utils.getMemoryUsage() );
    System.gc();
    logger.log( "Memory usage: " + Utils.getMemoryUsage() );

    endTrack("Annotating " + queryEntity);

    return consistentRelations;
  }

  //
  // Public Utilities
  @SuppressWarnings("unchecked")
  public EntityGraph extractRelationGraph(KBPOfficialEntity entity, int documentsPerEntity, int graphDepth, Maybe<? extends Function<String, Boolean>> docidFilter) {
    // -- IR
    // Get supporting sentences
    // Get sentences number of documents and keep pulling out sentences
    // till you find sufficiently many of them.
    forceTrack("Querying IR");
    List<String> docids = irComponent.queryDocIDs(entity.name, entity.type,
        entity.representativeDocument.isDefined() ? new HashSet<String>(Arrays.asList(entity.representativeDocument.get())) : new HashSet<String>(),
        documentsPerEntity * 5);
    List<Annotation> rawDocuments = new ArrayList<Annotation>();
    Iterator<String> docidIter = docids.iterator();
    while (rawDocuments.size() < documentsPerEntity && docidIter.hasNext()) {
      String docid = docidIter.next();
      if (!docidFilter.isDefined() || docidFilter.get().apply(docid)) { rawDocuments.add(irComponent.fetchDocument(docid)); }
    }
    logger.log("fetched " + rawDocuments.size() + " documents");
    endTrack("Querying IR");
    if( rawDocuments.size() == 0 ) {
      logger.warn("No documents found :-/!");
      return new EntityGraph();
    }

    // Begin constructing the graph
    forceTrack("Construct graph");

    // -- First pass construct a graph using the relation classifier
    EntityGraph graph = extractRelationGraphWithClassifer(entity, rawDocuments, graphDepth, 2 * Props.TEST_GRAPH_MAXSIZE / rawDocuments.size());
    assert graph.isValidGraph();
    logger.log(BOLD, "Num Edges: " + graph.getNumEdges() );
    logger.log("Memory usage: " + Utils.getMemoryUsage() );

    // -- Second pass for Reverb
    if( Props.TEST_GRAPH_REVERB_DO ) {
      // Take care of the annotation required
      AnnotationPipeline pipeline = new AnnotationPipeline();
      pipeline.addAnnotator(new PostIRAnnotator(entity.name, Maybe.Just(entity.type.name), Maybe.<String>Nothing(), Maybe.<String>Nothing(), true));
      pipeline.addAnnotator(new EntityMentionAnnotator(entity));
      pipeline.addAnnotator(new SlotMentionAnnotator());
      pipeline.addAnnotator(new RelationMentionAnnotator(entity, Collections.EMPTY_LIST, AnnotateMode.ALL_PAIRS));

      forceTrack("Augmenting with Reverb extractions");
      for( Annotation doc : rawDocuments ) {
        // Annotate with the details
        doc = CoreMapUtils.copyDocument(doc);
        pipeline.annotate(doc);
        // Extract relations using ReVerb
        // TODO(gabor) removed in DEFT repository
      }
      endTrack("Augmenting with Reverb extractions");
    }
    assert graph.isValidGraph();
    logger.log(BOLD, "Num Edges: " + graph.getNumEdges() );
    logger.log( "Memory usage: " + Utils.getMemoryUsage() );
    endTrack("Construct graph");

    return graph;
  }

  public EntityGraph extractRelationGraphWithClassifer( KBPEntity entity, List<Annotation> documents, int graphDepth, int entityBudget) {
    DirectedGraph<KBPEntity,Pair<SentenceGroup,List<CoreMap>>> datumGraph = new DirectedGraph<KBPEntity,Pair<SentenceGroup,List<CoreMap>>>();

    // Construct datum
    startTrack("Collecting datums over graph");
    for( Annotation doc : documents ) {
      extractDatumGraphForClassifer(datumGraph, doc, entity, graphDepth, entityBudget);
    }
    logger.log("Num edges in datum graph: " + datumGraph.getNumEdges() );
    logger.log( "Memory usage: " + Utils.getMemoryUsage() );
    endTrack("Collecting datums over graph");

    // Debug output
    if(!datumGraph.containsVertex(entity)) { // Nope, didn't find anything.
      return new EntityGraph();
    } else {
      int totalDatums = 0;
      for( Pair<SentenceGroup,List<CoreMap>> datums : datumGraph.getOutgoingEdges(entity) ) {
        totalDatums += datums.first.size();
      } {
        int sentenceGroupCount = datumGraph.getOutgoingEdges(entity).size();
        logger.log( String.format( "Found %d sentence groups (ave. %.2f sentences per group)", sentenceGroupCount, (double) totalDatums / ((double) sentenceGroupCount) ) );
      }
      logger.log( "Memory usage: " + Utils.getMemoryUsage() );
    }

    startTrack("Running classifier");
    EntityGraph graph = new EntityGraph();
    // HACK: Adding the original entity to the graph so that we'll always pick it up and use it as the canonical version.
    graph.addVertex(entity);
    // Run classifiers
    for( Pair<SentenceGroup,List<CoreMap>> pair : datumGraph.getAllEdges() ) {
      KBPEntity headEntity = pair.first.key.getEntity();
      SentenceGroup sg = pair.first;
      CoreMap[] sentences = pair.second.toArray(new CoreMap[pair.second.size()]);
      // Run relation extractor
      for( KBPSlotFill fill : findSlotsWithClassifier(sg, sentences) ) {
        assert(fill.key.entityName.equals(headEntity.name));
        graph.add( fill.key.getEntity(), fill.key.getSlotEntity().orCrash(), fill );
      }
      // Run rules
      for( KBPSlotFill fill : findSlotsWithRules(sg, sentences) ) {
        assert(fill.key.entityName.equals(headEntity.name));
        graph.add( fill.key.getEntity(), fill.key.getSlotEntity().orCrash(), fill );
      }
    }
    endTrack("Running classifier");

    return graph;
  }

  /**
   * Construct a graph over the entities in a sentence by "recursively" searching for the entities.
   * The entity graph represents a graph between entities linked by sentences.
   * The sentences are later processed to find relations
   */
  protected void extractDatumGraphForClassifer(DirectedGraph<KBPEntity, Pair<SentenceGroup, List<CoreMap>>> graph, Annotation document,
                                               KBPEntity root, int maxDepth, int entityBudget) {
    // Construct a queue
    Queue<Pair<KBPEntity,Integer>> workQueue = new LinkedList<Pair<KBPEntity,Integer>>();
    workQueue.offer( new Pair<KBPEntity,Integer>( root, maxDepth ) );

    int perDocumentEntityBudget = Props.TEST_GRAPH_DOCUMENT_BUDGET;

    while( workQueue.size() != 0 ) {
      // Dequeue an entity (with it's depth information)
      Pair<KBPEntity, Integer> item = workQueue.poll();
      KBPEntity entity = item.first(); int depth = item.second();
      forceTrack( String.format( "Augmenting entity: %s at depth %d", entity, depth )  );
      assert depth > 0;

      // IMPORTANT!: Create a copy of the document for the entity specific
      // annotations.
      // Things down the pipeline make use of these annotations.
      Annotation documentForEntity = CoreMapUtils.copyDocument(document);
      List<CoreMap> rawSentences = documentForEntity.get(CoreAnnotations.SentencesAnnotation.class);
      assert rawSentences != document.get(CoreAnnotations.SentencesAnnotation.class);

      // -- Annotation
      PostIRAnnotator annotator = new PostIRAnnotator(entity.name, Maybe.Just(entity.type.name), Maybe.<String>Nothing(), Maybe.<String>Nothing(), true);

      Redwood.startTrack("Annotating " + rawSentences.size() + " sentences...");
      // Find the relevant sentences
      List<CoreMap> supportingSentences = process.annotateSentenceFeatures(entity,
          CollectionUtils.filter( CollectionUtils.subList( rawSentences, annotator.findRelevantSentences(documentForEntity) ), new Function<CoreMap, Boolean>() {
            @Override
            public Boolean apply(CoreMap in) {
              if( in.get(CoreAnnotations.TokensAnnotation.class).size() > Props.TEST_SENTENCES_MAX_TOKENS ) {
                logger.debug("Ignoring suspiciously long sentence (> " + Props.TEST_SENTENCES_MAX_TOKENS + " tokens): " + CoreMapUtils.sentenceToMinimalString(in));
                return false;
              }
              return true;  //To change body of implemented methods use File | Settings | File Templates.
            }
          } ),
          AnnotateMode.ALL_PAIRS);

      logger.debug( "Found " + supportingSentences.size() + " relevant sentences." );
      Redwood.endTrack("Annotating " + rawSentences.size() + " sentences...");

      // -- Featurize
      Annotation supportingSubDocument = new Annotation(supportingSentences);
      for( Pair<SentenceGroup,List<CoreMap>> pair : process.featurizeWithSentences(supportingSubDocument, relationFilterForFeaturizer).values() ) {
        //  Use the sentence group and stuff to populate edges
        KBPair key = pair.first.key;
        KBPEntity headEntity = key.getEntity();
        if (!key.getSlotEntity().isDefined()) {
          logger.warn("could not create slot entity for " + key);
          continue;
        }
        KBPEntity tailEntity = key.getSlotEntity().get();

        // -- Add to workQueue
        // Try to make the slot value an entity.
        // Check if the entity is already in the graph
        if( depth > 1 &&
            (!graph.containsVertex( tailEntity )) &&
            tailEntity.type.isEntityType() ) {
            // TODO(arun): Do something intelligent in how you prioritize
            // what you add to the work queue; maybe have a priority queue
            // (i.e. sorted list) with "promise"
            if ( entityBudget > 0 ) {
              workQueue.add( new Pair<KBPEntity,Integer>(tailEntity, depth-1) );
              logger.debug( "Adding entity to work queue: " + tailEntity );
              entityBudget--;
            } else {
              logger.debug("Reached maximum size of graph and will not add entity to work queue: " + tailEntity );
            }
        }
        if(!graph.containsVertex(tailEntity)) {
          if( perDocumentEntityBudget > 0 ) {
            // By adding the vertex to the graph, we can be sure that we
            // will never double visit the entity
            graph.addVertex( tailEntity );
            perDocumentEntityBudget--;
          } else {
            logger.log( "perDocumentEntityBudget exhausted, not adding " + tailEntity );
          }
        }
        if( graph.containsVertex( tailEntity ) ) {
          // -- Add to graph
          if( graph.isEdge(headEntity,tailEntity) ) {
            Pair<SentenceGroup, List<CoreMap>> pair_ = graph.getEdges(headEntity, tailEntity).get(0);
            pair_.first.merge(pair.first);
            pair_.second.addAll(pair.second);
          } else {
            graph.add(headEntity, tailEntity, pair);
          }
        }
      }
      endTrack( String.format( "Augmenting entity: %s at depth %d", entity, depth )  );
      logger.debug( String.format( "Have %d more entities to process", workQueue.size() ) );
    }
  }


  /**
   * Extract datums from the classifier, but without knowing what the pivot entity is.
   * Use {@link InferentialSlotFiller#extractRelationGraph(edu.stanford.nlp.kbp.slotfilling.common.KBPOfficialEntity, int, int, edu.stanford.nlp.kbp.slotfilling.common.Maybe)}
   * if possible, as this (a) extracts fewer datums, but ones which are generally more relevant to the entity, (b) more
   * reliably disambiguates coreference, and (c) has an entity budget that can be more efficiently enforced.
   *
   * @param originalDocument The original document to annotate.
   */
  public EntityGraph  extractRelationGraph(Annotation originalDocument) {
    startTrack("Extracting datums");
    DirectedGraph<KBPEntity,Pair<SentenceGroup,List<CoreMap>>> datumGraph = new DirectedGraph<KBPEntity,Pair<SentenceGroup,List<CoreMap>>>();
    // Annotate the document
    startTrack("Annotating " + originalDocument.get(CoreAnnotations.DocIDAnnotation.class));
    Annotation doc = CoreMapUtils.copyDocument(originalDocument);  // copy to ensure that this function is functional
    PostIRAnnotator annotator = new PostIRAnnotator("<no entity given>", Maybe.<String>Nothing(), Maybe.<String>Nothing(), Maybe.<String>Nothing(), true);
    annotator.annotate(doc);
    assert doc.get(CoreAnnotations.SentencesAnnotation.class) != originalDocument.get(CoreAnnotations.SentencesAnnotation.class);
    endTrack("Annotating " + originalDocument.get(CoreAnnotations.DocIDAnnotation.class));

    // Collect candidate entities to treat as the "pivot" entity
    // Sadly, this has to come after PostIR, as we need to know every entity's antecedent.
    startTrack("Collecting candidate pivot entities");
    Set<KBPEntity> entityCandidates = new HashSet<KBPEntity>();
    for (CoreMap sentence : doc.get(CoreAnnotations.SentencesAnnotation.class)) {
      for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
        if ((token.ner().equals(NERTag.PERSON.name) || token.ner().equals(NERTag.ORGANIZATION.name)) &&
            token.get(CoreAnnotations.AntecedentAnnotation.class) != null) {
          // Add an entity that is a PER or ORG, and has a valid antecedent
          entityCandidates.add(KBPNew.entName(token.get(CoreAnnotations.AntecedentAnnotation.class))
              .entType(NERTag.fromString(token.ner()).orCrash()).KBPEntity());
        }
      }
    }
    for (KBPEntity pivot : entityCandidates) {
      logger.debug(pivot);
      datumGraph.addVertex(pivot);
    }
    logger.log("" + entityCandidates.size() + " pivot entities found.");
    endTrack("Collecting candidate pivot entities");

    // Collect datums for each entity
    startTrack("Populating graph");
    for (KBPEntity pivot : entityCandidates) {
      forceTrack("Pivoting on " + pivot.toString());
      List<CoreMap> supportingSentences = process.annotateSentenceFeatures(pivot, doc.get(CoreAnnotations.SentencesAnnotation.class), AnnotateMode.ALL_PAIRS);
      logger.debug( "found " + supportingSentences.size() + " relevant sentences." );
      Annotation supportingSubDocument = new Annotation(supportingSentences);
      for( Pair<SentenceGroup,List<CoreMap>> pair : process.featurizeWithSentences(supportingSubDocument, relationFilterForFeaturizer).values() ) {
        //  Use the sentence group and stuff to populate edges
        KBPair key = pair.first.key;
        KBPEntity headEntity = key.getEntity();
        if (!key.getSlotEntity().isDefined()) {
          logger.warn("could not create slot entity for " + key);
          continue;
        }
        KBPEntity tailEntity = key.getSlotEntity().get();

        if( datumGraph.isEdge(headEntity, tailEntity) ) {
          Pair<SentenceGroup, List<CoreMap>> pair_ = datumGraph.getEdges(headEntity, tailEntity).get(0);
          pair_.first.merge(pair.first);
          pair_.second.addAll(pair.second);
        } else {
          datumGraph.add(headEntity, tailEntity, pair);
        }
      }
      endTrack("Pivoting on " + pivot.toString());
    }
    endTrack("Populating graph");
    endTrack("Extracting datums");

    startTrack("Running classifier");
    EntityGraph graph = new EntityGraph();
    // Run classifiers
    int numEdges = 0;
    for( Pair<SentenceGroup,List<CoreMap>> pair : datumGraph.getAllEdges() ) {
      KBPEntity headEntity = pair.first.key.getEntity();
      SentenceGroup sg = pair.first;
      CoreMap[] sentences = pair.second.toArray(new CoreMap[pair.second.size()]);
      // Run relation extractor
      for( KBPSlotFill fill : findSlotsWithClassifier(sg, sentences) ) {
        assert(fill.key.entityName.equals(headEntity.name));
        graph.add( fill.key.getEntity(), fill.key.getSlotEntity().orCrash(), fill );
        numEdges += 1;
      }
      // Run rules
      for( KBPSlotFill fill : findSlotsWithRules(sg, sentences) ) {
        assert(fill.key.entityName.equals(headEntity.name));
        graph.add( fill.key.getEntity(), fill.key.getSlotEntity().orCrash(), fill );
        numEdges += 1;
      }
    }
    logger.log("" + numEdges + " edges in graph");
    endTrack("Running classifier");

    // Run ReVerb
    if (Props.TEST_GRAPH_REVERB_DO) {
      forceTrack("Augmenting with Reverb extractions");
      // Extract relations using ReVerb
      // TODO(gabor) removed in DEFT repository
      logger.log("" + numEdges + " edges in graph");
      endTrack("Augmenting with Reverb extractions");
    }

    // Return
    assert graph.isValidGraph();
    return graph;
  }

  /**
   * Get a list of slot fills (ie. edges) as output by the classifier
   * @param sg The sentence group corresponding to a potential edge.
   * @param sentences The actual sentences corresponding to this group, for the classifier.
   * @return A list of slot fills along this edge (effectively, the relations with provenance, as the key is defined in the sentence group)
   */
  List<KBPSlotFill> findSlotsWithClassifier(SentenceGroup sg, CoreMap[] sentences)  {
    // vvv RUN CLASSIFIER vvv
    Counter<Pair<String, Maybe<KBPRelationProvenance>>> relationsAsStrings =
        classifyComponent.classifyRelations(sg, Maybe.Just(sentences) );
    // ^^^                ^^^

    // Convert to Probabilities
    List<KBPSlotFill> slotFills = new ArrayList<KBPSlotFill>();
    for (Map.Entry<Pair<String, Maybe<KBPRelationProvenance>>, Double> entry : relationsAsStrings.entrySet()) {
      RelationType rel = RelationType.fromString(entry.getKey().first).orCrash();
      Maybe<KBPRelationProvenance> provenance = entry.getKey().second;
      double prob = entry.getValue();
      //if (Props.TEST_PROBABILITYPRIORS) {
      //  prob = new Probabilities(irComponent, allSlotFills, entry.getValue())
      //      .ofSlotValueGivenRelationAndEntity(sg.key.slotValue, rel, queryEntity);
      //}
      KBPair key = sg.key;
      if (!key.slotType.isDefined()) { logger.warn("slot type is not defined for KBPair: " + key); } // needed for some of the consistency checks
      slotFills.add(KBPNew.from(key).rel(rel).provenance(provenance).score(prob).KBPSlotFill());
    }

    return slotFills;
  }

  /**
   * Get a list of slot fills (ie. edges) as output by the various rule-based classifiers
   * @param sg The sentence group corresponding to a potential edge.
   * @param sentences The actual sentences corresponding to this group, for the rule based systems.
   * @return A list of slot fills along this edge (effectively, the relations with provenance, as the key is defined in the sentence group)
   */
  List<KBPSlotFill> findSlotsWithRules(SentenceGroup sg, CoreMap[] sentences)  {
    KBPair key = sg.key;
    KBPEntity head = key.getEntity();
    if (!key.slotType.isDefined()) { return new ArrayList<KBPSlotFill>(); }
    KBPEntity tail = KBPNew.entName(key.slotValue).entType(key.slotType.get()).KBPEntity();

    List<KBPSlotFill> slotFills = new ArrayList<KBPSlotFill>();
    // -- Rule based sentences
    startTrack("Rule based additions");
    List<CoreMap> sentencesInOfficialIndex = new ArrayList<CoreMap>();
    // Get sentences in official index
    for (CoreMap sent : sentences) {
      if (KBPRelationProvenance.isOfficialIndex(sent.get(KBPAnnotations.SourceIndexAnnotation.class))) { sentencesInOfficialIndex.add(sent); }
    }
    // Get heuristic fills
    for(Map.Entry<Pair<String, Maybe<KBPRelationProvenance>>, Double> fill :
        HeuristicRelationExtractor.allExtractors.apply( Pair.makePair(sg.key, sentences) ).entrySet() ) {
      logger.debug(head.name + " | " + fill.getKey().first + " | " + tail.name);
      slotFills.add( KBPNew.from(sg.key).rel(fill.getKey().first).provenance(fill.getKey().second).score(fill.getValue()).KBPSlotFill());
    }
    if (Props.TEST_RULES_ALTERNATENAMES_DO) {
      for (KBPSlotFill altName : AlternateNamesExtractor.extractSlots(head, sentencesInOfficialIndex)) {
        logger.log(altName.key.entityName + " | " + altName.key.relationName + " | " + altName.key.slotValue);
        slotFills.add(altName);
      }
    }
    endTrack("Rule based additions");
    logger.debug("" + slotFills.size() + " slots extracted");

    return slotFills;
  }

  protected void printRelations( KBPEntity entity, String slotValue, List<KBPSlotFill> fills) {
    final java.text.DecimalFormat df = new java.text.DecimalFormat("0.000");
    // Create a copy to sort it.
    fills = new ArrayList<KBPSlotFill>(fills);
    Collections.sort(fills);

    StringBuilder b = new StringBuilder();
    b.append(entity.name).append(" | ");
    for (int i = 0; i < Math.min(fills.size(), 3); ++i) {
      b.append(fills.get(i).key.relationName)
          .append(" [").append(df.format(fills.get(i).score.getOrElse(0.0))).append("] | ");
    }
    b.append(slotValue);
    logger.log(b);
  }

  protected void printRelations( KBPEntity head, List<KBPSlotFill> fills) {
    Map<String, List<KBPSlotFill>> groupedFills = CollectionUtils.groupBy( fills, new Function<KBPSlotFill, String>() {
          @Override
          public String apply(KBPSlotFill in) {
            return in.key.slotValue;
          }
        });
    for( Map.Entry<String,List<KBPSlotFill>> entry: groupedFills.entrySet() ){
      printRelations(head, entry.getKey(), entry.getValue());
    }
  }

  protected void printRelations( EntityGraph graph) {
    // Display predictions
    startTrack("Relation Predictions");
    for( Triple<KBPEntity, KBPEntity, List<KBPSlotFill>> triple : CollectionUtils.groupedEdges(graph) ) {
      if (triple.third.size() > 0) {
        printRelations(triple.first, triple.second.name, triple.third);
      }
    }
    logger.log("" + graph.getAllEdges().size() + " slots extracted");
    endTrack("Relation Predictions");
  }

  @SuppressWarnings("UnusedDeclaration")
  protected void printGraph( String name, EntityGraph relationGraph) {
    VisualizationUtils.logGraph(name,
        CollectionUtils.map(relationGraph, new Function<Triple<KBPEntity, KBPEntity, List<KBPSlotFill>>, Triple<String, String, List<String>>>() {
          @Override
          public Triple<String, String, List<String>> apply(Triple<KBPEntity, KBPEntity, List<KBPSlotFill>> in) {
            String e1 = in.first.name;
            String e2 = in.second.name;
            List<String> relations = new ArrayList<String>();
            for (KBPSlotFill fill : in.third) {
              relations.add(String.format("%s:%.2f", fill.key.relationName, fill.score.getOrElse(-1.0)));
            }
            return Triple.makeTriple(e1, e2, relations);
          }
        }));
  }
}

