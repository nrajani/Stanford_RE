package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.kbp.entitylinking.classify.namematcher.RuleBasedNameMatcher;
import edu.stanford.nlp.kbp.slotfilling.classify.HeuristicRelationExtractor;
import edu.stanford.nlp.kbp.slotfilling.classify.RelationClassifier;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.kbp.slotfilling.ir.PostIRAnnotator;
import edu.stanford.nlp.kbp.slotfilling.process.KBPProcess;
import edu.stanford.nlp.kbp.slotfilling.process.RelationFilter;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.*;

import java.io.*;
import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * <p>An interface for the relevant methods for filtering valid slot fills.
 * This is intended for the KBP Slotfilling Validation exercise, NOT as an internal validation
 * method. In particular:
 *
 * <ul>
 *   <li>It reads in slot outputs from raw offsets. This is inherently somewhat unstable</li>
 *   <li>It does not perform any inference or competition; any gains from these are lost</li>
 *   <li>It runs the nonlocal slot validators in a fundamentally different mode of operation,
 *       intended for cases where, e.g., there are duplicate slot fills.</li>
 * </ul>
 *
 * @author Gabor Angeli
 */
public class KBPSlotValidator extends KBPEvaluator {

  /**
   * An object to represent a slot fill read in from a response file.
   * It contains the raw fill, along with metadata (e.g., the query id; our classifier predictions; etc.)
   */
  public static class KBPSlotFillQuery {
    public final KBPSlotFill fill;
    public final String validationQueryID;
    public final Maybe<Double> classifierScore;
    public final Maybe<List<RelationType>> classifierPrediction;

    private KBPSlotFillQuery(KBPSlotFill fill, String validationQueryID, Maybe<Double> classifierScore, Maybe<List<RelationType>> classifierPrediction) {
      this.fill = fill;
      this.validationQueryID = validationQueryID;
      this.classifierScore = classifierScore;
      this.classifierPrediction = classifierPrediction;
    }
    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof KBPSlotFillQuery)) return false;
      KBPSlotFillQuery that = (KBPSlotFillQuery) o;
      return fill.equals(that.fill) && validationQueryID.equals(that.validationQueryID);
    }
    @Override
    public int hashCode() {
      int result = fill.hashCode();
      result = 31 * result + validationQueryID.hashCode();
      return result;
    }
  }

  /** The gold responses for this year -- or, as many of them as are available */
  private final GoldResponseSet goldResponses;

  /**
   * Create a new SlotFill Validator object
   *
   * @see KBPEvaluator#KBPEvaluator(Properties, KBPIR, KBPProcess, RelationClassifier)
   */
  public KBPSlotValidator(Properties props, KBPIR ir, KBPProcess process, RelationClassifier classify) {
    super(props, ir, process, classify);
    goldResponses = new GoldResponseSet();
  }

  /**
   * Filter the given candidate slot fills.
   * @param candidates The candidate slot fills. These may contain duplicates, as well as slot fills which do not come
   *                   from our KBP system.
   * @return A set of valid slot fills. These must be strictly equal (==) to slot fills in the candidate set; no rewrites
   *         are allowed.
   */
  public IdentityHashSet<KBPSlotFill> filterSlotFills(Iterable<KBPSlotFillQuery> candidates) {
    startTrack("Validating Slot Fills");
    IdentityHashSet<KBPSlotFill> allQueryFills = new IdentityHashSet<KBPSlotFill>();
    Counter<String> preFiltered = new ClassicCounter<String>();
    // Create a mapping from entity to the associated slot fills
    Map<KBPEntity, List<KBPSlotFill>> fillsForEntity = new HashMap<KBPEntity, List<KBPSlotFill>>();
    for (KBPSlotFillQuery query : candidates) {
      KBPOfficialEntity entity = KBPNew.from(query.fill.key).KBPOfficialEntity();
      if (!fillsForEntity.containsKey(entity)) { fillsForEntity.put(entity, new ArrayList<KBPSlotFill>()); }

      // Register this as a query slot fill
      allQueryFills.add(query.fill);

      // Filter based on classifier output
      // filter: no datum found at all
      if (Props.VALIDATE_FORCECLASSIFIABLE && !query.classifierScore.isDefined()) {
        if (!isInOurSlotfillerOutput(entity, query.fill.key.kbpRelation(), query.fill.key.slotValue)) {
          log("could not find datum for sentence");
          preFiltered.incrementCount("no datum");
          continue;
        }
      }
      // filter: NER type is wrong
      if (Props.VALIDATE_FORCETYPE && !query.fill.key.slotType.isDefined()) {
        if (!isInOurSlotfillerOutput(entity, query.fill.key.kbpRelation(), query.fill.key.slotValue)) {
          log("could not find NER type of slot value");
          preFiltered.incrementCount("no NER");
          continue;
        }
      }
      // filter: classifier predicted no relation
      if (Props.VALIDATE_FILTERNORELATION && !query.classifierPrediction.isDefined()) {
        // case: our classifier didn't predict anything. Try to pass it through rules
        if (!query.fill.provenance.isDefined() || !query.fill.provenance.get().containingSentenceLossy.isDefined()) {
          log("invalid provenance");
          continue;
        }
        CoreMap[] sentences = new CoreMap[]{ query.fill.provenance.orCrash().containingSentenceLossy.orCrash()};
        List<RelationType> rulePredictedRealtions = new ArrayList<RelationType>();
        for (Map.Entry<Pair<String, Maybe<KBPRelationProvenance>>, Double> fill :
            HeuristicRelationExtractor.allExtractors.apply(Pair.makePair((KBPair) query.fill.key, sentences)).entrySet()) {
          rulePredictedRealtions.add(RelationType.fromString(fill.getKey().first).orCrash());
        }
        for (KBPSlotFill altName : AlternateNamesExtractor.extractSlots(query.fill.key.getEntity(), Arrays.asList(sentences))) {
          rulePredictedRealtions.add( altName.key.kbpRelation() );
        }
        if (!containsEquivalentRelation(rulePredictedRealtions, query.fill.key.kbpRelation())) {
          if (!isInOurSlotfillerOutput(entity, query.fill.key.kbpRelation(), query.fill.key.slotValue)) {
            log("classifier predicted no_relation (" + query.fill.key.kbpRelation() + "; rules predicted [" + StringUtils.join(rulePredictedRealtions, ", ") + "])");
            preFiltered.incrementCount("no relation");
            continue;
          }
        }
      }
      // filter: classifier disagrees with prediction
      if (query.classifierPrediction.isDefined() && !containsEquivalentRelation(query.classifierPrediction.get(), query.fill.key.kbpRelation())) {
        // case: the classifer predicted a different relation than the slot fill
        // See if the output relation is a valid rewrite of the original relation
        boolean rewriteState = Props.TEST_CONSISTENCY_REWRITE;
        Props.TEST_CONSISTENCY_REWRITE = true;
        List<KBPSlotFill> input = new ArrayList<KBPSlotFill>();
        List<KBPSlotFill> filtered = new ArrayList<KBPSlotFill>();
        for (RelationType classifiedRelation : query.classifierPrediction.get()) {
          KBPSlotFill modifiedFill = KBPNew.from(query.fill).rel(classifiedRelation).KBPSlotFill();
          input.add(modifiedFill);
          // rewrite geographic regions appropriately
          for (KBPSlotFill rewritten : WorldKnowledgePostProcessor.singleton(Props.TEST_CONSISTENCY_WORLDKNOWLEDGE_DIR).configure(true, false, false, false).isValidSlotAndRewrite(entity, modifiedFill)) {
            filtered.add(rewritten);
          }
        }
        // rewrite other terms
        filtered.addAll(SlotfillPostProcessor.unaryRewrites.postProcess(entity, input));
        Props.TEST_CONSISTENCY_REWRITE = rewriteState;
        if (filtered.size() == 0 || !filtered.get(0).key.kbpRelation().sameEquivalenceClass(query.fill.key.kbpRelation())) {
          // Nope. the rewritten relation still differs from the prediction
          if (!isInOurSlotfillerOutput(entity, query.fill.key.kbpRelation(), query.fill.key.slotValue)) {
            log("classifier disagreed with the prediction (" + query.fill.key.kbpRelation() + " but predicted [" + StringUtils.join(query.classifierPrediction.get(), ", ") + "])");
            preFiltered.incrementCount("different relation");
            continue;
          }
        }
      }

      // Add the slot fill for post-processing
      fillsForEntity.get(entity).add(query.fill);
    }

    log(BLUE, "Pre-filtered " + ((int) preFiltered.totalCount()) + " / " + allQueryFills.size() + " slot fills");

    IdentityHashSet<KBPSlotFill> validFills = new IdentityHashSet<KBPSlotFill>();
    GoldResponseSet checklist = new GoldResponseSet();
    for (Map.Entry<KBPEntity, List<KBPSlotFill>> entry : fillsForEntity.entrySet()) {
      for (KBPSlotFill validFill : SlotfillPostProcessor.validators.postProcess(entry.getKey(), entry.getValue(), checklist)) {
        if (allQueryFills.contains(validFill)) {
          validFills.add(validFill);
        } else {
          warn("The validators rewrote a slot fill -- this is assumed to be incorrect then");
        }
      }
    }

    // Show stats on slots filtered
    startTrack("Pre-Filter Stats");
    for (Map.Entry<String, Double> entry : preFiltered.entrySet()) {
      log(entry.getKey() + ": " + entry.getValue().intValue());
    }
    log(BLUE, "TOTAL " + ((int) preFiltered.totalCount()));
    endTrack("Pre-Filter Stats");
    log(BLUE, "    TOTAL TRUE: " + (validFills.size()) + " / " + allQueryFills.size() + " slot fills");
    log(BLUE, "TOTAL FILTERED: " + (allQueryFills.size() - validFills.size()) + " / " + allQueryFills.size() + " slot fills");

    // Get P/R
    // Collect stats
    int correct = 0;
    int gold    = 0;
    Set<KBTriple> validTriples = new HashSet<KBTriple>();
    for (KBPSlotFill fill : validFills) { validTriples.add(fill.key); }
    int guessed = validTriples.size();
    for (KBPSlotFill key : goldResponses.correctFills()) {
      if (validTriples.contains(key.key)) { correct += 1; }
      gold += 1;
    }
    // Compute
    double p = correct / guessed;
    double r = correct / gold;
    double f1 = 2.0 * p * r / (p + r);
    log(GREEN, "     PRECISION: " + p);
    log(GREEN, "        RECALL: " + r);
    log(GREEN, "            F1: " + f1);

    endTrack("Validating Slot Fills");
    return validFills;
  }

  /** A cache for our system's output */
  private Map<String, List<KBPSlotFill>> systemOutput = new HashMap<String, List<KBPSlotFill>>();
  /** Check if this slot fill was output by our system, or if something sufficiently similar was */
  private synchronized boolean isInOurSlotfillerOutput(KBPOfficialEntity entity, RelationType relation, String slotValue) {
    // Make sure we should run the slotfiller
    if (!Props.VALIDATE_RUNSLOTFILLER) { return false; }
    // Make sure slotfiller has been run for entity
    if (!systemOutput.containsKey(entity.id.orCrash())) {
      systemOutput.put(entity.id.orCrash(), super.slotFiller.fillSlots(entity));
    }
    // Check if query in output
    for (KBPSlotFill fill : systemOutput.get(entity.id.orCrash())) {
      if (fill.key.kbpRelation().sameEquivalenceClass(relation) && approximateEntityMatchScore(fill.key.slotValue, slotValue)) {
        return true;
      }
    }
    // Return false
    return false;
  }

  /** Check if the collection of relation types contains a relation equivalent to the target relation */
  private static boolean containsEquivalentRelation(Collection<RelationType> collection, RelationType elem) {
    for (RelationType rel : collection) { if (rel.sameEquivalenceClass(elem)) { return true; } }
    return false;
  }

  /**
   * Approximately check if two entities are equivalent.
   * Internally, calls the entity linker on two dummy objects without context.
   * TODO(gabor) really, we can do a better job of passing more information through here.
   */
  private static boolean approximateEntityMatchScore( String higherGloss, String lowerGloss) {
    return Props.KBP_ENTITYLINKER.sameEntity(
        new EntityContext(KBPNew.entName(higherGloss).entType(NERTag.PERSON).KBPEntity()),
        new EntityContext(KBPNew.entName(lowerGloss).entType(NERTag.PERSON).KBPEntity())
    );
  }

  /**
   * Get the collection of system responses to validate.
   * @return A list of all system candidate slot fills, including duplicates, for each query entity.
   */
  @SuppressWarnings("UnusedDeclaration")
  public List<KBPSlotFillQuery> allSystemResponses() {
    startTrack("Fetching System Responses");
    // Get query files
    assert Props.VALIDATION_QUERIES.isDefined();
    File[] inputFiles;
    if (Props.VALIDATION_QUERIES.orCrash().isFile() && Props.VALIDATION_QUERIES.orCrash().exists()) {
      inputFiles = new File[] { Props.VALIDATION_QUERIES.orCrash() };
    } else if (Props.VALIDATION_QUERIES.orCrash().isDirectory()) {
      inputFiles = Props.VALIDATION_QUERIES.orCrash().listFiles();
      if (inputFiles == null) { throw new IllegalStateException("Could not list files in validation query directory!"); }
    } else {
      throw new IllegalStateException("Cannot read validation queries from: " + Props.VALIDATION_QUERIES.orCrash());
    }

    // Get entity mapping
    List<KBPOfficialEntity> testEntities = testEntities();
    Map<String, KBPOfficialEntity> queryId2entity = new HashMap<String, KBPOfficialEntity>();
    for (KBPOfficialEntity entity : testEntities) {
      queryId2entity.put(entity.queryId.orCrash(), entity);
    }

    // Result
    List<KBPSlotFillQuery> queries = new ArrayList<KBPSlotFillQuery>();

    try {
      // Read query files
      for (File query : inputFiles) {
        log("reading file " + query.getPath());
        BufferedReader reader = new BufferedReader(new FileReader(query));
        String line;
        while( (line = reader.readLine()) != null) {
          // Read fields, as per their names in the README
          String[] fields = line.split("\t");
          String SF_query_id = fields[0].trim();
          String slot_name = fields[1].trim();
          String SFV_query_id = fields[2].trim();
          String docid = fields[3].trim();
          if (docid.equalsIgnoreCase("NIL")) { continue; }
          String slot_filler = fields[4].trim();
          String filler_offsets = fields[5].trim();
          String entity_offsets = fields[6].trim();
          String pred_offsets = fields[7].trim();
          String conf_score = fields[8].trim();

          // Parse fields
          KBPOfficialEntity entity = queryId2entity.get(SF_query_id);
          assert entity != null;
          RelationType rel = RelationType.fromString(slot_name).orCrash();
          Span[] fillerSpan = new Span[filler_offsets.split(",").length];
          fillerSpan[0] = Span.fromValues((Object[]) filler_offsets.split(",")[0].split("-"));
          if (filler_offsets.split(",").length > 1) {
            fillerSpan[1] = Span.fromValues((Object[]) filler_offsets.split(",")[1].split("-"));
          }
          Span[] entitySpan = new Span[entity_offsets.split(",").length];
          try {
            entitySpan[0] = Span.fromValues((Object[]) entity_offsets.split(",")[0].split("-"));
            if (entity_offsets.split(",").length > 1) {
              entitySpan[1] = Span.fromValues((Object[]) entity_offsets.split(",")[1].split("-"));
            }
          } catch (NumberFormatException e) {
            entitySpan = fillerSpan;
          }

          // Add slot fill
          queries.add(constructQuery(SFV_query_id, entity, rel, docid, fillerSpan, entitySpan, slot_filler, Double.parseDouble(conf_score)));
        }
      }

      // Return
      return queries;
    } catch (IOException e) {
      throw new RuntimeException(e);
    } finally {
      endTrack("Fetching System Responses");

    }
  }

  /**
   * A helper function to construct an internal representation of a query line.
   *
   * God help ye who tries to make sense of this. Turns out, other systems output all sorts of
   * hippie nonsense.
   *
   * @param entity The entity, as recovered from the query file (not parsed explicitly).
   * @param relation The relation in the response, as a {@link RelationType}.
   * @param docid The document id of the provenance.
   * @param fillerSpans The span offsets (character) of the slot value in the document.
   * @param entitySpans The span offsets (character) of the entity in the document.
   * @return A representation of this slot fill query -- effectively, a SlotFill with auxilliary information
   */
  private KBPSlotFillQuery constructQuery(String validationQueryID, KBPOfficialEntity entity, RelationType relation, String docid,
                                          Span[] fillerSpans, Span[] entitySpans, String slotValue,
                                          double confidence) {
    // Fetch document
    Annotation doc = null;
    try {
      doc = irComponent.fetchDocument(docid.trim());
      new PostIRAnnotator(entity.name, Maybe.Just(entity.type.name), Maybe.<String>Nothing(), Maybe.<String>Nothing(), true)
          .annotate(doc);
    } catch (IllegalArgumentException e) {
      warn("could not find document: " + docid);
    }

    // Variables to fill
    int sentenceIndex = -1;
    KBPRelationProvenance provenance = null;
    Maybe<Double> classifierScore = Maybe.Nothing();
    Maybe<List<RelationType>> classiferPrediction = Maybe.Nothing();
    Maybe<NERTag> slotType = Maybe.Nothing();

    // Get sentence
    List<CoreMap> sentences = doc == null ? Collections.<CoreMap>emptyList() : doc.get(CoreAnnotations.SentencesAnnotation.class);
    Span entityTokenSpan;
    Span slotValueTokenSpan;
    for (int i = 0; i < sentences.size(); ++i) {
      List<CoreLabel> tokens = sentences.get(i).get(CoreAnnotations.TokensAnnotation.class);
      Span sentenceSpan = new Span(tokens.get(0).beginPosition(), tokens.get(tokens.size() - 1).endPosition());
      for (Span fillerSpan : fillerSpans) {
        for (Span entitySpan : entitySpans) {
          if (sentenceIndex >= 0) { continue; }
          if (sentenceSpan.contains(fillerSpan) && sentenceSpan.contains(entitySpan)) {

            // Found a sentence with the entity and slot fill
            sentenceIndex = i;
            entityTokenSpan = findBestSpan(tokens, entitySpan);
            slotValueTokenSpan = findBestSpan(tokens, fillerSpan);

            // Get slot type
            Counter<NERTag> slotTypes = new ClassicCounter<NERTag>();
            for (int k : slotValueTokenSpan) {
              if (!tokens.get(k).ner().equals(Props.NER_BLANK_STRING)) {
                for (NERTag tag : NERTag.fromString(tokens.get(k).ner())) {
                  slotTypes.incrementCount(tag);
                }
              }
            }
            if (slotTypes.size() > 0) {
              slotType = Maybe.Just(Counters.argmax(slotTypes));
            }

            // Construct provenance
            provenance = new KBPRelationProvenance(docid, Props.INDEX_OFFICIAL.getPath(),
                i, entityTokenSpan, slotValueTokenSpan, sentences.get(i));

            // Construct datum
            List<CoreMap> containingSentences = Arrays.asList(sentences.get(i));
            List<CoreMap> validSentences = processComponent.annotateSentenceFeatures(entity, containingSentences);
            if (validSentences.size() > 0) {
              for (SentenceGroup datum : processComponent.featurizeSentence(validSentences.get(0), Maybe.<RelationFilter>Nothing())) {
                String queryString = StringUtils.join(tokens.subList(slotValueTokenSpan.start(), slotValueTokenSpan.end()), " ").trim();
                if (datum.key.entityName.equals(entity.name) &&  // entities match
                    (approximateEntityMatchScore(datum.key.slotValue, queryString) ||  // strings are close enough
                        datum.key.slotType.equalsOrElse(Utils.inferFillType(relation).orCrash(), false) ||  // strings are close and have same type
                        (datum.key.slotValue.contains(queryString) || queryString.contains(datum.key.slotValue)) ||
                        RuleBasedNameMatcher.isAcronym(entity.name, tokens.subList(slotValueTokenSpan.start(), slotValueTokenSpan.end())) ||
                        RuleBasedNameMatcher.isAcronym(queryString, entity.name.split("\\s+"))
                    )) {
                  // Classify datum
                  classifierScore = Maybe.Just(classifier.classifyRelation(datum, relation, Maybe.<CoreMap[]>Nothing()).first);
                  Counter<Pair<String, Maybe<KBPRelationProvenance>>> predictions = classifier.classifyRelations(datum, Maybe.<CoreMap[]>Nothing());
                  if (predictions.size() > 0) {
                    List<RelationType> relations = new ArrayList<RelationType>();
                    for (Pair<String, Maybe<KBPRelationProvenance>> prediction : predictions.keySet()) {
                      relations.add(RelationType.fromString(prediction.first).orCrash());
                    }
                    classiferPrediction = Maybe.Just(relations);
                  }

                }
              }
            }

            // Break (found our answer)
            break;
          }
        }
      }
    }

    // Create slot fill
    KBPSlotFill candidate =
        KBPNew.from(entity)
            .slotValue(slotValue).slotType(slotType)
            .rel(relation)
            .provenance(Maybe.fromNull(provenance)).score(confidence).KBPSlotFill();

    // Return
    if (sentenceIndex < 0) {
      warn("could not find sentence for spans: " + entitySpans[0] + " and " + fillerSpans[0] + " (multi-sentence span?)");
    } else if (provenance == null) {
      warn("could not find provenance for query id " + validationQueryID);
    } else if (!classifierScore.isDefined()) {
      warn("could not find datum for validation query id " + validationQueryID);
    }
    return new KBPSlotFillQuery(
        candidate,
        validationQueryID,
        classifierScore,
        classiferPrediction);

  }

  /**
   * Find the best matching token span given a sentence and character offsets.
   * @param tokens The sentence to find the span in.
   * @param charSpan The character offset span to absorb.
   * @return A token offset span within the sentence corresponding to the closest words
   *         to the character offset boundaries.
   */
  private Span findBestSpan(List<CoreLabel> tokens, Span charSpan) {
    // Find best start
    int start = -1;
    int min = Integer.MAX_VALUE;
    for (int i = 0; i < tokens.size(); ++i) {
      int dist = Math.abs(tokens.get(i).beginPosition() - charSpan.start());
      if (dist < min) { min = dist; start = i; }
      if (dist == 0) { break; }
    }
    if (min != 0) { debug("could not find exact word match for token start (dist=" + min + ")"); }
    // Find best end
    int end = -1;
    min = Integer.MAX_VALUE;
    for (int i = start; i < tokens.size(); ++i) {
      int dist = Math.abs(tokens.get(i).endPosition() - charSpan.end());
      if (dist < min) { min = dist; end = i + 1; }
      if (dist == 0) { break; }
    }
    if (min != 0) { debug("could not find exact word match for token end (dist=" + min + ")"); }
    // Return
    return new Span(start, end);
  }

  /**
   * Run the slot fill validator.
   */
  @Override
  public Maybe<KBPScore> run() {
    try {
      startTrack("Running Validator");
      // Run evaluation (and dump file)
      String outputFileName = Props.WORK_DIR.getPath() + File.separator + Props.KBP_RUNID + ".validation.output";
      logger.log(FORCE, BLUE, BOLD, "writing query file to " + outputFileName);
      PrintStream os = new PrintStream(outputFileName, "UTF-8");
      List<KBPSlotFillQuery> input = allSystemResponses();
      officialOutputWriter.outputValidSlotsForEntity(os, input, filterSlotFills(input));
      endTrack("Running Validator");
      os.close();
      return Maybe.Nothing();
    } catch (FileNotFoundException e) {
      throw new RuntimeException(e);
    } catch (UnsupportedEncodingException e) {
      throw new RuntimeException(e);
    }
  }
}
