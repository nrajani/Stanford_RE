package edu.stanford.nlp.kbp.slotfilling.train;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

import java.io.File;
import java.io.IOException;
import java.sql.Connection;
import java.sql.SQLException;
import java.text.DecimalFormat;
import java.util.*;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.slotfilling.classify.KBPDataset;
import edu.stanford.nlp.kbp.slotfilling.classify.RelationClassifier;
import edu.stanford.nlp.kbp.slotfilling.classify.TrainingStatistics;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.common.CollectionUtils;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.process.FeatureFactory;
import edu.stanford.nlp.kbp.slotfilling.process.KBPProcess;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.util.logging.Redwood;

/**
 * Finds supporting relational data and trains a classifiers.
 */
public class KBPTrainer {
  public static enum UnlabeledSelectMode { NEGATIVE, NOT_POSITIVE, NOT_LABELED, NOT_POSITIVE_WITH_NEGPOS, NOT_LABELED_WITH_NEGPOS }
  public static enum MinimizerType { QN, SGD, SGDTOQN }

  protected static final Redwood.RedwoodChannels logger = Redwood.channels("Train");

  protected KBPIR querier;
  protected KBPProcess process;
  protected RelationClassifier classifier;

  @SuppressWarnings({"FieldCanBeLocal", "UnusedDeclaration"})
  private final Properties props;
  
  class EmptyIterator implements Iterator
  {
	  public Object next()
	  {
		  return null;
	  }
	  public boolean hasNext()
	  {
		  return false;
	  }
	  public void remove()
	  {
		  throw new UnsupportedOperationException();
	  }
  }

  public KBPTrainer(Properties props, KBPIR querier, KBPProcess process, RelationClassifier classifier) {
    this.querier = querier;
    this.process = process;
    this.classifier = classifier;
    this.props = props;
  }

  /**
   * Train the classifier on the data provided
   * @param dataset - List of training data
   * @return - classifier
   */
  public Pair<RelationClassifier, TrainingStatistics> trainOnData( KBPDataset<String, String> dataset ) {
    TrainingStatistics statistics = classifier.train(dataset);
    return Pair.makePair(classifier, statistics);
  }

  /**
   * Find relevant sentences, construct relation datums and train the classifier
   * @param tuples - List of true (entity, rel, entity) triples.
   * @return - classifier
   */
  public Pair<RelationClassifier, TrainingStatistics> trainOnTuples( List<KBTriple> tuples ) {
    return trainOnData(makeDataset(findDatums(tuples)));
  }

  /**
   * Creates a (lazy) iterator of datums given a collection of query tuples.
   * If a datum is found in the cache, it is returned. Otherwise, the datum is created lazily.
   * NOTE: When fetching from cached, the datums can include those not in the requested tuples
   *       and the ordering is different from the input tuples
   * @param tuples The tuples to query
   * @return A lazy iterator of {@link SentenceGroup}s corresponding to the datums for that query.
   *         Note that this includes both positive and negative datums.
   */
  public Iterator<SentenceGroup> findDatums(final Collection<KBTriple> tuples) {
    // Shortcut if we're reading only cached data
    // Don't get flustered by the code. In a Real Language™ this would simply be:
    //   rtn = null; withKeyDatumTable(DB_TABLE_DATUM_CACHE, (psql) => rtn = values(psql).flatMap( _.values() )); return rtn;
    // Namely, with a postgres connection, get all values (with some nasty flat-mapping going on since the cache stores maps).
    // This has the advantage of doing a linear scan, rather than n * O( log(n) ) random disk accesses
    if (Props.CACHE_DATUMS_IGNOREUNCACHED) {
      final Pointer<Iterator<SentenceGroup>> allDatums = new Pointer<Iterator<SentenceGroup>>();
      final Set<String> datumsToReturn = new HashSet<String>();
      for (KBTriple triple : tuples) { datumsToReturn.add(PostgresUtils.KeyValueCallback.keyToString(triple)); }
      PostgresUtils.withKeyDatumTable(Props.DB_TABLE_DATUM_CACHE, new PostgresUtils.KeyDatumCallback() {
        @Override
        public void apply(Connection psql) throws SQLException {
          allDatums.set(CollectionUtils.flatMapIgnoreNull(this.entries(psql, Props.DB_TABLE_DATUM_CACHE), new Function<Map.Entry<String, Map<KBPair, SentenceGroup>>, Iterator<SentenceGroup>>() {
            @Override
            public Iterator<SentenceGroup> apply(Map.Entry<String, Map<KBPair, SentenceGroup>> in) {
              if (datumsToReturn.contains(in.getKey())) {
                // sort so we have a consistent iteration order
                ArrayList<SentenceGroup> values = new ArrayList<SentenceGroup>(in.getValue().values());
                Collections.sort(values);
                return values.iterator();
              } else {
                // discard this example
                return new EmptyIterator();
              }
            }
          }));
        }
      });
      if (allDatums.dereference().isDefined()) { return allDatums.dereference().get(); }
    }

    // Else, start caching!
    return CollectionUtils.iteratorFromMaybeIterableFactory(new Factory<Maybe<Iterable<SentenceGroup>>>() {
      Iterator<KBTriple> iter = tuples.iterator();

      @Override
      public Maybe<Iterable<SentenceGroup>> create() {
        if (iter.hasNext()) {
          final KBTriple key = iter.next();
          final Pointer<Map<KBPair, SentenceGroup>> datums = new Pointer<Map<KBPair, SentenceGroup>>();

          // Try Cache
          if (Props.CACHE_DATUMS_DO) {
            PostgresUtils.withKeyDatumTable(Props.DB_TABLE_DATUM_CACHE, new PostgresUtils.KeyDatumCallback() {
              @Override
              public void apply(Connection psql) throws SQLException {
                Maybe<Map<KBPair, SentenceGroup>> cachedValue = get(psql, Props.DB_TABLE_DATUM_CACHE, keyToString(key));
                if (cachedValue.isDefined()) {
                  datums.set(cachedValue.get());
                }
              }
            });
          }

          // Run Featurizer
          if (!datums.dereference().isDefined()) {
            startTrack(key.toString());
            KBPEntity entity1 = key.getEntity();
            String entity2 = key.slotValue;
            // vv (1) Query Sentence In Lucene vv
            // Query just for entity1 and entity2 without reln (
            // so we don't bias the training data with what we think is indicative of the relation)
            List<CoreMap> sentences = querier.querySentences(entity1.name, entity2, Props.TRAIN_SENTENCES_PER_ENTITY);
            // ^^
            logger.logf("Found %d sentences for %s", sentences.size(), key);
            // vv (2) Annotate Sentence vv
            sentences = process.annotateSentenceFeatures(entity1, sentences);
            // ^^
            logger.logf("Keeping %d sentences after annotation", sentences.size());

            HashMap<KBPair, SentenceGroup> featurized;
            if (sentences.size() > 0) {
              try {
                // Get datums from sentences.
                Annotation annotation = new Annotation("");
                annotation.set(CoreAnnotations.SentencesAnnotation.class, sentences);
                // vv (3) Featurize Sentence vv
                featurized = process.featurize(annotation);
                // ^^
                datums.set(featurized);
              } catch (RuntimeException e) {
                logger.warn(e);
              }
            }
            endTrack(key.toString());
          }

          // Cache
          if (Props.CACHE_DATUMS_DO) {
            PostgresUtils.withKeyDatumTable(Props.DB_TABLE_DATUM_CACHE, new PostgresUtils.KeyDatumCallback() {
              @Override
              public void apply(Connection psql) throws SQLException {
                if (datums.dereference().isDefined()) {
                  put(psql, Props.DB_TABLE_DATUM_CACHE, keyToString(key), datums.dereference().get());
                } else {
                  put(psql, Props.DB_TABLE_DATUM_CACHE, keyToString(key), new HashMap<KBPair, SentenceGroup>());
                }
              }
            });
          }

          // Return
          if (datums.dereference().isDefined()) {
            ArrayList<SentenceGroup> values = new ArrayList<SentenceGroup>(datums.dereference().get().values());
            Collections.sort(values);
            return Maybe.Just((Iterable<SentenceGroup>) values);
          } else {
            return Maybe.Nothing();
          }
        } else {
          return null;
        }
      }
    });
  }

  /**
   * @see KBPTrainer#makeDataset(java.util.Iterator, edu.stanford.nlp.kbp.slotfilling.common.Maybe)
   * @param datums list of sentence examples.
   * @return A KBP Dataset
   */
  @SuppressWarnings("UnusedDeclaration")
  public KBPDataset<String, String> makeDataset(Map<KBPair,SentenceGroup> datums) {
    return makeDataset(datums.values().iterator(), Maybe.<Map<KBPair, Set<String>>>Nothing());
  }

  /**
   * @see KBPTrainer#makeDataset(java.util.Iterator, edu.stanford.nlp.kbp.slotfilling.common.Maybe)
   * @param datums list of sentence examples.
   * @param positiveRelations The known positive labels
   * @return A KBP Dataset
   */
  public KBPDataset<String, String> makeDataset(Map<KBPair,SentenceGroup> datums,
                                                Map<KBPair, Set<String>> positiveRelations) {
    return makeDataset(datums.values().iterator(), Maybe.Just(positiveRelations));
  }

  /**
   * @see KBPTrainer#makeDataset(java.util.Iterator, edu.stanford.nlp.kbp.slotfilling.common.Maybe)
   * @param datums list of sentence examples.
   * @return A KBP Dataset
   */
  public KBPDataset<String, String> makeDataset(Iterator<SentenceGroup> datums) {
    return makeDataset(datums, Maybe.<Map<KBPair, Set<String>>>Nothing());
  }

  /**
   * datums contain a list of sentence examples that might possibly express the KBTriple key.
   * This function aggregates all the possible labels between an entity pair and clusters together
   * the sentences. It then samples _some negative "relations"/labels_ and produces a dataset that
   * contains the entity pair, lists of positive and negative labels and the supporting sentences.
   * @param datums list of sentence examples.
   * @param positiveRelations The known positive labels, if we want to override the default.
   * @return A KBP Dataset
   */
  public KBPDataset<String, String> makeDataset(Iterator<SentenceGroup> datums,
                                                Maybe<Map<KBPair, Set<String>>> positiveRelations) {
    startTrack("Making dataset");
    logger.log("Train unlabeled = " + Props.TRAIN_UNLABELED + " with " + Props.TRAIN_UNLABELED_SELECT);
    // Go through the datums and collect all the duplicated instances across types
    KBPDataset<String, String> dataset = new KBPDataset<String,String>();
    Random rand = new Random(0);
    int numDatumsWithPositiveLabels = 0;
    int numDatumsWithMultipleLabels = 0;
    int totalSentenceGroups = 0;
    Set<String> allLabels = new HashSet<String>();
    for (RelationType rel:RelationType.values()) {
      allLabels.add(rel.canonicalName);
    }
    for( SentenceGroup group : new IterableIterator<SentenceGroup>(datums)) {
      totalSentenceGroups++;
      // Get all positive labels
      KBPair key = group.key;

      Set<String> positiveLabels = positiveRelations.isDefined() ? positiveRelations.get().get(key) : this.querier.getKnownRelationsForPair(key);
      if (positiveLabels == null) { positiveLabels = new HashSet<String>(); }
      if (positiveLabels.size() > 0) { numDatumsWithPositiveLabels += 1; }
      if (positiveLabels.size() > 1) { numDatumsWithMultipleLabels += 1; }
      for (String posLabel : positiveLabels) { assert RelationType.fromString(posLabel).orCrash().canonicalName.equals(posLabel); }

      // Subsample negatives with some probability. This means you will set all relations to be false.
      boolean addExample = (positiveLabels.size() > 0 || rand.nextDouble() <= Props.TRAIN_NEGATIVES_SUBSAMPLERATIO);

      Set<String> negativeLabels = new HashSet<String>();
      if (addExample) {
        // Get negative labels
        KBPEntity entity = key.getEntity();
        for (RelationType rel : RelationType.values()) {
          // Never add a negative that we know is a positive
          if (positiveLabels.contains(rel.canonicalName)) { continue; }
          if (Props.TRAIN_NEGATIVES_INCOMPLETE) {
            // Only add the negative label if we know of a positive slot fill
            // for the relation, and it's not this candidate slot fill.
            // For example, Barack Obama born_in Arizona should be added, since we know he was born in Hawaii
            // For safety, case insensitive matches are considered matches

            // Get the set of known slot fill values for this relation for this entity and check
            //  if they are potentially incompatible with the slot value that we are considering adding as a negative example
            // Check whether:
            //   1) we have any known slot fills  for this relation (first sign of potential incompatibility)
            //   2) one of the slot fill values for the relation matches the the one we are considering adding as negative example
            //      (then the slot value is compatible, so we don't want to add it to our negative examples)
            Set<String> knownSlotFills = querier.getKnownSlotValuesForEntityAndRelation(entity, rel);
            boolean relSlotFillsPotentiallyIncompatibleWithThisSlotValue = knownSlotFills.size() > 0;
            for (String knownFill : knownSlotFills) {
              if (Utils.sameSlotFill(key.slotValue, knownFill)) { relSlotFillsPotentiallyIncompatibleWithThisSlotValue = false; break; }
            }
            // Only add the negative label if the slot type is incompatible
            // That is, if the candidate negative label cannot co-occur with any of the positive labels,
            // add it as a negative example.
            boolean compatibleSlotType = true;
            if (Props.TRAIN_NEGATIVES_INCOMPATIBLE) {
              compatibleSlotType = positiveLabels.isEmpty();
              CHECK_COMPATIBLE: for (String positiveLabel : positiveLabels) {
                for (RelationType positiveRel : RelationType.fromString(positiveLabel)) {
                  compatibleSlotType |= rel.plausiblyCooccursWith(positiveRel) || rel == positiveRel;
                  // Good enough, break
                  if (compatibleSlotType) break CHECK_COMPATIBLE;
                }
              }
            }
            // Add negative label, if we can
            // That is, either the relation is single valued and already has a known taken slot,
            // or it's incompatible to begin with.
            if ( (relSlotFillsPotentiallyIncompatibleWithThisSlotValue && rel.cardinality == RelationType.Cardinality.SINGLE) ||
                 !compatibleSlotType) {
              negativeLabels.add(rel.canonicalName);
            }
          } else {
            // Add all negative relations, even if we can't confirm that they're negative
            negativeLabels.add(rel.canonicalName);
          }
        }
        if (Props.TRAIN_NEGATIVES_INCOMPLETE) {
          for (String positive : positiveLabels) { assert !negativeLabels.contains(positive); }
          for (String negative : negativeLabels) { assert !positiveLabels.contains(negative); }
        }
        negativeLabels.removeAll(positiveLabels);
      }

      // Get unknown labels
      Set<String> unknownLabels = new HashSet<String>();
      if (Props.TRAIN_UNLABELED) {
        switch (Props.TRAIN_UNLABELED_SELECT) {
          case NEGATIVE:
            unknownLabels.addAll(negativeLabels);
            break;
          case NOT_POSITIVE:
            unknownLabels.addAll(Sets.diff(allLabels, positiveLabels));
            break;
          case NOT_POSITIVE_WITH_NEGPOS:
            if (positiveLabels.size() > 0 || negativeLabels.size() > 0)
              unknownLabels.addAll(Sets.diff(allLabels, positiveLabels));
            break;
          case NOT_LABELED:
            unknownLabels.addAll(Sets.diff(Sets.diff(allLabels, positiveLabels), negativeLabels));
            break;
          case NOT_LABELED_WITH_NEGPOS:
            if (positiveLabels.size() > 0 || negativeLabels.size() > 0)
              unknownLabels.addAll(Sets.diff(Sets.diff(allLabels, positiveLabels), negativeLabels));
            break;
          default: throw new UnsupportedOperationException("Unsupported train.unlabeled.select " + Props.TRAIN_UNLABELED_SELECT);
        }
      }

      // Decide if we want to add this group
      // We only add the group only if we are going to use unlabeled data, or this group has positive or negative labels
      boolean addGroup = unknownLabels.size() > 0 || positiveLabels.size() > 0 || negativeLabels.size() > 0;

      if (addGroup) {
        // Add datum to dataset
        if (!Props.INDEX_COREF_DO) { group = group.filterFeature(FeatureFactory.COREF_FEATURE); }
        // Regular add datum
        dataset.addDatum( positiveLabels, negativeLabels, unknownLabels, group, group.sentenceGlossKeys);
//        logger.debug("Added datum " + dataset.size() + group.key + " with " + positiveLabels + " pos labels, "
//                + negativeLabels.size() + " neg labels, " + unknownLabels.size() + " unknown labels");
      }
    }

    // Dump dataset to debug track
    if (Props.TRAIN_DUMPDATASET) {
      forceTrack("Dataset Dump");
      for (int group = 0; group < dataset.size(); ++group) {
        // Get relation labels
        Set<String> positiveLabels = dataset.getPositiveLabels(group);
        String positiveGloss = StringUtils.join(positiveLabels, " ");
        Set<String> negativeLabels = dataset.getNegativeLabels(group);
        // Some error checking
        for (String label : dataset.getPositiveLabels(group)) {
          assert !negativeLabels.contains(label);
        }
        for (String label : dataset.getNegativeLabels(group)) {
          assert !positiveLabels.contains(label);
        }
        // Dump sentences
        for (int sent = 0; sent < dataset.getNumSentencesInGroup(group); ++sent) {
          for (CoreMap sentence : this.process.recoverSentenceGloss(dataset.getSentenceGlossKey(group, sent))) {
            String gloss = CoreMapUtils.sentenceToProvenanceString(sentence);
            logger.debug(positiveGloss + " -> " + gloss);
          }
        }
      }
      endTrack("Dataset Dump");
    }
    // Dump relation statistics
    Counter<String> relationCounts = new ClassicCounter<String>();
    for (int i = 0; i < dataset.size(); ++i) {
      for (String posrel : dataset.getPositiveLabels(i)) { relationCounts.incrementCount(posrel); }
    }
    startTrack("Relation Label Distribution");
    for (Pair<String, Double> entry : Counters.toSortedListWithCounts(relationCounts)) {
      logger.log( "" + new DecimalFormat("000.00%").format(entry.second / relationCounts.totalCount()) + ": " + entry.first);
    }
    endTrack("Relation Label Distribution");


    // Post process dataset
    forceTrack("Applying feature count threshold (" + Props.FEATURE_COUNT_THRESHOLD + ")");
    dataset.applyFeatureCountThreshold(Props.FEATURE_COUNT_THRESHOLD);
    endTrack("Applying feature count threshold (" + Props.FEATURE_COUNT_THRESHOLD + ")");

    // Sanity Checks
    if (dataset.size() < numDatumsWithPositiveLabels) { throw new IllegalStateException("Fewer datums in dataset than in input"); }
    // Group size statistics
    int sumSentencesPerPositiveGroup = 0;
    int numPositiveGroups = 0;
    int sumSentencesPerNegativeGroup = 0;
    int numNegativeGroups = 0;
    int sumSentencesPerNoPosNoNegGroup = 0;
    for (int exI = 0; exI < dataset.size(); ++exI) {
      if (dataset.getPositiveLabels(exI).size() > 0) {
        numPositiveGroups += 1;
        sumSentencesPerPositiveGroup += dataset.getNumSentencesInGroup(exI);
      } else if (dataset.getNegativeLabels(exI).size() > 0) {
        // no positive, has negatives
        numNegativeGroups += 1;
        sumSentencesPerNegativeGroup += dataset.getNumSentencesInGroup(exI);
      } else {
        // no positive, no negatives, but still here - must just have unknown labels
        sumSentencesPerNoPosNoNegGroup += dataset.getNumSentencesInGroup(exI);
      }
    }
    // Print info
    startTrack("Dataset Info");
    logger.log(BLUE, "                               size: " + dataset.size());
    logger.log(BLUE, "          number of feature classes: " + dataset.numFeatures());
    logger.log(BLUE, "                number of relations: " + dataset.numClasses());
    logger.log(BLUE, "                  datums in dataset: " + numPositiveGroups + " positive (" + numDatumsWithMultipleLabels + " with multiple relations); " + (dataset.size() - numPositiveGroups) + " negative groups");
    logger.log(BLUE, "average sentences in positive group: " + (((double) sumSentencesPerPositiveGroup) / ((double) numPositiveGroups)));
    logger.log(BLUE, "average sentences in negative group: " + (((double) sumSentencesPerNegativeGroup) / ((double) numNegativeGroups)));
    logger.log(BLUE, "average sentences in unlabeled group: " + (((double) sumSentencesPerNoPosNoNegGroup) / ((double) (dataset.size() - numPositiveGroups - numNegativeGroups))));
    logger.log(BLUE, "         sentence groups considered: " + totalSentenceGroups);
    endTrack("Dataset Info");

    // Sanity Checks
    for (String label : dataset.labelIndex) {
      if (!label.equals(RelationType.fromString(label).orCrash().canonicalName)) {
        throw new IllegalStateException("Unknown relation label being added to dataset: " + label);
      }
    }

    endTrack("Making dataset");
    return dataset;
  }

  public TrainingStatistics run() {
    // Train
    Pair<RelationClassifier, TrainingStatistics> statistics = trainOnTuples(querier.trainingData());
    // Save classifier
    this.classifier = statistics.first;
    try {
      logger.log(BOLD, BLUE, "saving model to " + Props.KBP_MODEL_PATH);
      classifier.save(Props.KBP_MODEL_PATH);
    } catch(IOException e) {
      logger.err("Could not save model.");
      logger.fatal(e);
    }
    // Save statistics
    try {
      IOUtils.writeObjectToFile(statistics.second,
          Props.WORK_DIR.getPath() + File.separator + "train_statistics.ser.gz");
    } catch (IOException e) {
      logger.err(e);
    }
    // Return
    return statistics.second;
  }

 }
