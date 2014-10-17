package edu.stanford.nlp.kbp.slotfilling.classify;

import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.*;

import edu.stanford.nlp.kbp.slotfilling.common.Maybe;
import edu.stanford.nlp.kbp.slotfilling.common.Pointer;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
import edu.stanford.nlp.kbp.slotfilling.common.SentenceGroup;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.ErasureUtils;
import edu.stanford.nlp.util.HashIndex;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.logging.Redwood;
import edu.stanford.nlp.util.logging.RedwoodConfiguration;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * An ensemble of relation extractors.
 * In part, this is used in training as a form of model combination,
 * and as a way to gather information about uncertain examples to annotate for
 * active learning.
 * In another part, this can be used at test time as a form of model combination for
 * multiple classifiers
 *
 * @author jtibs (the vast majority of the code)
 * @author Gabor Angeli (various forms of combination for classify() method; various tweaks)
 */
public class EnsembleRelationExtractor extends RelationClassifier {
  protected static final Redwood.RedwoodChannels logger = Redwood.channels("Ensemble");

  public enum EnsembleMethod {
    DEFAULT,      // train each model over the full train set
    BAGGING,      // bootstrap aggregating (resample the training set with replacement,
                  // and train a model on each sample)
    SUBAGGING     // subsample aggregating (break the training set into subsamples, and
                  // train a model on each sample) 
  }
  private List<RelationClassifier> classifiers;
  private Properties properties;
  private EnsembleMethod method;
  private List<ModelType> modelTypes;

  /** Used by {@link edu.stanford.nlp.kbp.slotfilling.classify.ModelType} via reflection */
  @SuppressWarnings("UnusedDeclaration")
  public EnsembleRelationExtractor(Properties properties) {
    this(properties, Props.TRAIN_ENSEMBLE_METHOD, Props.TRAIN_ENSEMBLE_COMPONENT, Props.TRAIN_ENSEMBLE_NUMCOMPONENTS);
  }
  
  public EnsembleRelationExtractor(Properties properties, EnsembleMethod method,
      ModelType modelType, int numSamples) {
    classifiers = new ArrayList<RelationClassifier>();
    
    this.properties = properties;
    this.method = method;
   
    modelTypes = new ArrayList<ModelType>();
    for (int s = 0; s < numSamples; s++)
      modelTypes.add(modelType);
  }
  
  @SuppressWarnings("UnusedDeclaration")
  public EnsembleRelationExtractor(Properties properties, EnsembleMethod method, List<ModelType> modelTypes) {
    classifiers = new ArrayList<RelationClassifier>();
    
    this.properties = properties;
    this.method = method;
    this.modelTypes = modelTypes;
  }

  public EnsembleRelationExtractor(RelationClassifier... classifiers) {
    this.classifiers = new ArrayList<RelationClassifier>(Arrays.asList(classifiers));
    this.method = Props.TRAIN_ENSEMBLE_METHOD;  // irrelevant, as everyone is already trained
    // Infer the model types of the classifiers; though this is also largely irrelevant
    this.modelTypes = new ArrayList<ModelType>(this.classifiers.size());
    for (RelationClassifier classifier : classifiers) {
      for (ModelType candidateType : ModelType.values()) {
        if (candidateType.modelClass.isAssignableFrom(classifier.getClass())) {
          this.modelTypes.add(candidateType);
          break;
        }
      }
    }
    while (modelTypes.size() != classifiers.length) {
      logger.warn("Could not find some model class");
      modelTypes.add(ModelType.GOLD);
    }
    this.properties = new Properties();
  }

  public int numSamples() {
    return modelTypes.size();
  }

  @Override
  public TrainingStatistics train(KBPDataset<String, String> trainSet) {
    final Pointer<TrainingStatistics> result = new Pointer<TrainingStatistics>();
    int numSamples = modelTypes.size();
    boolean haveClassifiersAlready = (classifiers != null && numSamples == classifiers.size());
    
    startTrack("Generating samples");
    final List<KBPDataset<String,String>> samples = generateSamples(trainSet, numSamples);
    logger.log("applying feature count thresholds (" + Props.FEATURE_COUNT_THRESHOLD + ")...");
    for (final KBPDataset<String, String> dataset : samples) {
      dataset.applyFeatureCountThreshold(Props.FEATURE_COUNT_THRESHOLD);
    }
    endTrack("Generating samples");

    List<Runnable> folds = new ArrayList<Runnable>();

    for (int s = 0; s < numSamples; s++) {
      File origDir = Props.KBP_MODEL_DIR;
      // make a new subdirectory to hold the serialized model for this sample
      File directory = new File(origDir.getPath() + File.separatorChar + "sample" + s);
      //noinspection ResultOfMethodCallIgnored
      directory.mkdirs();
      Props.KBP_MODEL_DIR = directory;

      // create classifiers
      forceTrack("Creating classifier on subsample #" + s);
      final RelationClassifier classifier = haveClassifiersAlready
          ? this.classifiers.get(s)
          : modelTypes.get(s).construct(properties);
      Props.KBP_MODEL_DIR = origDir;
      final int sampleIndex = s;
      folds.add(new Runnable() {
        @Override
        public void run() {
          TrainingStatistics statistics = classifier.train(samples.get(sampleIndex));
          synchronized (EnsembleRelationExtractor.this) {
            if (!result.dereference().isDefined()) result.set(statistics);
            else result.dereference().get().merge(statistics);
          }
        }
      });
      classifiers.add(classifier);
      endTrack("Creating classifier on subsample #" + s);
    }

    // Run classifiers
    // JointBayes multithreads on its own -- don't capture stderr if this is the case.
    if (numSamples > 1 && !(Props.TRAIN_ENSEMBLE_COMPONENT == ModelType.JOINT_BAYES && !Props.TRAIN_JOINTBAYES_MULTITHREAD)) { RedwoodConfiguration.current().restore(System.err).apply(); }
    threadAndRun(folds, numSamples);
    if (numSamples > 1 && !(Props.TRAIN_ENSEMBLE_COMPONENT == ModelType.JOINT_BAYES && !Props.TRAIN_JOINTBAYES_MULTITHREAD)) { RedwoodConfiguration.current().capture(System.err).apply(); }

    // merge statistics
    statistics = Maybe.Just(result.dereference().get());
    return result.dereference().get();
  }

  private List<KBPDataset<String,String>> generateSamples(
      KBPDataset<String, String> trainSet, int numSamples) {
    switch (method) {
      case DEFAULT:
        return cloneData(trainSet, numSamples);
      case BAGGING:
        return sampleData(trainSet, numSamples);
      case SUBAGGING:
        return partitionData(trainSet, numSamples);
      default:
        throw new RuntimeException("Unsupported ensemble method " + method.name() + ".");
    }
  }

  private List<KBPDataset<String, String>> cloneData(KBPDataset<String, String> trainSet, int numSamples) {
    // Get variables we'll be using
    List<KBPDataset<String, String>> result = new ArrayList<KBPDataset<String, String>>();
    Set<Integer>[] posLabels = trainSet.getPositiveLabelsArray();
    Set<Integer>[] negLabels = trainSet.getNegativeLabelsArray();
    Set<Integer>[] unkLabels = trainSet.getUnknownLabelsArray();
    int[][][] data = trainSet.getDataArray();

    // Copy the datasets
    // note[gabor]: this is necessary since JointBayes shuffles the data around
    for (int s = 0; s < numSamples; s++) {
      KBPDataset<String, String> clone = new KBPDataset<String,String>(
          new HashIndex<String>(trainSet.featureIndex()),
          new HashIndex<String>(trainSet.labelIndex()));
      for (int i = 0; i < trainSet.size(); ++i) {
        clone.addDatum(posLabels[i], negLabels[i], unkLabels[i], data[i], trainSet.getSentenceGlossKey(i));
      }
      result.add(clone);
    }

    // Return the result
    return result;
  }

  private List<KBPDataset<String, String>> partitionData(
      KBPDataset<String, String> trainSet, int numSamples) {
    List<KBPDataset<String, String>> result = new ArrayList<KBPDataset<String, String>>();
    logger.log("numSamples: " + numSamples);
    Set<Integer>[] posLabels = trainSet.getPositiveLabelsArray();
    Set<Integer>[] negLabels = trainSet.getNegativeLabelsArray();
    Set<Integer>[] unkLabels = trainSet.getUnknownLabelsArray();
    int[][][] data = trainSet.getDataArray();
    
    for (int p = 0; p < numSamples; p++) {
      result.add(new KBPDataset<String,String>(new HashIndex<String>(trainSet.featureIndex()),
                                                new HashIndex<String>(trainSet.labelIndex())));
    }
    
    final int n = trainSet.size();
    Integer[] indices = new Integer[n];
    for (int i = 0; i < n; ++i) { indices[i] = i; }
    Collections.shuffle(Arrays.asList(indices));

    for (int indexI = 0; indexI < n; indexI++) {
      int i = indices[indexI];
      int partition = indexI % numSamples;
      assert partition < result.size();

      for (int offset = 0; offset < Props.TRAIN_ENSEMBLE_SUBAGREDUNDANCY; ++offset) {
        result.get( (partition + offset) % result.size() ).addDatum(posLabels[i], negLabels[i], unkLabels[i], data[i],
            trainSet.getSentenceGlossKey(i));
      }
    }
    return result;
  }
  
  private List<KBPDataset<String, String>> sampleData(
      KBPDataset<String, String> trainSet, int numSamples) {

    Set<Integer>[] posLabels = trainSet.getPositiveLabelsArray();
    Set<Integer>[] negLabels = trainSet.getNegativeLabelsArray();
    Set<Integer>[] unkLabels = trainSet.getUnknownLabelsArray();
    int[][][] data = trainSet.getDataArray();
    
    List<KBPDataset<String, String>> result = new ArrayList<KBPDataset<String, String>>();
    logger.log("numSamples: " + numSamples);
    final int n = Math.min(Props.TRAIN_ENSEMBLE_BAGSIZE, trainSet.size());

    for (int p = 0; p < numSamples; p++) {
      Random random = new Random(p);  // new random for each sample
      KBPDataset<String, String> sample = new KBPDataset<String,String>(new HashIndex<String>(trainSet.featureIndex()),
          new HashIndex<String>(trainSet.labelIndex()));
      
      for (int i = 0; i < n; i++) {
        int index = random.nextInt(n);
        sample.addDatum(posLabels[index], negLabels[index], unkLabels[index],
            data[index], trainSet.getSentenceGlossKey(index));
      }
      result.add(sample);
    }
    return result;
  }
  
  @Override
  public Counter<Pair<String,Maybe<KBPRelationProvenance>>> classifyRelations(SentenceGroup group, Maybe<CoreMap[]> rawSentences) {
    Counter<Pair<String, Maybe<KBPRelationProvenance>>> result = new ClassicCounter<Pair<String, Maybe<KBPRelationProvenance>>>();

    // Intermediate Variables
    Map<String, KBPRelationProvenance> valueToProvenance = new HashMap<String, KBPRelationProvenance>();
    Set<String> relationPredictions = new HashSet<String>();
    Counter<String> highestWeightForPrediction = new ClassicCounter<String>();
    List<Counter<String>> classifierPredictions = new ArrayList<Counter<String>>();

    // Collect Predictions And Statistics
    for (RelationClassifier classifier : classifiers) {
      Counter<String> predictions = new ClassicCounter<String>();
      for (Map.Entry<Pair<String, Maybe<KBPRelationProvenance>>, Double> entry : classifier.classifyRelations(group, rawSentences).entrySet()) {
        predictions.incrementCount(entry.getKey().first, entry.getValue());  // register prediction
        relationPredictions.add(entry.getKey().first);                       // add to key set
        if (entry.getKey().second.isDefined() &&                             // register provenance if highest weight so far
            highestWeightForPrediction.getCount(entry.getKey().first) < entry.getValue()) {
          valueToProvenance.put(entry.getKey().first, entry.getKey().second.get());
        }
        highestWeightForPrediction.setCount(entry.getKey().first,            // update highest confidence weight
                                            Math.max(highestWeightForPrediction.getCount(entry.getKey().first), entry.getValue()));
      }
      classifierPredictions.add(predictions);
    }

    // Populate Results
    for (String relation : relationPredictions) {
      // Assess classifier agreement
      int classifiersWhoAgree = 0;
      boolean firstClassifierAgrees = classifierPredictions.get(0).containsKey(relation);
      double firstClassifierWeight = classifierPredictions.get(0).getCount(relation);
      double noisyOrInverse = 1.0;
      double maxWeight = 0.0;
      double secondMaxWeight = 0.0;
      for (Counter<String> prediction : classifierPredictions) {
        if (prediction.containsKey(relation)) {
          double weight = prediction.getCount(relation);
          classifiersWhoAgree += 1;
          noisyOrInverse *= (1.0 - weight);
          if (weight > maxWeight) {
            secondMaxWeight = maxWeight;
            maxWeight = weight;
          } else if (weight > secondMaxWeight) {
            secondMaxWeight = weight;
          }
        }
      }
      // Add relation
      switch (Props.TEST_ENSEMBLE_COMBINATION) {
        case AGREE_ANY:
          if (classifiersWhoAgree > 0) {
            result.setCount(Pair.makePair(relation, Maybe.fromNull(valueToProvenance.get(relation))), 1.0 - noisyOrInverse);
          }
          break;
        case AGREE_ALL:
          if (classifiersWhoAgree >= classifiers.size()) {
            result.setCount(Pair.makePair(relation, Maybe.fromNull(valueToProvenance.get(relation))), 1.0 - noisyOrInverse);
          }
          break;
        case AGREE_MOST:
          if (classifiersWhoAgree >= classifiers.size() / 2) {
            result.setCount(Pair.makePair(relation, Maybe.fromNull(valueToProvenance.get(relation))), 1.0 - noisyOrInverse);
          }
          break;
        case AGREE_TWO:
          if (classifiersWhoAgree >= 2) {
            result.setCount(Pair.makePair(relation, Maybe.fromNull(valueToProvenance.get(relation))),
                            1.0 - (1.0 - maxWeight)*(1.0 - secondMaxWeight));  // noisy or of top two weights
          }
          break;
        case AGREE_FIRST:
          if (firstClassifierAgrees) {
            result.setCount(Pair.makePair(relation, Maybe.fromNull(valueToProvenance.get(relation))), firstClassifierWeight);
          }
          break;
        default:
          throw new IllegalStateException("Unknown combination method for ensemble model: " + Props.TEST_ENSEMBLE_COMBINATION);
      }
    }

    return result;
  }

  @Override
  public void load(ObjectInputStream in) throws IOException, ClassNotFoundException {
    int numSamples = in.readInt();
    method = ErasureUtils.uncheckedCast(in.readObject());
    
    modelTypes =  new ArrayList<ModelType>();
    for (int s = 0; s < numSamples; s++) {
      ModelType type = ErasureUtils.uncheckedCast(in.readObject());
      modelTypes.add(type);
    }
    
    classifiers =  new ArrayList<RelationClassifier>();
    for (int s = 0; s < numSamples; s++) {
      RelationClassifier classifier = modelTypes.get(s).construct(properties);
      classifier.load(in);
      classifiers.add(classifier);
    }
  }

  @Override
  public void save(ObjectOutputStream out) throws IOException {
    out.writeInt(modelTypes.size());
    out.writeObject(method);
    for (ModelType type : modelTypes)
      out.writeObject(type);
    for (RelationClassifier classifier : classifiers)
      classifier.save(out);
  }
}
