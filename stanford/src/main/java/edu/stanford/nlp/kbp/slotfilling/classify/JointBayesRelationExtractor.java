package edu.stanford.nlp.kbp.slotfilling.classify;

import edu.stanford.nlp.classify.*;
import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.kbp.slotfilling.train.KBPTrainer;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.*;
import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * Implements the MIML-RE model from EMNLP 2012
 * @author Mihai
 * @author Julie Tibshirani (jtibs)
 * @author nallapat@ai.sri.com
 *
 */
@SuppressWarnings("ConstantConditions")
public class JointBayesRelationExtractor extends RelationClassifier {
  private static final long serialVersionUID = -7961154075748697901L;
  private static final Redwood.RedwoodChannels logger = Redwood.channels("MIML-RE");

  private final boolean partOfEnsemble = Props.TRAIN_MODEL == ModelType.ENSEMBLE;

  static enum LOCAL_CLASSIFICATION_MODE {
    WEIGHTED_VOTE,
    SINGLE_MODEL
  }

  public static enum InferenceType {
    SLOW,
    STABLE
  }

  private static final LOCAL_CLASSIFICATION_MODE localClassificationMode =
    LOCAL_CLASSIFICATION_MODE.WEIGHTED_VOTE;

  /**
   * sentence-level multi-class classifier, trained across all sentences
   * one per fold to avoid overfitting
   */
  protected LinearClassifier<String, String> [] zClassifiers;
  /** this is created only if localClassificationMode == SINGLE_MODEL */
  LinearClassifier<String, String> zSingleClassifier;
  /** one two-class classifier for each top-level relation */
  protected Map<String, LinearClassifier<String, String>> yClassifiers;

  private Index<String> featureIndex;
  private Index<String> yLabelIndex;
  protected Index<String> zLabelIndex;

  private static String ATLEASTONCE_FEAT = "atleastonce";
  private static String NONE_FEAT = "none";
  private static String UNIQUE_FEAT = "unique";
  private static String SIGMOID_FEAT = "sigmoid";
  private static List<String> Y_FEATURES_FOR_INITIAL_MODEL;

  static {
    Y_FEATURES_FOR_INITIAL_MODEL = new ArrayList<String>();
    Y_FEATURES_FOR_INITIAL_MODEL.add(NONE_FEAT);
    if (Props.TRAIN_JOINTBAYES_YFEATURES.contains(Props.Y_FEATURE_CLASS.ATLEAST_ONCE)) {
      Y_FEATURES_FOR_INITIAL_MODEL.add(ATLEASTONCE_FEAT);
    }
    if (Props.TRAIN_JOINTBAYES_YFEATURES.contains(Props.Y_FEATURE_CLASS.UNIQUE)) {
      Y_FEATURES_FOR_INITIAL_MODEL.add(UNIQUE_FEAT);
    }
    if (Props.TRAIN_JOINTBAYES_YFEATURES.contains(Props.Y_FEATURE_CLASS.SIGMOID)) {
      Y_FEATURES_FOR_INITIAL_MODEL.add(SIGMOID_FEAT);
    }
  }

  /** Run EM for this many epochs */
  private final int numberOfTrainEpochs;

  /** Organize the Z classifiers into this many folds for cross validation */
  protected int numberOfFolds;

  /**
   * Should we skip the EM loop?
   * If true, this is essentially a multiclass local LR classifier
   */
  private final boolean onlyLocalTraining;

  /** Required to know where to save the initial models */
  private final String initialModelPath;

  /** Sigma for the Z classifiers */
  private final double zSigma;
  /** Sigma for the Y classifiers */
  private final double ySigma;

  /** Counts number of flips for Z labels in one epoch */
  private AtomicInteger zUpdatesInOneEpoch = new AtomicInteger(0);

  private final LocalFilter localDataFilter;

  private final InferenceType inferenceType;

  /** Should we train Y models? */
  private final boolean trainY;

  /** These label dependencies were seen in training */
  protected Set<String> knownDependencies;

  private String serializedModelPath;

  /** Number of threads to use */
  private int numberOfThreads = -1;

  private final KBPTrainer.MinimizerType zClassifierMinimizerType;

  private final Object lock = "I'm a lock :)";

  /** A constructor to match the signature of other classifiers -- don't delete me, even if I don't appear used! */
  public JointBayesRelationExtractor(Properties props) {
    this(props, false);
  }

  public JointBayesRelationExtractor(Properties props, boolean onlyLocal) {
    // We need workDir, serializedRelationExtractorName, modelType, samplingRatio to serialize the initial models
    String srn = "kbp_relation_model";
    if (srn.endsWith(Props.SER_EXT))
      srn = srn.substring(0, srn.length() - Props.SER_EXT.length());
    String serializedRelationExtractorName = srn;
    double samplingRatio = Props.TRAIN_NEGATIVES_SUBSAMPLERATIO;
    if (Props.TRAIN_JOINTBAYES_LOADINITMODEL_FILE != null) {
      initialModelPath = Props.TRAIN_JOINTBAYES_LOADINITMODEL_FILE;
    } else {
      initialModelPath = makeInitialModelPath(
          Props.KBP_MODEL_DIR.getPath(),
          serializedRelationExtractorName,
          ModelType.JOINT_BAYES,
          samplingRatio);
    }
    numberOfTrainEpochs = Props.TRAIN_JOINTBAYES_EPOCHS;
    numberOfFolds = Props.TRAIN_JOINTBAYES_FOLDS;
    if (numberOfFolds < 2) {
      throw new IllegalArgumentException("Must have at least two folds: " + numberOfFolds);
    }
    numberOfThreads = Props.TRAIN_JOINTBAYES_MULTITHREAD ? Execution.threads : 1;
    zSigma = Props.TRAIN_JOINTBAYES_ZSIGMA;
    ySigma = 1.0;
    localDataFilter = MetaClass.create(Props.TRAIN_JOINTBAYES_FILTER).createInstance();
    inferenceType = Props.TRAIN_JOINTBAYES_INFERENCETYPE;
    trainY = Props.TRAIN_JOINTBAYES_TRAINY;
    onlyLocalTraining = onlyLocal;
    serializedModelPath = makeModelPath(
        Props.KBP_MODEL_DIR.getPath(),
        serializedRelationExtractorName,
        ModelType.JOINT_BAYES,
        samplingRatio);
    zClassifierMinimizerType = Props.TRAIN_JOINTBAYES_ZMINIMIZER;
    log(BLUE, "y features: " + StringUtils.join(Props.TRAIN_JOINTBAYES_YFEATURES, " | "));
  }

  private static String makeInitialModelPath(
      String workDir,
      String serializedRelationExtractorName,
      ModelType modelType,
      double samplingRatio) {
    return workDir + File.separator + serializedRelationExtractorName +
      "." + modelType + "." + (int) (100.0 * samplingRatio) +
      ".initial" + Props.SER_EXT;
  }
  private static String makeModelPath(
      String workDir,
      String serializedRelationExtractorName,
      ModelType modelType,
      double samplingRatio) {
    return workDir + File.separator + serializedRelationExtractorName +
      "." + modelType + "." + (int) (100.0 * samplingRatio) +
      Props.SER_EXT;
  }

  private int foldStart(int fold, int size) {
    int foldSize = size / numberOfFolds;
    assert(foldSize > 0);
    int start = fold * foldSize;
    assert(start < size);
    return start;
  }

  private int foldEnd(int fold, int size) {
    // padding if this is the last fold
    if(fold == numberOfFolds - 1)
      return size;

    int foldSize = size / numberOfFolds;
    assert(foldSize > 0);
    int end = (fold + 1) * foldSize;
    assert(end <= size);
    return end;
  }

  private int [][] initializeZLabels(KBPDataset<String, String> data) {
    // initialize Z labels with the predictions of the local classifiers
    int[][] zLabels = new int[data.getDataArray().length][];
    for(int f = 0; f < numberOfFolds; f ++){
      LinearClassifier<String, String> zClassifier = zClassifiers[f];
      assert(zClassifier != null);
      for(int i = foldStart(f, data.getDataArray().length); i < foldEnd(f, data.getDataArray().length); i ++){
        int [][] group = data.getDataArray()[i];
        zLabels[i] = new int[group.length];
        for(int j = 0; j < group.length; j ++){
          int [] datum = group[j];
          for (int point : datum) {
            if (point >= this.featureIndex.size()) {
              logger.log(point + "\t" + this.featureIndex.size());
            }
            assert point < this.featureIndex.size();
            assert point < zClassifier.featureIndex().size();
            assert point >= 0;
          }
          Counter<String> scores = zClassifier.scoresOf(datum);
          List<Pair<String, Double>> sortedScores = sortPredictions(scores);
          int sys = zLabelIndex.indexOf(sortedScores.get(0).first());
          if (sys < 0) throw new IllegalStateException("Unknown relation: " + sortedScores.get(0).first());
          assert(sys != -1);
          zLabels[i][j] = sys;
        }
      }
    }

    return zLabels;
  }

  private void detectDependencyYFeatures(KBPDataset<String, String> data) {
    knownDependencies = new HashSet<String>();
    for(int i = 0; i < data.size(); i ++){
      Set<Integer> labels = data.getPositiveLabelsArray()[i];
      for(Integer src: labels) {
        String srcLabel = data.labelIndex().get(src);
        for(Integer dst: labels) {
          if(src.intValue() == dst.intValue()) continue;
          String dstLabel = data.labelIndex().get(dst);
          String f = makeCoocurrenceFeature(srcLabel, dstLabel);
          logger.debug("FOUND COOC: " + f);
          knownDependencies.add(f);
        }
      }
    }
  }

  private static String makeCoocurrenceFeature(String src, String dst) {
    return "co:s|" + src + "|d|" + dst + "|";
  }

  /**
   * Train a classifier for inferring the hidden Z labels, given a dataset
   * @param zFactory The factory to create the classifier from
   * @param zDataset The dataset to use for training
   * @param epoch The current epoch of training
   * @param fold The current fold of training
   * @return A Runnable which performs this task.
   */
  private Runnable createZClassifierTrainer(final LinearClassifierFactory<String, String> zFactory,
                                            Dataset<String, String> zDataset,
                                            final int epoch, final int fold) {
    final int[][] dataArray = zDataset.getDataArray();
    final int[] labelsArray = zDataset.getLabelsArray();
    final double[][] initWeights = (zClassifiers[fold] != null)? zClassifiers[fold].weights(): null;

    // Some Debugging
    Counter<String> relationCounts = new ClassicCounter<String>();
    for (int label : labelsArray) { relationCounts.incrementCount(zLabelIndex.get(label)); }
    if (!partOfEnsemble || !Props.TRAIN_JOINTBAYES_MULTITHREAD) { startTrack("Relation Label Distribution"); }
    for (Pair<String, Double> entry : Counters.toSortedListWithCounts(relationCounts)) {
      logger.log( "" + new DecimalFormat("000.00%").format(entry.second / relationCounts.totalCount()) + ": " + entry.first);
    }
    if (!partOfEnsemble || !Props.TRAIN_JOINTBAYES_MULTITHREAD) { endTrack("Relation Label Distribution"); }

    // Create Trainer
    return new Runnable() {
      public void run(){
        String title = "EPOCH " + epoch + ": Training Z classifier for fold #" + fold;
        if (!partOfEnsemble || !Props.TRAIN_JOINTBAYES_MULTITHREAD) { startTrack(title); }
        int [][] foldTrainArray = makeTrainDataArrayForFold(dataArray, fold);
        int [] foldTrainLabels = makeTrainLabelArrayForFold(labelsArray, fold);
        Dataset<String, String> zd = new Dataset<String, String>(zLabelIndex, foldTrainLabels, featureIndex, foldTrainArray);
//        LinearClassifier<String, String> zClassifier = zFactory.trainClassifierWithInitialWeights(zd, initWeights); // TODO(gabor) this doesn't work for some reason?
        LinearClassifier<String, String> zClassifier = zFactory.trainClassifier(zd);

        synchronized (lock) {
          zClassifiers[fold] = zClassifier;
        }
        if (!partOfEnsemble || !Props.TRAIN_JOINTBAYES_MULTITHREAD) { Redwood.endTrack(title); } else { Redwood.finishThread(); }
      }
    };
  }

  /**
   * Create a classifier for inferring the final Y relation
   * @param yFactory The factory to create the classifier from
   * @param yDatasets The dataset to use for training
   * @param yLabel The observed Y label
   * @param epoch The current epoch of training
   * @return A runnable which performs this task.
   */
  private Runnable createYClassifierTrainer(final LinearClassifierFactory<String, String> yFactory,
                                            Map<String, RVFDataset<String, String>> yDatasets,
                                            final String yLabel, final int epoch) {
    final RVFDataset<String, String> trainSet = yDatasets.get(yLabel);

    if(trainSet.size() == 0 && !Props.JUNIT) {
      logger.debug("Empty train set.  yLabel="+yLabel);
      throw new RuntimeException("[JointBayesRelationExtractor.createYClassifierTrainer] Empty train set.  yLabel="+yLabel);
    }

    int[] allLabels = trainSet.getLabelsArray();
    boolean hasMultipleValues = false;
    for(int i = 0; i < allLabels.length - 1; i++) {
      if(allLabels[i] != allLabels[i+1]) {
        hasMultipleValues = true;
        break;
      }
    }
    if(!hasMultipleValues) {
      logger.debug("Train set all same value.  val="+ (allLabels.length > 0 ? allLabels[0] : "none") +"  yLabel="+yLabel);
    }

    return new Runnable() {
      public void run(){
        String title = "EPOCH " + epoch + ": Training Y classifier for label " + yLabel;
        if (!partOfEnsemble || !Props.TRAIN_JOINTBAYES_MULTITHREAD) { startTrack(title); }
        LinearClassifier<String, String> yClassifer = yFactory.trainClassifier(trainSet);
        synchronized (lock) {
          yClassifiers.put(yLabel, yClassifer);
        }
        if (!partOfEnsemble || !Props.TRAIN_JOINTBAYES_MULTITHREAD) { endTrack(title); } else { Redwood.finishThread(); }
      }
    };
  }

  /**
   * Run inference to label the latent Z variables
   * @param zClassifier The classifier to use for the inference
   * @param yDatasets The datasets for the Y labeler to create
   * @param data The raw data we are training from
   * @param zLabelsPredictedByZ A variable for storing the local predictions of the Z variables
   * @param zLabels A variable for the globally inferred predictions of the Z variables
   * @param epoch The current epoch
   * @param groupIndex The index of the sentence group we are classifying
   * @return A runnable which performs this task
   */
  private Runnable createZLabeller(final LinearClassifier<String, String> zClassifier,
                                   final Map<String, RVFDataset<String, String>> yDatasets,
                                   final KBPDataset<String, String> data,
                                   int[][] zLabelsPredictedByZ,
                                   int[][] zLabels,
                                   final int epoch,
                                   final int groupIndex,
                                   final Pointer<Triple<int[], Counter<String>[], double[]>> confidences) {
    // Copy data to prevent concurrency bugs
    int[][][] rawData = data.getDataArray();
    final int[][] group = rawData[groupIndex];
//    final int[][] group = new int[rawData[groupIndex].length][];
//    for (int sentI = 0; sentI < group.length; ++sentI) {
//      group[sentI] = new int[rawData[groupIndex][sentI].length];
//      System.arraycopy(rawData[groupIndex][sentI], 0, group[sentI], 0, group[sentI].length);
//    }
    final Set<Integer> positiveLabels = data.getPositiveLabelsArray()[groupIndex];
    final Set<Integer> negativeLabels = data.getNegativeLabelsArray()[groupIndex];
    final int[] zLabelsPredictedByZi = zLabelsPredictedByZ[groupIndex];
    final int[] zLabelsi = zLabels[groupIndex];
    final Maybe<String>[] fixedZ = data.getAnnotatedLabels(groupIndex);
//    @SuppressWarnings("unchecked") final Maybe<String>[] fixedZ = new Maybe[data.getAnnotatedLabels(groupIndex).length];
//    System.arraycopy(data.getAnnotatedLabels(groupIndex), 0, fixedZ, 0, fixedZ.length);
    assert(group.length == fixedZ.length);
    // Create runnable
    return new Runnable() {
      public void run() {

        int[] originalIndex;
        Counter<String> [] jointZProbs =
            ErasureUtils.uncheckedCast(new Counter[group.length]);

        synchronized (group) {
          originalIndex = randomizeGroup(group, fixedZ, epoch);


          predictZLabels(group, zLabelsPredictedByZi, zClassifier);

          switch(inferenceType) {
            case SLOW:
              confidences.set( Triple.makeTriple(
                  originalIndex, jointZProbs,
                  inferZLabels(group, positiveLabels, negativeLabels, zLabelsi, fixedZ, jointZProbs, zClassifier, epoch) ));
              break;
            case STABLE:
              confidences.set( Triple.makeTriple(
                  originalIndex, jointZProbs,
                  inferZLabelsStable(group, positiveLabels, negativeLabels, zLabelsi, fixedZ, jointZProbs, zClassifier, epoch) ));
              break;
            default:
              throw new RuntimeException("ERROR: unknown inference type: " + inferenceType);
          }

          // given these predicted z labels, update the features in the y dataset
          //printGroup(zLabels[i], positiveLabels);
          synchronized (lock) {
            for (int y : positiveLabels) {
              String yLabel = yLabelIndex.get(y);
              addYDatum(yDatasets.get(yLabel), yLabel, zLabelsi, jointZProbs, true);
            }
            for (int y : negativeLabels) {
              String yLabel = yLabelIndex.get(y);
              addYDatum(yDatasets.get(yLabel), yLabel, zLabelsi, jointZProbs, false);
            }
          }
        }
      }
    };
  }

  @Override
  public TrainingStatistics train(KBPDataset<String, String> data) {
    if (numberOfThreads <= 0) numberOfThreads = Runtime.getRuntime().availableProcessors();
    logger.log("Number of threads is " + numberOfThreads);
    // filter some of the groups
    forceTrack("Filtering data");
    if(localDataFilter instanceof LargeFilter) {
      // TODO(gabor) why was this hidden in an If?
      List<int [][]> filteredGroups = new ArrayList<int[][]>();
      List<String[]> filteredSentenceGloss = new ArrayList<String[]>();
      List<Set<Integer>> filteredPosLabels = new ArrayList<Set<Integer>>();
      List<Set<Integer>> filteredNegLabels = new ArrayList<Set<Integer>>();
      List<Set<Integer>> filteredUnkLabels = new ArrayList<Set<Integer>>();
      List<Maybe<String>[]> filteredAnnotatedLabels = new ArrayList<Maybe<String>[]>();
      for(int i = 0; i < data.size(); i ++) {
        if(localDataFilter.filterY(data.getDataArray()[i], data.getPositiveLabelsArray()[i])) {
          filteredGroups.add(data.getDataArray()[i]);
          filteredPosLabels.add(data.getPositiveLabelsArray()[i]);
          filteredNegLabels.add(data.getNegativeLabelsArray()[i]);
          filteredUnkLabels.add(data.getUnknownLabelsArray()[i]);
          filteredAnnotatedLabels.add(data.getAnnotatedLabels(i));
          filteredSentenceGloss.add(data.getSentenceGlossKey(i));
        }
      }
      data = new KBPDataset<String, String>(
          filteredGroups.toArray(new int[filteredGroups.size()][][]),
          data.featureIndex(), data.labelIndex,
          filteredPosLabels.toArray(ErasureUtils.<Set<Integer> []>uncheckedCast(new Set[filteredPosLabels.size()])),
          filteredNegLabels.toArray(ErasureUtils.<Set<Integer> []>uncheckedCast(new Set[filteredNegLabels.size()])),
          filteredUnkLabels.toArray(ErasureUtils.<Set<Integer> []>uncheckedCast(new Set[filteredUnkLabels.size()])),
          filteredAnnotatedLabels.toArray(ErasureUtils.<Maybe<String>[][]>uncheckedCast(new Maybe[filteredAnnotatedLabels.size()][])),
          filteredSentenceGloss.toArray(new String[filteredSentenceGloss.size()][]));
    }
    endTrack("Filtering data");

    LinearClassifierFactory<String, String> zFactory = getZClassifierFactory();
    LinearClassifierFactory<String, String> yFactory =
      new LinearClassifierFactory<String, String>(1e-4, false, ySigma);
    zFactory.setVerbose(false);
    yFactory.setVerbose(false);
    logger.log("Created classifiers");

    startTrack("Initializing classifiers");
    boolean reInitialize = true;
    if(initialModelPath != null && new File(initialModelPath).exists() && Props.TRAIN_JOINTBAYES_LOADINITMODEL) {
      // Try to load initial model
      try {
        loadInitialModels(initialModelPath);
        if (data.featureIndex().size() > this.featureIndex.size()) {
          logger.warn(RED, "Loaded an initial model with fewer features than the dataset! Ignoring...");
        } else {
          reInitialize = false;
        }
        // yClassifiers = initializeYClassifiersWithAtLeastOnce(yLabelIndex);
      } catch (Exception e1) {
        throw new RuntimeException(e1);
      }
    } else if (Props.TRAIN_JOINTBAYES_LOADINITMODEL) {
      if (initialModelPath == null) {
        logger.warn(RED, "Cannot load initial model: no initial model path");
      } else if (!new File(initialModelPath).exists()) {
        logger.warn(RED, "Cannot load initial model: model does not exist at " + initialModelPath);
      }
    }
    if (reInitialize) {
      featureIndex = data.featureIndex();
      yLabelIndex = data.labelIndex;
      zLabelIndex = new HashIndex<String>(yLabelIndex);
      zLabelIndex.add(RelationMention.UNRELATED);

      // initialize classifiers
      zClassifiers = initializeZClassifierLocally(data, featureIndex, zLabelIndex);
      yClassifiers = initializeYClassifiersWithAtLeastOnce(yLabelIndex);

      if(initialModelPath != null) {
        try {
          saveInitialModels(initialModelPath);
        } catch (IOException e1) {
          logger.err(RED, "Could not save initial model: " + e1.getMessage());
        }
      }
    }
    endTrack("Initializing classifiers");

    // stop training after initialization
    // this is essentially a local model!
    // TODO(gabor) There really are statistics to extract from here
    if(onlyLocalTraining) return TrainingStatistics.undefined();

    detectDependencyYFeatures(data);

    for(String y: yLabelIndex) {
      int yi = yLabelIndex.indexOf(y);
      if (yi < 0) throw new IllegalStateException("Unknown relation: " + y);
      logger.log("YLABELINDEX " + y + " = " + yi);
    }

    // calculate total number of sentences
    int totalSentences = 0;
    for (int[][] group : data.getDataArray())
      totalSentences += group.length;

    // initialize predicted z labels
    int[][] zLabels = initializeZLabels(data);
    computeConfusionMatrixForCounts("LOCAL", zLabels, data.getPositiveLabelsArray());
    computeYScore("LOCAL", zLabels, data.getPositiveLabelsArray());

    // z dataset initialized with nil labels and the sentence-level features array
    // Dataset<String, String> zDataset = initializeZDataset(totalSentences, zLabels, data.getDataArray());

    // y dataset initialized to be empty, as it will be populated during the E step
    Map<String, RVFDataset<String, String>> yDatasets = initializeYDatasets();

    // For Confidence estimation of Z labels
    TrainingStatistics statistics = TrainingStatistics.empty();
    Pair<Counter<String>[], double[]>[] confidences = new Pair[data.size()];

    boolean guessYLabels = Props.TRAIN_UNLABELED;
    if (guessYLabels) {
      // Save original labels
      data.finalizeLabels();
    }
    // run EM
    startTrack("EM");
    for (int epoch = 0; epoch < numberOfTrainEpochs; epoch++) {
      zUpdatesInOneEpoch = new AtomicInteger(0);
      logger.log("***EPOCH " + epoch + "***");

      // we compute scores in each epoch using these labels
      int [][] zLabelsPredictedByZ = new int[zLabels.length][];
      for(int i = 0; i < zLabels.length; i ++)
        zLabelsPredictedByZ[i] = new int[zLabels[i].length];

      //
      // E-step
      //
      forceTrack("E-Step");

      if (guessYLabels && epoch > 0) {
        forceTrack("Guessing Y labels");
        // This implements the "Distant Supervision for Relation Extraction with an Incomplete Knowledge Base"
        //   extension to MIML-RE from Min et al at NAACL2013

        // restore original labels
        data.restoreLabels();
        Set<Integer>[] posLabels = data.getPositiveLabelsArray();
        Set<Integer>[] negLabels = data.getNegativeLabelsArray();

        // Using the overall z classifier - determine the most likely positive relations
        // For each datum group, for each unknown label that has not already been marked as positive,
        // compute the probability that the label is positive
        Set<Integer>[] unkLabels = data.getUnknownLabelsArray();

        int nPositive = data.countLabels(posLabels);
        int nNegative = data.countLabels(negLabels);
        int nUnknown = data.countLabels(unkLabels);

        int[][][] rawData = data.getDataArray();
        // Use priority queue to track the (n*theta - nPositive) datum group and relation
        // It is unclear from the paper whether n is the number of bags or number of bags*relations
//        int expectedPositive = (int) (Props.TRAIN_JOINTBAYES_PERCENT_POSITIVE * unkLabels.length);
        int expectedPositive = (int) (Props.TRAIN_JOINTBAYES_PERCENT_POSITIVE * unkLabels.length * (data.labelIndex.size()));
        int numberToChange = expectedPositive - nPositive;

        log("Before relabeling: " + nPositive + " positive, " + nNegative + " negative, " + nUnknown + " unknown");
        log("Relabeling parameters: " + Props.TRAIN_JOINTBAYES_PERCENT_POSITIVE + " theta, " + unkLabels.length + " groups, " + data.labelIndex().size() + " labels");

        if (numberToChange > 0) {
          log("Target " + expectedPositive + " positive, need to change " + numberToChange + " unknown");
          BoundedPriorityQueue<Triple<Double, Integer,Integer>> priorityQueue = new BoundedPriorityQueue<Triple<Double, Integer, Integer>>(numberToChange,
                  new Comparator<Triple<Double, Integer, Integer>>() {
                    @Override
                    public int compare(Triple<Double, Integer, Integer> o1, Triple<Double, Integer, Integer> o2) {
                      return o1.compareTo(o2);
                    }
                  }
          );

          if (zSingleClassifier != null) {
            // Have single z classifier
            for (int i = 0; i < unkLabels.length; i++) {
              Set<Integer> unkLabelsNotPositive = Sets.diff(unkLabels[i], posLabels[i]);
              int[][] group = rawData[i];
              Counter<Integer> yLogProbs = computeYLogProbs(zSingleClassifier, group, unkLabelsNotPositive);
              for (int yIndex:yLogProbs.keySet()) {
                double yLobProb = yLogProbs.getCount(yIndex);
                priorityQueue.add(Triple.makeTriple(yLobProb, i, yIndex));
              }
            }
          } else {
            // What if we don't have a zSingleClassifier?
            // For each group, infer the y labels for the yDatasets that we are uncertain about
            for(int fold = 0; fold < numberOfFolds; fold ++) {
              int start = foldStart(fold, data.getDataArray().length);
              int end = foldEnd(fold, data.getDataArray().length);
              for (int i = start; i < end; i++) {
                Set<Integer> unkLabelsNotPositive = Sets.diff(unkLabels[i], posLabels[i]);
                int[][] group = rawData[i];
                Counter<Integer> yLogProbs = computeYLogProbs(zClassifiers[fold], group, unkLabelsNotPositive);
                for (int yIndex:yLogProbs.keySet()) {
                  double yLobProb = yLogProbs.getCount(yIndex);
                  priorityQueue.add(Triple.makeTriple(yLobProb, i, yIndex));
                }
              }
            }
          }

          // Make everything in our priority queue positive
          // Put into positive labels, and remove from negative labels
          for (Triple<Double, Integer, Integer> t:priorityQueue) {
            logger.debug("Relabel datum " + t.second + " as belonging to " + data.labelIndex().get(t.third) + ": logProb " + t.first);
            posLabels[t.second].add(t.third);
            negLabels[t.second].remove(t.third);
          }
          nPositive = data.countLabels(posLabels);
          nNegative = data.countLabels(negLabels);
          log("After relabeling: " + nPositive + " positive, " + nNegative + " negative, " + priorityQueue.size() + " changed");
        } else {
          log("No relabeling: target of " + expectedPositive + " reached");
        }
        for (int i = 0; i < unkLabels.length; i++ ) {
          // Label rest of the unknowns as negative
          for (int j:unkLabels[i]) {
            if (!posLabels[i].contains(j)) negLabels[i].add(j);
          }
        }
        nNegative = data.countLabels(negLabels);
        log("After marking unknowns negative: " + nPositive + " positive, " + nNegative + " negative");
        endTrack("Guessing Y labels");
      }

      // for each group, infer the hidden sentence labels z_i,s
      for(int fold = 0; fold < numberOfFolds; fold ++) {
        LinearClassifier<String, String> zClassifier = zClassifiers[fold];
        int start = foldStart(fold, data.getDataArray().length);
        int end = foldEnd(fold, data.getDataArray().length);
        ArrayList<Runnable> threads = new ArrayList<Runnable>();
        @SuppressWarnings("unchecked") Pointer<Triple<int[], Counter<String>[], double[]>>[] confidencePointers = new Pointer[end - start];
        for (int i = start; i < end; i++) {
          confidencePointers[i-start] = new Pointer<Triple<int[], Counter<String>[], double[]>>();
          Runnable r = createZLabeller(zClassifier, yDatasets, data, zLabelsPredictedByZ, zLabels, epoch, i,
                                       confidencePointers[i-start]);
          threads.add(r);
        }
        Redwood.Util.threadAndRun("EPOCH " + epoch + ": Inferring hidden sentence labels Z_i's", threads, numberOfThreads);
        // Compute statistics
        startTrack("Updating training statistics");
        for (int groupI = start; groupI < end; ++groupI) {
          Triple<int[], Counter<String>[], double[]> confidenceForGroup = confidencePointers[groupI - start].dereference().orCrash(); // should be defined after threadAndRun
          int[] originalIndices = confidenceForGroup.first;
          @SuppressWarnings("unchecked") Counter<String>[] reMappedCounter = new Counter[originalIndices.length];
          double[] reMappedConfidences = new double[originalIndices.length];
          for (int sentenceI = 0; sentenceI < originalIndices.length; ++ sentenceI) {
            reMappedCounter[originalIndices[sentenceI]] = confidenceForGroup.second[sentenceI];
            reMappedConfidences[originalIndices[sentenceI]] = confidenceForGroup.third[sentenceI];
          }
          confidences[groupI] = Pair.makePair(reMappedCounter, reMappedConfidences);
        }
        endTrack("Updating training statistics");
      }

      computeConfusionMatrixForCounts("EPOCH " + epoch, zLabels, data.getPositiveLabelsArray());
      computeConfusionMatrixForCounts("(Z ONLY) EPOCH " + epoch, zLabelsPredictedByZ, data.getPositiveLabelsArray());
      computeYScore("EPOCH " + epoch, zLabels, data.getPositiveLabelsArray());
      computeYScore("(Z ONLY) EPOCH " + epoch, zLabelsPredictedByZ, data.getPositiveLabelsArray());

      logger.log("In epoch #" + epoch + " zUpdatesInOneEpoch = " + zUpdatesInOneEpoch);
      if(zUpdatesInOneEpoch.get() == 0){
        logger.log("Stopping training. Did not find any changes in the Z labels!");
        endTrack("E-Step");
        break;
      }

      // update the labels in the z dataset
      Dataset<String, String> zDataset = initializeZDataset(totalSentences, zLabels, data.getDataArray());
      endTrack("E-Step");

      //
      // M step
      //
      startTrack("M-STEP");
      // learn the weights of the sentence-level multi-class classifier
      {
        ArrayList<Runnable> threads = new ArrayList<Runnable>();
        for(int fold = 0; fold < numberOfFolds; fold ++){
          Runnable r = createZClassifierTrainer(zFactory, zDataset, epoch, fold);
          threads.add(r);
        }
        if (partOfEnsemble && Props.TRAIN_JOINTBAYES_MULTITHREAD) {
          // Case: part of ensemble; custom multithreading
          log("EPOCH " + epoch + ": Training Z classifiers");
          ExecutorService threadPool = Executors.newFixedThreadPool(Math.max(1, Execution.threads / Props.TRAIN_ENSEMBLE_NUMCOMPONENTS));
          for( Runnable thread : threads) { threadPool.submit(thread); }
          threadPool.shutdown();
          try {
            threadPool.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS);
          } catch (InterruptedException e) {
            throw new RuntimeException(e);
          }
        } else {
          // Case: use default Redwood multithreading
          Redwood.Util.threadAndRun("EPOCH " + epoch + ": Training Z classifiers", threads, numberOfThreads);
        }
      }

      // learn the weights of each of the top-level two-class classifiers
      if(trainY) {
        ArrayList<Runnable> threads = new ArrayList<Runnable>();
        for (String yLabel : yLabelIndex) {
          Runnable r = createYClassifierTrainer(yFactory, yDatasets, yLabel, epoch);
          threads.add(r);
        }
        if (partOfEnsemble && Props.TRAIN_JOINTBAYES_MULTITHREAD) {
          // Case: part of ensemble; custom multithreading
          log("EPOCH " + epoch + ": Training Y classifiers");
          ExecutorService threadPool = Executors.newFixedThreadPool(Math.max(1, Execution.threads / Props.TRAIN_ENSEMBLE_NUMCOMPONENTS));
          for( Runnable thread : threads) { threadPool.submit(thread); }
          threadPool.shutdown();
          try {
            threadPool.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS);
          } catch (InterruptedException e) {
            throw new RuntimeException(e);
          }
        } else {
          Redwood.Util.threadAndRun("EPOCH " + epoch + ": Training Y classifiers", threads, numberOfThreads);
        }
      }
      makeSingleZClassifier(zDataset, zFactory);

      // save this epoch's model
      String epochPath = makeEpochPath(epoch);
      try {
        if(epochPath != null) {
          save(epochPath);
        }
      } catch (IOException ex) {
        logger.err(RED, "WARNING: could not save model of epoch " + epoch + " to path: " + epochPath);
        logger.err(RED, "Exception message: " + ex.getMessage());
      }

      // clear our y datasets so they can be repopulated on next iteration
      yDatasets = initializeYDatasets();
      endTrack("M-STEP");
    }
    endTrack("EM");

    Dataset<String, String> zDataset = initializeZDataset(totalSentences, zLabels, data.getDataArray());
    makeSingleZClassifier(zDataset, zFactory);

    // Compute Statistics (from most recent E step)
    startTrack("Computing Statistics");
    for (int groupI = 0; groupI < confidences.length; ++groupI) {
      if (confidences != null && confidences[groupI] != null) {
        Pair<Counter<String>[], double[]> groupStats = confidences[groupI];
        Counter<String>[] groupPredictions = groupStats.first;
        double[] groupConfidences = groupStats.second;
        for (int sentenceI = 0; sentenceI < groupPredictions.length; ++sentenceI) {
          TrainingStatistics.SentenceKey key =
              new TrainingStatistics.SentenceKey(data.sentenceGlossKeys[groupI][sentenceI]);
          TrainingStatistics.SentenceStatistics value =
              new TrainingStatistics.SentenceStatistics(Counters.exp(groupPredictions[sentenceI]),
                  Math.exp(groupConfidences[sentenceI]));
          statistics.addInPlace(key, value);
        }
      }
    }
    endTrack("Computing Statistics");
    this.statistics = Maybe.Just(statistics);
    return statistics;
  }

  private int[] randomizeGroup(int[][] group, Maybe<String>[] knownLabels, int randomSeed) {
    assert(group.length == knownLabels.length);
    Random rand = new Random(randomSeed);
    int[] originalIndex = new int[group.length];
    for (int j=0; j<group.length; ++j) { originalIndex[j] = j; }
    if (Props.HACKS_SQUASHRANDOM) { return originalIndex; }
    for(int j = group.length - 1; j > 0; j --){
      int randIndex = rand.nextInt(j);
      // Randomize dataset
      int [] tmp = group[randIndex];
      group[randIndex] = group[j];
      group[j] = tmp;
      // Ensure known labels follow
      Maybe<String> tmp2 = knownLabels[randIndex];
      knownLabels[randIndex] = knownLabels[j];
      knownLabels[j] = tmp2;
      // Mapping to original index
      int tmpIndex = originalIndex[randIndex];
      originalIndex[randIndex] = originalIndex[j];
      originalIndex[j] = tmpIndex;
    }
    return originalIndex;
  }

  Triple<Double, Double, Double> computeYScore(String name, int [][] zLabels, Set<Integer> [] golds) {
    int labelCorrect = 0, labelPredicted = 0, labelTotal = 0;
    int groupCorrect = 0, groupTotal = 0;
    int nilIndex = zLabelIndex.indexOf(RelationMention.UNRELATED);
    if (nilIndex < 0) throw new IllegalStateException("Unknown relation: " + RelationMention.UNRELATED);
    for(int i = 0; i < golds.length; i ++) {
      Set<Integer> pred = new HashSet<Integer>();
      for(int z: zLabels[i])
        if(z != nilIndex) pred.add(z);
      Set<Integer> gold = golds[i];

      labelPredicted += pred.size();
      labelTotal += gold.size();
      for(int z: pred)
        if(gold.contains(z))
          labelCorrect ++;

      groupTotal ++;
      boolean correct = true;
      if(pred.size() != gold.size()) {
        correct = false;
      } else {
        for(int z: pred) {
          if(! gold.contains(z)) {
            correct = false;
            break;
          }
        }
      }
      if(correct) groupCorrect ++;
    }

    double p = (double) labelCorrect / (double) labelPredicted;
    double r = (double) labelCorrect / (double) labelTotal;
    double f1 = p != 0 && r != 0 ? 2*p*r/(p + r) : 0;
    double a = (double) groupCorrect / (double) groupTotal;
    logger.log(BLUE, BOLD, "LABEL SCORE for " + name + ": P " + p + " R " + r + " F1 " + f1);
    logger.log(BLUE, BOLD, "GROUP SCORE for " + name + ": A " + a);
    return Triple.makeTriple(p, r, a);
  }

  /**
   * Get the training accuracy of a saved model. The confusion matrix is also printed.
   * Note that this is a z-label only accuracy -- that is, the Y label is not conditioned on.
   * @param dataset The training dataset
   * @return A triple (precision, recall, accuracy).
   */
  public Triple<Double, Double, Double> trainingAccuracy(KBPDataset<String, String> dataset) {
    int[][][] data = dataset.getDataArray();
    int[][] zLabels = new int[data.length][];

    // Run inference
    for (int exI = 0; exI < data.length; ++exI) {
      zLabels[exI] = new int[data[exI].length];
      List<Datum<String, String>> sentences = dataset.getDatumGroup(exI);
      for (int groupI = 0; groupI < sentences.size(); ++groupI) {
        Collection<String> sentence = sentences.get(groupI).asFeatures();
        Counter<String> labels = classifyLocally(sentence);
        zLabels[exI][groupI] = zLabelIndex.indexOf(Counters.argmax(labels));
      }
    }

    @SuppressWarnings("unchecked") Set<Integer>[] golds = dataset.getPositiveLabelsArray();
    Set<Integer>[] goldsReinterned = ErasureUtils.uncheckedCast(new Set[golds.length]);
    for (int i = 0; i < golds.length; ++i) {
      goldsReinterned[i] = new HashSet<Integer>();
      for (Integer gold : golds[i]) {
        goldsReinterned[i].add(zLabelIndex.indexOf(dataset.labelIndex().get(gold)));
      }
    }
    computeConfusionMatrixForCounts("TRAIN", zLabels, goldsReinterned);
    return computeYScore("Z ONLY", zLabels, goldsReinterned);
  }

  @Deprecated
  void computeConfusionMatrixForCounts_(String name, int [][] zLabels, Set<Integer> [] golds) {
    Counter<Integer> pos = new ClassicCounter<Integer>();
    Counter<Integer> neg = new ClassicCounter<Integer>();
    int nilIndex = zLabelIndex.indexOf(RelationMention.UNRELATED);
    if (nilIndex < 0) throw new IllegalStateException("Unknown relation: " + RelationMention.UNRELATED);
    for(int i = 0; i < zLabels.length; i ++) {
      int [] zs = zLabels[i];
      Counter<Integer> freqs = new ClassicCounter<Integer>();
      for(int z: zs)
        if(z != nilIndex)
          freqs.incrementCount(z);
      Set<Integer> gold = golds[i];
      for(int z: freqs.keySet()) {
        int f = (int) freqs.getCount(z);
        if(gold.contains(z)){
          pos.incrementCount(f);
        } else {
          neg.incrementCount(f);
        }
      }
    }
    logger.log("CONFUSION MATRIX for " + name);
    logger.log("CONFUSION MATRIX POS: " + pos);
    logger.log("CONFUSION MATRIX NEG: " + neg);
  }

  void computeConfusionMatrixForCounts(String name, int [][] zLabels, Set<Integer> [] golds) {
    int labelCount = zLabelIndex.size();
    Double[][] confusionMatrix = new Double[labelCount][labelCount];
    for( int z = 0; z < labelCount; z++ )
      for( int z_ = 0; z_ < labelCount; z_++ )
        confusionMatrix[z][z_] = 0.0;

    int nilIndex = zLabelIndex.indexOf(RelationMention.UNRELATED);
    if (nilIndex < 0) throw new IllegalStateException("Unknown relation: " + RelationMention.UNRELATED);

    // For each group add labels to the confusion matrix.
    for(int i = 0; i < zLabels.length; i ++) {
      Set<Integer> pred = new HashSet<Integer>();
      for( int z : zLabels[i] ) pred.add(z); // Initialize the set

      Set<Integer> gold = new HashSet<Integer>(golds[i]);
      // For every label we have right, add to the conf. matrix and
      // remove from the set.
      Iterator<Integer> it = pred.iterator();
      while( it.hasNext() ) {
        int z = it.next();
        if( gold.contains( z ) ) {
          confusionMatrix[z][z] += 1;
          gold.remove(z);
          it.remove();
        }
      }

      // Now, split the weight amongst the remaining nodes.
      for(int z: pred) {
        for( int z_ : gold ) {
          confusionMatrix[z_][z] += 1.0/gold.size();
        }
      }
    }
    if (!partOfEnsemble || !Props.TRAIN_JOINTBAYES_MULTITHREAD) { startTrack( "confusion matrix (" + name + ")" ); }
    logger.log(
        StringUtils.makeTextTable(
            confusionMatrix,
            zLabelIndex.objectsList().toArray(),
            zLabelIndex.objectsList().toArray(),
            0, 1, true));
    if (!partOfEnsemble || !Props.TRAIN_JOINTBAYES_MULTITHREAD) { Redwood.endTrack( "confusion matrix (" + name + ")" ); }
  }

  void printGroup(int [] zs, Set<Integer> ys) {
    System.err.print("ZS:");
    for(int z: zs) {
      String zl = zLabelIndex.get(z);
      System.err.print(" " + zl);
    }
    logger.log();
    Set<String> missed = new HashSet<String>();
    System.err.print("YS:");
    for(Integer y: ys) {
      String yl = yLabelIndex.get(y);
      System.err.print(" " + yl);
      boolean found = false;
      for(int z: zs) {
        String zl = zLabelIndex.get(z);
        if(zl.equals(yl)) {
          found = true;
          break;
        }
      }
      if(! found) {
        missed.add(yl);
      }
    }
    logger.log();
    if(missed.size() > 0) {
      System.err.print("MISSED " + missed.size() + ":");
      for(String m: missed) {
        System.err.print(" " + m);
      }
    }
    logger.log();
    logger.log("END GROUP");
  }

  private void addYDatum(
      RVFDataset<String, String> yDataset,
      String yLabel,
      int [] zLabels,
      Counter<String> [] zLogProbs,
      boolean isPositive) {
    Counter<String> yFeats = extractYFeatures(yLabel, zLabels);
    //if(yFeats.size() > 0) logger.log("YFEATS" + (isPositive ? " POSITIVE " : " NEGATIVE ") + yLabel + " " + yFeats);
    RVFDatum<String, String> datum =
      new RVFDatum<String, String>(yFeats,
          (isPositive ? yLabel : RelationMention.UNRELATED));
    yDataset.add(datum);
  }

  private String makeEpochPath(int epoch) {
    String epochPath = null;
    if(epoch < numberOfTrainEpochs && serializedModelPath != null) {
      if(serializedModelPath.endsWith(".ser")) {
        epochPath =
          serializedModelPath.substring(0, serializedModelPath.length() - ".ser".length()) +
          "_EPOCH" + epoch + ".ser";
      } else {
        epochPath = serializedModelPath + "_EPOCH" + epoch;
      }
    }
    return epochPath;
  }

  private void makeSingleZClassifier(
      Dataset<String, String> zDataset,
      LinearClassifierFactory<String, String> zFactory) {
    if(localClassificationMode == LOCAL_CLASSIFICATION_MODE.SINGLE_MODEL) {
      // train a single Z classifier over the entire data
      logger.log("Training the final Z classifier...");
      zSingleClassifier = zFactory.trainClassifierWithInitialWeights(zDataset, zSingleClassifier);
    } else {
      zSingleClassifier = null;
    }
  }

  private int[] flatten(int[][] array, int size) {
    int[] result = new int[size];
    int count = 0;
    for (int[] row : array) {
      for (int element : row)
        result[count++] = element;
    }
    return result;
  }

  public static abstract class LocalFilter {
    public abstract boolean filterZ(int [][] data, Set<Integer> posLabels);
    public boolean filterY(int [][] data, Set<Integer> posLabels) { return true; }
  }

  public static class LargeFilter extends LocalFilter {
    final int threshold;
    public LargeFilter(int thresh) {
      this.threshold = thresh;
    }
    @Override
    public boolean filterZ(int[][] data, Set<Integer> posLabels) {
      if(data.length > threshold) return false;
      return true;
    }
    @Override
    public boolean filterY(int[][] data, Set<Integer> posLabels) {
      if(data.length > threshold) return false;
      return true;
    }
  }

  public static class AllFilter extends LocalFilter {
    @Override
    public boolean filterZ(int[][] data, Set<Integer> posLabels) {
      return true;
    }
  }

  public static class SingleFilter extends LocalFilter {
    @Override
    public boolean filterZ(int[][] data, Set<Integer> posLabels) {
      if(posLabels.size() <= 1) return true;
      return false;
    }
  }

  public static class RedundancyFilter extends LocalFilter {
    @Override
    public boolean filterZ(int[][] data, Set<Integer> posLabels) {
      if(posLabels.size() <= 1 && data.length > 1) return true;
      return false;
    }
  }

  private static Dataset<String, String> makeLocalData(
      int [][][] dataArray,
      Set<Integer> [] posLabels,
      Set<Integer> [] negLabels,
      Index<String> labelIndex,
      Index<String> featureIndex,
      LocalFilter f,
      int fold) {
    // Detect the size of the dataset for the local classifier
    int flatSize = 0, posGroups = 0, negGroups = 0;
    for(int i = 0; i < dataArray.length; i ++) {
      if(! f.filterZ(dataArray[i], posLabels[i])) continue;
      if(posLabels[i].size() == 0 && negLabels[i].size() > 0) {
        // negative example
        flatSize += dataArray[i].length;
        negGroups ++;
      } else {
        // 1+ positive labels
        flatSize += dataArray[i].length * posLabels[i].size();
        posGroups ++;
      }
    }
    logger.log("Explored " + posGroups + " positive groups and " + negGroups + " negative groups, yielding " + flatSize + " flat/local datums.");

    //
    // Construct the flat local classifier
    //
    int [][] localTrainData = new int[flatSize][];
    int [] localTrainLabels = new int[flatSize];
    float [] weights = new float[flatSize];
    int offset = 0, posCount = 0;
    Set<Integer> myNegLabels = new HashSet<Integer>();
    int nilIndex = labelIndex.indexOf(RelationMention.UNRELATED);
    if (nilIndex < 0) throw new IllegalStateException("Unknown relation: " + RelationMention.UNRELATED);
    myNegLabels.add(nilIndex);
    for(int i = 0; i < dataArray.length; i ++) {
      if(! f.filterZ(dataArray[i], posLabels[i])) { continue; }
      int [][] group = dataArray[i];
      Set<Integer> labels = posLabels[i];
      if(labels.size() == 0) labels = myNegLabels;
      float weight = (float) 1.0 / (float) labels.size();
      for(Integer label: labels) {
        for(int j = 0; j < group.length; j ++){
          localTrainData[offset] = group[j];
          localTrainLabels[offset] = label;
          weights[offset] = weight;
          if(label != nilIndex) posCount ++;
          offset ++;
          if(offset >= flatSize) break;
        }
        if(offset >= flatSize) break;
      }
      if(offset >= flatSize) break;
    }

    Dataset<String, String> dataset = new WeightedDataset<String, String>(
        labelIndex,
        localTrainLabels,
        featureIndex,
        localTrainData, localTrainData.length,
        weights);
    logger.log("Fold #" + fold + ": Constructed a dataset with " + localTrainData.length +
        " datums, out of which " + posCount + " are positive.");
    if(posCount == 0) throw new RuntimeException("ERROR: cannot handle a dataset with 0 positive examples!");

    return dataset;
  }

  private int [] makeTrainLabelArrayForFold(int [] labelArray, int fold) {
    int start = foldStart(fold, labelArray.length);
    int end = foldEnd(fold, labelArray.length);
    int [] train = new int[labelArray.length - end + start];
    int trainOffset = 0;
    for(int i = 0; i < start; i ++){
      train[trainOffset] = labelArray[i];
      trainOffset ++;
    }
    for(int i = end; i < labelArray.length; i ++){
      train[trainOffset] = labelArray[i];
      trainOffset ++;
    }
    return train;
  }

  private int [][] makeTrainDataArrayForFold(int [][] dataArray, int fold) {
    int start = foldStart(fold, dataArray.length);
    int end = foldEnd(fold, dataArray.length);
    int [][] train = new int[dataArray.length - end + start][];
    int trainOffset = 0;
    for(int i = 0; i < start; i ++){
      train[trainOffset] = dataArray[i];
      trainOffset ++;
    }
    for(int i = end; i < dataArray.length; i ++){
      train[trainOffset] = dataArray[i];
      trainOffset ++;
    }
    return train;
  }

  private Pair<int [][][], int [][][]> makeDataArraysForFold(int [][][] dataArray, int fold) {
    int start = foldStart(fold, dataArray.length);
    int end = foldEnd(fold, dataArray.length);
    int [][][] train = new int[dataArray.length - end + start][][];
    int [][][] test = new int[end - start][][];
    int trainOffset = 0, testOffset = 0;
    for(int i = 0; i < dataArray.length; i ++){
      if(i < start){
        train[trainOffset] = dataArray[i];
        trainOffset ++;
      } else if(i < end) {
        test[testOffset] = dataArray[i];
        testOffset ++;
      } else {
        train[trainOffset] = dataArray[i];
        trainOffset ++;
      }
    }
    return new Pair<int[][][], int[][][]>(train, test);
  }

  @SuppressWarnings("unchecked")
  private Pair<Set<Integer> [], Set<Integer> []> makeLabelSetsForFold(Set<Integer> [] labelSet, int fold) {
    int start = foldStart(fold, labelSet.length);
    int end = foldEnd(fold, labelSet.length);
    Set<Integer>[] train = new HashSet[labelSet.length - end + start];
    Set<Integer>[] test = new HashSet[end - start];
    int trainOffset = 0, testOffset = 0;
    for(int i = 0; i < labelSet.length; i ++){
      if(i < start){
        train[trainOffset] = labelSet[i];
        trainOffset ++;
      } else if(i < end) {
        test[testOffset] = labelSet[i];
        testOffset ++;
      } else {
        train[trainOffset] = labelSet[i];
        trainOffset ++;
      }
    }
    return new Pair<Set<Integer>[], Set<Integer>[]>(train, test);
  }

  private LinearClassifierFactory<String, String> getZClassifierFactory() {
    LinearClassifierFactory<String, String> factory =
            new LinearClassifierFactory<String, String>(1e-4, false, zSigma);
    switch (zClassifierMinimizerType) {
      case SGD: factory.useInPlaceStochasticGradientDescent(75, 1000, zSigma); break;
      case SGDTOQN: factory.useHybridMinimizerWithInPlaceSGD(75, 1000, zSigma); break;
      case QN: break; // default
    }
    return factory;
  }

  private Runnable createLocalZClassifierInitializer(final LinearClassifier<String, String> [] localClassifiers,
                                                     KBPDataset<String, String> data,
                                                     final Index<String> featureIndex,
                                                     final Index<String> labelIndex,
                                                     final int fold) {
    final int[][][] dataArray = data.getDataArray();
    final Set<Integer> [] positiveLabelsArray = data.getPositiveLabelsArray();
    final Set<Integer> [] negativeLabelsArray = data.getNegativeLabelsArray();
    return new Runnable() {
      public void run() {
        String title = "Initializing local Z classifier for fold #" + fold;
        if (!partOfEnsemble || !Props.TRAIN_JOINTBAYES_MULTITHREAD) { startTrack(title); }
        logger.log("Constructing dataset for the local model in fold #" + fold + "...");
        Pair<int [][][], int [][][]> dataArrays = makeDataArraysForFold(dataArray, fold);
        Pair<Set<Integer> [], Set<Integer> []> posLabelSets =
                makeLabelSetsForFold(positiveLabelsArray, fold);
        Pair<Set<Integer> [], Set<Integer> []> negLabelSets =
                makeLabelSetsForFold(negativeLabelsArray, fold);

        int [][][] trainDataArray = dataArrays.first();
        Set<Integer> [] trainPosLabels = posLabelSets.first();
        Set<Integer> [] trainNegLabels = negLabelSets.first();
        int [][][] testDataArray = dataArrays.second();
        Set<Integer> [] testPosLabels = posLabelSets.second();
        Set<Integer> [] testNegLabels = negLabelSets.second();

        //
        // Construct the flat local classifier
        //
        Dataset<String, String> dataset =
                makeLocalData(trainDataArray, trainPosLabels, trainNegLabels, labelIndex, featureIndex, localDataFilter, fold);

        //
        // Train local classifier
        //
        logger.log("Fold #" + fold + ": Training local model...");
        LinearClassifierFactory<String, String> factory = getZClassifierFactory();
        LinearClassifier<String, String> localClassifier = factory.trainClassifier(dataset);
        logger.log("Fold #" + fold + ": Training of the local classifier completed.");

        //
        // Evaluate the classifier on the multidataset - Currently only checks positive labels
        //
        int nilIndex = labelIndex.indexOf(RelationMention.UNRELATED);
        if (nilIndex < 0) throw new IllegalStateException("Unknown relation: " + RelationMention.UNRELATED);
        logger.log("Fold #" + fold + ": Evaluating the local classifier on the hierarchical dataset...");
        int total = 0, predicted = 0, correct = 0;
        for(int i = 0; i < testDataArray.length; i ++){
          int [][] group = testDataArray[i];
          Set<Integer> gold = testPosLabels[i];
          Set<Integer> pred = new HashSet<Integer>();
          for(int j = 0; j < group.length; j ++){
            int [] datum = group[j];
            Counter<String> scores = localClassifier.scoresOf(datum);
            List<Pair<String, Double>> sortedScores = sortPredictions(scores);
            int sys = labelIndex.indexOf(sortedScores.get(0).first());
            if (sys < 0) throw new IllegalStateException("Unknown relation: " + sortedScores.get(0).first());
            if(sys != nilIndex) pred.add(sys);
          }
          total += gold.size();
          predicted += pred.size();
          for(Integer pv: pred) {
            if(gold.contains(pv)) correct ++;
          }
        }
        double p = (double) correct / (double) predicted;
        double r = (double) correct / (double) total;
        double f1 = (p != 0 && r != 0 ? 2*p*r/(p+r) : 0);
        logger.log("Fold #" + fold + ": Training score on the hierarchical dataset: P " + p + " R " + r + " F1 " + f1);

        logger.log("Fold #" + fold + ": Created the Z classifier with " + labelIndex.size() +
                " labels and " + featureIndex.size() + " features.");
        synchronized (lock) {
          localClassifiers[fold] = localClassifier;
        }
        if (!partOfEnsemble || !Props.TRAIN_JOINTBAYES_MULTITHREAD) { Redwood.endTrack(title); }
      }
    };
  }

  @SuppressWarnings("unchecked")
  private LinearClassifier<String, String> [] initializeZClassifierLocally(
      KBPDataset<String, String> data,
      Index<String> featureIndex,
      Index<String> labelIndex) {

    LinearClassifier<String, String> [] localClassifiers = new LinearClassifier[numberOfFolds];

    // construct the initial model for each fold
    ArrayList<Runnable> threads = new ArrayList<Runnable>();
    for(int fold = 0; fold < numberOfFolds; fold ++){
      Runnable r = createLocalZClassifierInitializer(localClassifiers, data, featureIndex, labelIndex, fold);
      threads.add(r);
    }
    Redwood.Util.threadAndRun("Initialize local Z classifiers", threads, numberOfThreads);
    return localClassifiers;
  }

  @SuppressWarnings("unchecked")
  private void loadInitialModels(String path) throws IOException, ClassNotFoundException {
    InputStream is = new FileInputStream(path);
    ObjectInputStream in = new ObjectInputStream(is);

    if (featureIndex != null) { throw new IllegalStateException("Loading over a trained model!"); }
    featureIndex = ErasureUtils.uncheckedCast(in.readObject());
    zLabelIndex = ErasureUtils.uncheckedCast(in.readObject());
    yLabelIndex = ErasureUtils.uncheckedCast(in.readObject());

    numberOfFolds = in.readInt();
    zClassifiers = new LinearClassifier[numberOfFolds];
    for(int i = 0; i < numberOfFolds; i ++){
      LinearClassifier<String, String> classifier =
        ErasureUtils.uncheckedCast(in.readObject());
      zClassifiers[i] = classifier;
      logger.log("Loaded Z classifier for fold #" + i + ": " + zClassifiers[i]);
    }

    int numLabels = in.readInt();
    yClassifiers = new HashMap<String, LinearClassifier<String, String>>();
    for (int i = 0; i < numLabels; i++) {
      String yLabel = ErasureUtils.uncheckedCast(in.readObject());
      LinearClassifier<String, String> classifier =
        ErasureUtils.uncheckedCast(in.readObject());
      yClassifiers.put(yLabel, classifier);
    }

    in.close();
  }

  private void saveInitialModels(String path) throws IOException {
    ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(path));
    out.writeObject(featureIndex);
    out.writeObject(zLabelIndex);
    out.writeObject(yLabelIndex);
    out.writeInt(zClassifiers.length);
    for(int i = 0; i < zClassifiers.length; i ++)
      out.writeObject(zClassifiers[i]);
    out.writeInt(yClassifiers.keySet().size());
    for (String yLabel : yClassifiers.keySet()) {
      out.writeObject(yLabel);
      out.writeObject(yClassifiers.get(yLabel));
    }
    out.close();
  }

  private Map<String, LinearClassifier<String, String>> initializeYClassifiersWithAtLeastOnce(Index<String> labelIndex) {
    Map<String, LinearClassifier<String, String>> classifiers =
      new HashMap<String, LinearClassifier<String, String>>();
    for (String yLabel : labelIndex) {
      Index<String> yFeatureIndex = new HashIndex<String>();
      yFeatureIndex.addAll(Y_FEATURES_FOR_INITIAL_MODEL);

      Index<String> thisYLabelIndex = new HashIndex<String>();
      thisYLabelIndex.add(yLabel);
      thisYLabelIndex.add(RelationMention.UNRELATED);

      double[][] weights = initializeWeights(yFeatureIndex.size(), thisYLabelIndex.size());
      setYWeightsForAtLeastOnce(weights, yFeatureIndex, thisYLabelIndex);
      classifiers.put(yLabel, new LinearClassifier<String, String>(weights, yFeatureIndex, thisYLabelIndex));
      logger.log("Created the classifier for Y=" + yLabel + " with " + yFeatureIndex.size() + " features");
    }
    return classifiers;
  }

  private static final double BIG_WEIGHT = +10;

  private static void setYWeightsForAtLeastOnce(double[][] weights,
      Index<String> featureIndex,
      Index<String> labelIndex) {
    int posLabel = -1, negLabel = -1;
    for(String l: labelIndex) {
      if(l.equalsIgnoreCase(RelationMention.UNRELATED)) {
        negLabel = labelIndex.indexOf(l);
        if (negLabel < 0) throw new IllegalStateException("Unknown relation: " + l);
      } else {
        debug("posLabel = " + l);
        posLabel = labelIndex.indexOf(l);
        if (posLabel < 0) throw new IllegalStateException("Unknown relation: " + l);
      }
    }
    assert(posLabel != -1);
    assert(negLabel != -1);

    int atLeastOnceIndex = featureIndex.indexOf(ATLEASTONCE_FEAT);
    int noneIndex = featureIndex.indexOf(NONE_FEAT);
    if (featureIndex.indexOf(ATLEASTONCE_FEAT) >= 0) {
      weights[featureIndex.indexOf(ATLEASTONCE_FEAT)][posLabel] = BIG_WEIGHT;
    }
    if (featureIndex.indexOf(SIGMOID_FEAT) >= 0) {
      weights[featureIndex.indexOf(SIGMOID_FEAT)][posLabel] = BIG_WEIGHT;
    }
    weights[noneIndex][negLabel] = BIG_WEIGHT;
    debug("posLabel = " + posLabel + ", negLabel = " + negLabel + ", atLeastOnceIndex = " + atLeastOnceIndex);
  }

  private static double[][] initializeWeights(int numFeatures, int numLabels) {
    double[][] weights = new double[numFeatures][numLabels];
    for (double[] row : weights)
      Arrays.fill(row, 0.0);

    return weights;
  }

  private Dataset<String, String> initializeZDataset(int totalSentences, int[][] zLabels, int[][][] data) {
    int[][] flatData = new int[totalSentences][];
    int count = 0;
    for (int i = 0; i < data.length; i++) {
      for (int s = 0; s < data[i].length; s++)
        flatData[count++] = data[i][s];
    }

    int[] flatZLabels = flatten(zLabels, totalSentences);
    logger.log("Created the Z dataset with " + flatZLabels.length + " datums.");

    return new Dataset<String, String>(zLabelIndex, flatZLabels, featureIndex, flatData);
  }

  private Map<String, RVFDataset<String, String>> initializeYDatasets() {
    Map<String, RVFDataset<String, String>> result = new HashMap<String, RVFDataset<String, String>>();
    for (String yLabel : yLabelIndex.objectsList())
      result.put(yLabel, new RVFDataset<String, String>());
    return result;
  }

  protected Counter<Integer> computeYLogProbs(LinearClassifier<String, String> zClassifier, int[][] group, Set<Integer> indices) {
    int[] zLabelsPredictedByZ = new int[group.length];
    predictZLabels(group, zLabelsPredictedByZ, zClassifier);
    return computeYLogProbs(zLabelsPredictedByZ, indices);
    // TODO: Sum over all possible z'is (logprob of z + log prob of y)
    //Counter<Integer> yLogProbs = new ClassicCounter<Integer>();
    //return yLogProbs;
  }

  // Given predicted z's, use the y classifier to compute the probabilities of the given y labels (from indices)
  protected Counter<Integer> computeYLogProbs(int[] zLabels, Set<Integer> indices) {
    Counter<Integer> yLogProbs = new ClassicCounter<Integer>();
    for (int y : indices) {
      String yLabel = yLabelIndex.get(y);
      Datum<String, String> yDatum =
              new RVFDatum<String, String>(extractYFeatures(yLabel, zLabels), "");
      Counter<String> yProbabilities = yClassifiers.get(yLabel).logProbabilityOf(yDatum);
      double v = yProbabilities.getCount(yLabel);
      yLogProbs.setCount(y, v);
    }
    return yLogProbs;
  }

  private void predictZLabels(int [][] group,
      int[] zLabels,
      LinearClassifier<String, String> zClassifier) {
    for (int s = 0; s < group.length; s++) {
      Counter<String> probs = zClassifier.logProbabilityOf(group[s]);
      zLabels[s] = zLabelIndex.indexOf(Counters.argmax(probs));
    }
  }

  private void computeZLogProbs(int[][] group,
      Counter<String> [] zLogProbs,
      Maybe<String>[] knownLabels,
      LinearClassifier<String, String> zClassifier,
      int epoch) {
    for (int s = 0; s < group.length; s ++) {
      if (knownLabels[s].isDefined()) {
        zLogProbs[s] = new ClassicCounter<String>();
        zLogProbs[s].setCount(knownLabels[s].get(), 0.0);
      } else {
        zLogProbs[s] = zClassifier.logProbabilityOf(group[s]);
      }
    }
  }

  /** updates the zLabels array with new predicted z labels */
  private double[] inferZLabelsStable(int[][] group,
      Set<Integer> positiveLabels,
      Set<Integer> negativeLabels,
      int[] zLabels,
      Maybe<String>[] fixedZ,
      Counter<String> [] jointZProbs,
      LinearClassifier<String, String> zClassifier,
      int epoch) {
    assert fixedZ.length == zLabels.length;
    assert group.length == zLabels.length;
    assert jointZProbs.length == zLabels.length;

    boolean showProbs = false;
    boolean verbose = false;

    if(verbose) {
      logger.log("inferZLabels: ");
      if(positiveLabels.size() > 1) logger.log("MULTI RELATION");
      else if(positiveLabels.size() == 1) logger.log("SINGLE RELATION");
      else logger.log("NIL RELATION");
      logger.log("positiveLabels: " + positiveLabels);
      logger.log("negativeLabels: " + negativeLabels);
      System.err.print("Current zLabels:");
      for(int i = 0; i < zLabels.length; i ++) System.err.print(" " + zLabels[i]);
      logger.log();
    }

    // compute the Z probabilities; these do not change
    Counter<String>[] zLogProbs = ErasureUtils.uncheckedCast(new Counter[group.length]);
    computeZLogProbs(group, zLogProbs, fixedZ, zClassifier, epoch);

    double[] maxLogProb = new double[group.length];
    for (int s = 0; s < group.length; s++) {
      if (fixedZ[s].isDefined()) { // can't flip this candidate
        zLabels[s] = zLabelIndex.indexOf(fixedZ[s].get());
        if (zLabels[s] < 0) throw new IllegalStateException("Unknown relation: " + fixedZ[s].get());
        maxLogProb[s] = 0.0; // log(1.0)
      }
      maxLogProb[s] = Double.NEGATIVE_INFINITY;
      int bestLabel = -1;

      Counter<String> zLogProbabilities = zLogProbs[s];
      assert Math.abs(Counters.exp(zLogProbabilities).totalCount() - 1.0) < 1e-5;
      Counter<String> jointProbabilities = new ClassicCounter<String>();

      int origZLabel = zLabels[s];
      for (String candidate : zLogProbabilities.keySet()) {
        int candidateIndex = zLabelIndex.indexOf(candidate);

        // start with z probability
        if(showProbs) logger.log("\tProbabilities for z[" + s + "]:");
        double logProb = zLogProbabilities.getCount(candidate);
        // Enforce P(z | x) if we have z labeled
        if (fixedZ[s].isDefined()) {
          if (candidate.equals(fixedZ[s].get())) logProb = Math.log(0.8);  // TODO(gabor) turker error is 0.8. Live with it.
          else logProb = Math.log(0.2 / ((double) (zLabelIndex.size() - 1)));
        }
        zLabels[s] = candidateIndex;
        if(showProbs) logger.log("\t\tlocal (" + zLabels[s] + ") = " + logProb);

        // add the y probabilities
        for (int y : positiveLabels) {
          String yLabel = yLabelIndex.get(y);
          Datum<String, String> yDatum =
            new RVFDatum<String, String>(extractYFeatures(yLabel, zLabels), "");
          Counter<String> yProbabilities = yClassifiers.get(yLabel).logProbabilityOf(yDatum);
          double v = yProbabilities.getCount(yLabel);
          if(showProbs) logger.log("\t\t\ty+ (" + y + ") = " + v);
          logProb += v;
        }
        for (int y : negativeLabels) {
          String yLabel = yLabelIndex.get(y);
          Datum<String, String> yDatum =
            new RVFDatum<String, String>(extractYFeatures(yLabel, zLabels), "");
          Counter<String> yProbabilities = yClassifiers.get(yLabel).logProbabilityOf(yDatum);
          double v = yProbabilities.getCount(RelationMention.UNRELATED);
          if(showProbs) logger.log("\t\t\ty- (" + y + ") = " + v);
          logProb += v;
        }

        if(showProbs) logger.log("\t\ttotal (" + zLabels[s] + ") = " + logProb);
        jointProbabilities.setCount(candidate, logProb);

        // update the current maximum
        if (logProb > maxLogProb[s]) {
          maxLogProb[s] = logProb;
          bestLabel = zLabels[s];
        }
      }

      if(bestLabel != -1 && bestLabel != origZLabel) {
        // found the best flip for this mention
        if(verbose) logger.log("\tNEW zLabels[" + s + "] = " + bestLabel);
        zLabels[s] = bestLabel;
        zUpdatesInOneEpoch.getAndIncrement();
      } else {
        // nothing good found
        zLabels[s] = origZLabel;
      }

      Counters.logNormalizeInPlace(jointProbabilities);
      jointZProbs[s] = jointProbabilities;
    } // end scan for group
    return maxLogProb;
  }

  /** updates the zLabels array with new predicted z labels */
  private double[] inferZLabels(int[][] group,
      Set<Integer> positiveLabels,
      Set<Integer> negativeLabels,
      int[] zLabels,
      Maybe<String>[] fixedZ,
      Counter<String> [] jointZProbs,
      LinearClassifier<String, String> zClassifier,
      int epoch) {
    assert fixedZ.length == zLabels.length;
    assert group.length == zLabels.length;
    assert jointZProbs.length == zLabels.length;

    boolean showProbs = false;
    boolean verbose = false;

    if(verbose) {
      System.err.print("inferZLabels: ");
      if(positiveLabels.size() > 1) logger.log("MULTI RELATION");
      else if(positiveLabels.size() == 1) logger.log("SINGLE RELATION");
      else logger.log("NIL RELATION");
      logger.log("positiveLabels: " + positiveLabels);
      logger.log("negativeLabels: " + negativeLabels);
      System.err.print("Current zLabels:");
      for(int i = 0; i < zLabels.length; i ++) System.err.print(" " + zLabels[i]);
      logger.log();
    }

    // compute the Z probabilities; these do not change
    computeZLogProbs(group, jointZProbs, fixedZ, zClassifier, epoch);

    // hill climbing until convergence
    // this is needed to guarantee labels that are consistent with "at least once"
    Set<Integer> flipped = new HashSet<Integer>();
    double[] maxProb = new double[group.length];
    while(true){
      double maxProbGlobal = Double.NEGATIVE_INFINITY;
      int bestLabelGlobal = -1;
      int bestSentence = -1;

      for (int s = 0; s < group.length; s++) {
        if(flipped.contains(s)) continue;
        if (fixedZ[s].isDefined()) { // can't flip this candidate
          zLabels[s] = zLabelIndex.indexOf(fixedZ[s].get());
          maxProb[s] = 0.0; // Log(1.0)
        }
        maxProb[s] = Double.NEGATIVE_INFINITY;
        int bestLabel = -1;

        Counter<String> zProbabilities = jointZProbs[s];
        List<String> sortedZLabels = Utils.sortRelationsByPrior(zProbabilities.keySet());
        Counter<String> jointProbabilities = new ClassicCounter<String>();

        int oldZLabel = zLabels[s];
        for (String candidate : sortedZLabels) {
          // start with z probability
          if(showProbs) logger.log("\tProbabilities for z[" + s + "]:");
          double prob = zProbabilities.getCount(candidate);
          if (fixedZ[s].isDefined()) {
            if (candidate.equals(fixedZ[s].get())) prob = Math.log(0.8);  // TODO(gabor) turker error is 0.8. Live with it.
            else prob = Math.log(0.2 / ((double) (zLabelIndex.size() - 1)));
          }
          zLabels[s] = zLabelIndex.indexOf(candidate);
          if(showProbs) logger.log("\t\tlocal (" + zLabels[s] + ") = " + prob);

          // add the y probabilities
          for (int y : positiveLabels) {
            String yLabel = yLabelIndex.get(y);
            Datum<String, String> yDatum =
              new RVFDatum<String, String>(extractYFeatures(yLabel, zLabels), "");
            Counter<String> yProbabilities = yClassifiers.get(yLabel).logProbabilityOf(yDatum);
            double v = yProbabilities.getCount(yLabel);
            if(showProbs) logger.log("\t\t\ty+ (" + y + ") = " + v);
            prob += v;
          }
          for (int y : negativeLabels) {
            String yLabel = yLabelIndex.get(y);
            Datum<String, String> yDatum =
              new RVFDatum<String, String>(extractYFeatures(yLabel, zLabels), "");
            Counter<String> yProbabilities = yClassifiers.get(yLabel).logProbabilityOf(yDatum);
            double v = yProbabilities.getCount(RelationMention.UNRELATED);
            if(showProbs) logger.log("\t\t\ty- (" + y + ") = " + v);
            prob += v;
          }

          if(showProbs) logger.log("\t\ttotal (" + zLabels[s] + ") = " + prob);
          jointProbabilities.setCount(candidate, prob);

          // update the current maximum
          if (prob > maxProb[s]) {
            maxProb[s] = prob;
            bestLabel = zLabels[s];
          }
        }
        //reset; we flip only the global best
        zLabels[s] = oldZLabel;

        // if we end up with a uniform distribution it means we did not predict anything. do not update
        if(bestLabel != -1 && bestLabel != zLabels[s] &&
           ! uniformDistribution(jointProbabilities) &&
           maxProb[s] > maxProbGlobal) {
          // found the best flip so far
          maxProbGlobal = maxProb[s];
          bestLabelGlobal = bestLabel;
          bestSentence = s;
        }

        Counters.logNormalizeInPlace(jointProbabilities);
        jointZProbs[s] = jointProbabilities;
      } // end this group scan

      // no changes found; we converged
      if(bestLabelGlobal == -1) break;

      // flip one Z
      assert(bestSentence != -1);
      zLabels[bestSentence] = bestLabelGlobal;
      zUpdatesInOneEpoch.getAndIncrement();
      flipped.add(bestSentence);
      if(verbose) logger.log("\tNEW zLabels[" + bestSentence + "] = " + zLabels[bestSentence]);

    } // end convergence loop

    // check for flips that didn't happen
    boolean missedY = false;
    for(Integer y: positiveLabels) {
      boolean found = false;
      for(int i = 0; i < zLabels.length; i ++){
        if(zLabels[i] == y){
          found = true;
          break;
        }
      }
      if(! found) {
        missedY = true;
        break;
      }
    }
    if(verbose && missedY) {
      if(zLabels.length < positiveLabels.size()) {
        logger.log("FOUND MISSED Y, smaller Z");
      } else {
        logger.log("FOUND MISSED Y, larger Z");
      }
    }

    return maxProb;
  }

  private static boolean uniformDistribution(Counter<String> probs) {
    List<String> keys = new ArrayList<String>(probs.keySet());
    if(keys.size() < 2) return false;
    double p = probs.getCount(keys.get(0));
    for(int i = 1; i < keys.size(); i ++){
      if(p != probs.getCount(keys.get(i))){
        return false;
      }
    }
    return true;
  }

  // TODO(gabor) this should be the only method, once JointBayesEqualityCheckingRelationExtractor is fixed
  private Counter<String> extractYFeatures(
      String yLabel,
      int[] zLabels) {
    return extractYFeatures(yLabel, zLabels, null);
  }

  private Counter<String> extractYFeatures(
      String yLabel,
      int[] zLabels,
      Counter<String> [] zLogProbs) {
    assert zLogProbs == null;
    int count = 0;
    String[] others = new String[zLabels.length];
    for (int s = 0; s < zLabels.length; s ++) {
      String zString = zLabelIndex.get(zLabels[s]);
      if (zString.equals(yLabel)) {
        count ++;
      } else if(! zString.equals(RelationMention.UNRELATED)) {
        others[s] = zString;
      }
    }

    Counter<String> features = new ClassicCounter<String>();

    // no Z proposed this label
    if(count == 0){
      features.setCount(NONE_FEAT, 1.0);
    }

    // was this label proposed by at least a Z?
    if (Props.TRAIN_JOINTBAYES_YFEATURES.contains(Props.Y_FEATURE_CLASS.ATLEAST_ONCE) &&
        count > 0) {
      features.setCount(ATLEASTONCE_FEAT, 1.0);
    }

    // label dependencies
    if (Props.TRAIN_JOINTBAYES_YFEATURES.contains(Props.Y_FEATURE_CLASS.COOC) &&
        count > 0) {
      for(String z: others) {
        if (z != null) {
          String f = makeCoocurrenceFeature(yLabel, z);
          features.setCount(f, 1.0);
        }
      }
    }

    // is the only prediction
    if (Props.TRAIN_JOINTBAYES_YFEATURES.contains(Props.Y_FEATURE_CLASS.UNIQUE) &&
        count > 0) {
      boolean unique = true;
      for(String z: others) { unique &= (z == null); }
      if (unique) {
        features.setCount(UNIQUE_FEAT, 1.0);
      }
    }

    // At least n relations are true
    if (Props.TRAIN_JOINTBAYES_YFEATURES.contains(Props.Y_FEATURE_CLASS.ATLEAST_N) &&
        count > 0) {
      features.setCount("atleast_" + count, 1.0);
    }

    // A smoothed version of the percentage of z labels predicted as this relation.
    // e.g., plot in Google:  plot 1 / (1 + e^(-10(x - 1/3)))  from 0 to 1
    if (Props.TRAIN_JOINTBAYES_YFEATURES.contains(Props.Y_FEATURE_CLASS.SIGMOID) &&
        count > 0) {
      double percent = ((double) count) / ((double) zLabels.length);
      double sigmoid = 1.0 / (1.0 + Math.exp(-10.0 * (percent - 1.0 / 3.0)));
      features.setCount(SIGMOID_FEAT, sigmoid);
    }

    return features;
  }

  public static List<Pair<String, Double>> sortPredictions(Counter<String> scores) {
    List<Pair<String, Double>> sortedScores = new ArrayList<Pair<String,Double>>();
    for(String key: scores.keySet()) {
      sortedScores.add(new Pair<String, Double>(key, scores.getCount(key)));
    }
    sortPredictions(sortedScores);
    return sortedScores;
  }

  private static void sortPredictions(List<Pair<String, Double>> scores) {
    Collections.sort(scores, new Comparator<Pair<String, Double>>() {
      @Override
      public int compare(Pair<String, Double> o1, Pair<String, Double> o2) {
        if(o1.second() > o2.second()) return -1;
        if(o1.second().equals(o2.second())){
          // this is an arbitrary decision to disambiguate ties
          int c = o1.first().compareTo(o2.first());
          if(c < 0) return -1;
          else if(c == 0) return 0;
          return 1;
        }
        return 1;
      }
    });
  }

  /**
   * Implements weighted voting over the different Z classifiers in each fold
   * @return Probabilities (NOT log probs!) for each known label
   */
  private Counter<String> classifyLocally(Collection<String> sentence) {
    Datum<String, String> datum = new BasicDatum<String, String>(sentence);

    if(localClassificationMode == LOCAL_CLASSIFICATION_MODE.WEIGHTED_VOTE) {
      Counter<String> sumProbs = new ClassicCounter<String>();

      for(int fold = 0; fold < numberOfFolds; fold ++) {
        LinearClassifier<String, String> zClassifier = zClassifiers[fold];
        Counter<String> probs = zClassifier.probabilityOf(datum);

        sumProbs.addAll(probs);
      }

      for(String l: sumProbs.keySet())
        sumProbs.setCount(l, sumProbs.getCount(l) / numberOfFolds);
      return sumProbs;
    }

    if(localClassificationMode == LOCAL_CLASSIFICATION_MODE.SINGLE_MODEL) {
      Counter<String> probs = zSingleClassifier.probabilityOf(datum);
      return probs;
    }

    throw new RuntimeException("ERROR: classification mode " + localClassificationMode + " not supported!");
  }

  public Counter<String> classifyOracleMentions(
      List<Collection<String>> sentences,
      Set<String> goldLabels) {
    Counter<String> [] zProbs =
      ErasureUtils.uncheckedCast(new Counter[sentences.size()]);

    //
    // Z level predictions
    //
    Counter<String> yLabels = new ClassicCounter<String>();
    Map<String, Counter<Integer>> ranks = new HashMap<String, Counter<Integer>>();
    for (int i = 0; i < sentences.size(); i++) {
      Collection<String> sentence = sentences.get(i);
      zProbs[i] = classifyLocally(sentence);

      if(! uniformDistribution(zProbs[i])) {
        List<Pair<String, Double>> sortedProbs = sortPredictions(zProbs[i]);
        double top = sortedProbs.get(0).second();
        for(int j = 0; j < sortedProbs.size() && j < 3; j ++) {
          String l = sortedProbs.get(j).first();
          double v = sortedProbs.get(j).second();
          if(v + 0.99 < top) break;
          if(! l.equals(RelationMention.UNRELATED)){ // && v + 0.99 > top){// && goldLabels.contains(l)) {
            double rank = 1.0 / (1.0 + (double) j);
            Counter<Integer> lRanks = ranks.get(l);
            if(lRanks == null) {
              lRanks = new ClassicCounter<Integer>();
              ranks.put(l, lRanks);
            }
            lRanks.setCount(j, rank);
          }
        }
      }
    }

    for(String l: ranks.keySet()) {
      double sum = 0;
      for(int position: ranks.get(l).keySet()) {
        sum += ranks.get(l).getCount(position);
      }
      double rank = sum / sentences.size(); // ranks.get(l).keySet().size();
      logger.log("RANK = " + rank);
      if(rank >= 0) // 0.001)
        yLabels.setCount(l, rank);
    }

    return yLabels;
  }

  @Override
  public Pair<Double, Maybe<KBPRelationProvenance>> classifyRelation(SentenceGroup input, RelationType relation, Maybe<CoreMap[]> rawSentences) {
    Counter<Pair<String, Maybe<KBPRelationProvenance>>> predictions =  classifyRelations(input, rawSentences, Props.TRAIN_JOINTBAYES_OUTDISTRIBUTION_TYPES.Y_GIVEN_ZSTAR);
    for (Map.Entry<Pair<String, Maybe<KBPRelationProvenance>>, Double> entry : predictions.entrySet()) {
      if (entry.getKey().first.equals(relation.canonicalName)) { return Pair.makePair(entry.getValue(), entry.getKey().second); }
    }
    return Pair.makePair(0.0, Maybe.<KBPRelationProvenance>Nothing());
  }

  @Override
  public Counter<Pair<String, Maybe<KBPRelationProvenance>>> classifyRelations(SentenceGroup input, Maybe<CoreMap[]> rawSentences) {
    return classifyRelations(input, rawSentences, Props.TRAIN_JOINTBAYES_OUTDISTRIBUTION);
  }

  private Counter<Pair<String, Maybe<KBPRelationProvenance>>> classifyRelations(SentenceGroup input, Maybe<CoreMap[]> rawSentences, Props.TRAIN_JOINTBAYES_OUTDISTRIBUTION_TYPES outputType) {
    List<Collection<String>> sentences = RelationClassifier.tupleToFeatureList(input);
    // Variables of interest (filled in below)
    String[]           zLabelsGivenX = new String[sentences.size()];
    Counter<String> [] pZGivenX      = ErasureUtils.uncheckedCast(new Counter[sentences.size()]);
    Counter<String>    pYGivenZStar  = new ClassicCounter<String>();
//    Counter<String>    pYGivenX      = new ClassicCounter<String>();
    Map<String,KBPRelationProvenance>    provenances  = new HashMap<String, KBPRelationProvenance>();

    //
    // Z level predictions
    //
    Counter<String> sumZGivenX = new ClassicCounter<String>();
    Counter<String> maxZGivenX = new ClassicCounter<String>();
    Counter<String> noisyOrZGivenX = new ClassicCounter<String>();
    for (int i = 0; i < sentences.size(); i++) {
      // Classify P(zi | xi)
      Collection<String> sentence = sentences.get(i);
      Counter<String> pZGivenXi = classifyLocally(sentence);
      // Compute Log P(zi | xi)
      pZGivenX[i] = new ClassicCounter<String>();
      for(String l: pZGivenXi.keySet()) {
        pZGivenX[i].setCount(l, pZGivenXi.getCount(l));
      }

      // Fill z label given x
      List<Pair<String, Double>> sortedProbs = sortPredictions(pZGivenXi);
      Pair<String, Double> prediction = sortedProbs.get(0);
      String predictedLabel = prediction.first();
      double predictionScore = prediction.second();
      zLabelsGivenX[i] = predictedLabel;

      if(! predictedLabel.equals(RelationMention.UNRELATED)) { // we do not output NIL labels
        // Add the sum z predictions \sum_i P(zi | xi)
        sumZGivenX.incrementCount(predictedLabel, predictionScore);
        // Update the max z predictions \sum_i P(zi | xi)
        if(! maxZGivenX.containsKey(predictedLabel) || predictionScore > maxZGivenX.getCount(predictedLabel)) {
          maxZGivenX.setCount(predictedLabel, predictionScore);
          if (input.getProvenance(i).isOfficial()) {
            provenances.put(predictedLabel, input.getProvenance(i).rewrite(predictionScore));
          }
        }

        // Set provenance (if official index)
        if (!provenances.containsKey(predictedLabel) && input.getProvenance(i).isOfficial()) {
          provenances.put(predictedLabel, input.getProvenance(i));
        }

        // Compute [product for] noisy or noisy_or( P(zi | xi) )
        double crt = (noisyOrZGivenX.containsKey(predictedLabel) ? noisyOrZGivenX.getCount(predictedLabel) : 1.0);
        crt = crt * (1.0 - predictionScore);
        noisyOrZGivenX.setCount(predictedLabel, crt);
      }
    }

    int[] zLabelGivenXInterned = new int[zLabelsGivenX.length];
    for (int i = 0; i < zLabelsGivenX.length; i++) {
      zLabelGivenXInterned[i] = zLabelIndex.indexOf(zLabelsGivenX[i]);
      if (zLabelGivenXInterned[i] < 0) throw new IllegalStateException("Unknown label: " + zLabelsGivenX[i]);
    }
    // Re-invert noisy or of z given x
    Counters.multiplyInPlace(noisyOrZGivenX, -1.0);
    Counters.addInPlace(noisyOrZGivenX, 1.0);

    //
    // Y level predictions
    //
    Counter<String> pYGivenZStarGreaterThanHalf = new ClassicCounter<String>();
    for (String yLabel : yClassifiers.keySet()) {
      // Get classifier
      LinearClassifier<String, String> yClassifier = yClassifiers.get(yLabel);
      Counter<String> features = extractYFeatures(yLabel, zLabelGivenXInterned);
      Datum<String, String> datum = new RVFDatum<String, String>(features, "");
      try {
        // Compute P( y | z )
        Counter<String> probs = yClassifier.probabilityOf(datum);
        double posScore = probs.getCount(yLabel);
        double negScore = probs.getCount(RelationMention.UNRELATED);
        double prob = posScore / (posScore + negScore);

        // Set P( y | z)
        pYGivenZStar.setCount(yLabel, prob);
        if ((!Props.TEST_THRESHOLD_JOINTBAYES_PERRELATION.containsKey(yLabel) && prob > Props.TEST_THRESHOLD_JOINTBAYES_DEFAULT) ||
            (Props.TEST_THRESHOLD_JOINTBAYES_PERRELATION.containsKey(yLabel) && prob > Props.TEST_THRESHOLD_JOINTBAYES_PERRELATION.get(yLabel))) {
          pYGivenZStarGreaterThanHalf.incrementCount(yLabel, prob);
        }
      } catch (Exception e) {
        logger.err(e);
      }
    }

    // Make prediction
    Counter<Pair<String, Maybe<KBPRelationProvenance>>> joint = new ClassicCounter<Pair<String, Maybe<KBPRelationProvenance>>>();
    for(String l: pYGivenZStar.keySet()) {
      double yProb = 1.0;
      double zProb = noisyOrZGivenX.getCount(l);
      switch (outputType) {
        case Y_GIVEN_ZSTAR:
          joint.setCount(Pair.makePair(l, Maybe.fromNull(provenances.get(l))), pYGivenZStar.getCount(l));
          break;
        case NOISY_OR:
          if ((!Props.TEST_THRESHOLD_JOINTBAYES_PERRELATION.containsKey(l) && yProb * zProb > Props.TEST_THRESHOLD_JOINTBAYES_DEFAULT) ||
              (Props.TEST_THRESHOLD_JOINTBAYES_PERRELATION.containsKey(l) && yProb * zProb > Props.TEST_THRESHOLD_JOINTBAYES_PERRELATION.get(l))) {
            joint.setCount(Pair.makePair(l, Maybe.fromNull(provenances.get(l))), yProb * zProb);
          }
          break;
        case Y_THEN_NOISY_OR:
          if (pYGivenZStarGreaterThanHalf.containsKey(l)) {
            joint.setCount(Pair.makePair(l, Maybe.fromNull(provenances.get(l))), yProb * zProb);
          }
          break;
        default:
          throw new IllegalStateException("Unknown output type: " + Props.TRAIN_JOINTBAYES_OUTDISTRIBUTION);
      }
    }

    if (Props.TRAIN_JOINTBAYES_OUTDISTRIBUTION == Props.TRAIN_JOINTBAYES_OUTDISTRIBUTION_TYPES.Y_GIVEN_ZSTAR) {
      Counters.normalize(joint);
    }

    return joint;
  }

  @Override
  public void save(ObjectOutputStream out) throws IOException {
    out.writeObject(knownDependencies);
    out.writeObject(zLabelIndex);
    out.writeInt(zClassifiers.length);
    for(int i = 0; i < zClassifiers.length; i ++)
      out.writeObject(zClassifiers[i]);
    out.writeObject(zSingleClassifier);
    out.writeInt(yClassifiers.keySet().size());
    for (String yLabel : yClassifiers.keySet()) {
      out.writeObject(yLabel);
      out.writeObject(yClassifiers.get(yLabel));
    }
  }

  @Override
  public void load(ObjectInputStream in) throws IOException, ClassNotFoundException {
    startTrack("Loading Joint Bayes relation extractor (from input stream)");
    knownDependencies = ErasureUtils.uncheckedCast(in.readObject());
    zLabelIndex = ErasureUtils.uncheckedCast(in.readObject());

    numberOfFolds = in.readInt();
    zClassifiers = ErasureUtils.uncheckedCast(new LinearClassifier[numberOfFolds]);
    for(int i = 0; i < numberOfFolds; i ++){
      LinearClassifier<String, String> classifier =
        ErasureUtils.uncheckedCast(in.readObject());
      zClassifiers[i] = classifier;
    }
    zSingleClassifier =
      ErasureUtils.uncheckedCast(in.readObject());

    int numLabels = in.readInt();
    yClassifiers = new HashMap<String, LinearClassifier<String, String>>();
    for (int i = 0; i < numLabels; i++) {
      String yLabel = ErasureUtils.uncheckedCast(in.readObject());
      LinearClassifier<String, String> classifier =
        ErasureUtils.uncheckedCast(in.readObject());
      yClassifiers.put(yLabel, classifier);
      logger.log("Loaded Y classifier for label " + yLabel +
          ": " + classifier.toAllWeightsString());
    }
    endTrack("Loading Joint Bayes relation extractor (from input stream)");
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof JointBayesRelationExtractor)) return false;

    JointBayesRelationExtractor that = (JointBayesRelationExtractor) o;

    if (numberOfFolds != that.numberOfFolds) return false;
    if (numberOfTrainEpochs != that.numberOfTrainEpochs) return false;
    if (onlyLocalTraining != that.onlyLocalTraining) return false;
    if (trainY != that.trainY) return false;
    if (initialModelPath != null ? !initialModelPath.equals(that.initialModelPath) : that.initialModelPath != null)
      return false;
    if (serializedModelPath != null ? !serializedModelPath.equals(that.serializedModelPath) : that.serializedModelPath != null)
      return false;

    return true;
  }

  @Override
  public int hashCode() {
    int result = numberOfTrainEpochs;
    result = 31 * result + numberOfFolds;
    result = 31 * result + (onlyLocalTraining ? 1 : 0);
    result = 31 * result + (initialModelPath != null ? initialModelPath.hashCode() : 0);
    result = 31 * result + (trainY ? 1 : 0);
    result = 31 * result + (serializedModelPath != null ? serializedModelPath.hashCode() : 0);
    return result;
  }

  public static JointBayesRelationExtractor load(String modelPath, Properties props) throws IOException, ClassNotFoundException {
    return RelationClassifier.load(modelPath, props, JointBayesRelationExtractor.class);
  }
}
