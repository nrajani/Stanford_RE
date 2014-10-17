package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.kbp.slotfilling.classify.RelationClassifier;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.common.CollectionUtils;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.ir.StandardIR;
import edu.stanford.nlp.kbp.slotfilling.process.KBPProcess;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.*;
import java.text.DecimalFormat;
import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * An implementation of a SlotFiller and an Evaluator.
 *
 * This class will fill slots given a query entity (complete with provenance).
 * It also evaluates a list of input-output (entity, List[SlotFill]) pairs.
 * The class depends on an IR component, the components in process to annotate documents, and the classifiers.
 *
 * @author Gabor Angeli
 */
public class KBPEvaluator implements Evaluator {

  protected static final Redwood.RedwoodChannels logger = Redwood.channels("Eval");

  // Defining Instance Variables
  private final Properties props;
  protected final KBPIR irComponent;
  protected final KBPProcess processComponent;
  protected final RelationClassifier classifier;
  public final SlotFiller slotFiller;
  private final Set<String> allSlotFillsSeen = new HashSet<String>();
  protected final OfficialOutputWriter officialOutputWriter;
  protected final GoldResponseSet goldResponses;

  /**
   * The mode used to compute precision and recall at test time
   *
   * @author kevinreschke
   *
   */
  public static enum ScoreMode {
    OFFICIAL,  //recall based on pooled results from other teams
    IR_RECALL;  //recall based on ir recall from this run

    public static ScoreMode fromString(String s) {
      if("official".equals(s))      return ScoreMode.OFFICIAL;
      else if("irRecall".equals(s)) return ScoreMode.IR_RECALL;
      else                          throw new RuntimeException("ERROR: Unknown classify type: " + s);
    }
  }

  public static enum SlotFillerType {
    SIMPLE,
    INFERENTIAL;

    public static SlotFillerType fromString(String s) {
      if("simple".equals(s))      return SlotFillerType.SIMPLE;
      else if("inferential".equals(s)) return SlotFillerType.INFERENTIAL;
      else                          throw new RuntimeException("ERROR: Unknown slot filler type: " + s);
    }
  }

  public static  enum ListOutput {
    BEST,
    ALL,
    TOP
  }

  public KBPEvaluator(Properties props,
                      KBPIR ir,
                      KBPProcess process,
                      RelationClassifier classify) {
    this.props = props;
    this.irComponent = ir;
    this.processComponent = process;
    this.classifier = classify;

    this.goldResponses = new GoldResponseSet(testEntities());

    switch( Props.TEST_SLOTFILLING_MODE ) {
      case SIMPLE: {
        this.slotFiller = new SimpleSlotFiller(props, ir, process, classify, goldResponses);
      } break;
      case INFERENTIAL: {
        this.slotFiller = new InferentialSlotFiller(props, ir, process, classify, goldResponses);
      } break;
      default: {
        throw new RuntimeException("Not implemented yet: " + Props.TEST_SLOTFILLING_MODE );
      }
    }

    switch (Props.KBP_YEAR) {
      case KBP2009: officialOutputWriter = new OfficialOutputWriter.OfficialOutputWriter2009(); break;
      case KBP2010: officialOutputWriter = new OfficialOutputWriter.OfficialOutputWriter2010(); break;
      case KBP2011: officialOutputWriter = new OfficialOutputWriter.OfficialOutputWriter2011(); break;
      case KBP2012: officialOutputWriter = new OfficialOutputWriter.OfficialOutputWriter2012(); break;
      case KBP2013: officialOutputWriter = new OfficialOutputWriter.OfficialOutputWriter2013((StandardIR) this.irComponent); break;
      default: throw new IllegalStateException("Cannot create official output writer for year: " + Props.KBP_YEAR);
    }
  }

  public SlotFiller getSlotFiller() {
    return slotFiller;
  }

  //
  // Interface
  //

  public Maybe<KBPScore> run() {
    // Get entities to test on (from file)
    startTrack("Getting Test Entities");
    List<KBPOfficialEntity> entities = testEntities();

    // Get subset for debugging
    if( !Props.TEST_QUERY_NAME.isEmpty() ) {
      entities = CollectionUtils.filter(entities, new Function<KBPOfficialEntity, Boolean>() {
        @Override
        public Boolean apply(KBPOfficialEntity in) {
          return in.name.equals(Props.TEST_QUERY_NAME);
        }
      });
    }
    else if(Props.TEST_NQUERIES < entities.size() - Props.TEST_QUERYSTART) {
      entities = entities.subList(Props.TEST_QUERYSTART, Props.TEST_QUERYSTART + Props.TEST_NQUERIES);
    }
    endTrack("Getting Test Entities");

    // Fill slots
  //  startTrack("Processing Test Entities [" + entities.size() + "]");
    Map<KBPOfficialEntity, Collection<KBPSlotFill>> fillsByEntity = new HashMap<KBPOfficialEntity, Collection<KBPSlotFill>>();
    for (KBPOfficialEntity entity : entities) {
      List<KBPSlotFill> fills = slotFiller.fillSlots(entity);
      if(fills!=null){
    	  fillsByEntity.put(entity,fills ); 
      }
      else{
    	  System.out.println("receiving null for fills."+ entities.size() );
      }
     
    }
   //endTrack("Processing Test Entities [" + entities.size() + "]");

    // Evaluate datums
    startTrack("Evaluating Test Entities");
    Maybe<KBPScore> score = evaluate(fillsByEntity);
    endTrack("Evaluating Test Entities");
    return score;
  }

  private Maybe<List<KBPOfficialEntity>> testEntitiesCache = Maybe.Nothing();

  @Override
  public List<KBPOfficialEntity> testEntities() {
    //noinspection LoopStatementThatDoesntLoop
    for (List<KBPOfficialEntity> cached : testEntitiesCache) { return cached; }
    testEntitiesCache = Maybe.Just(DataUtils.testEntities(Props.TEST_QUERIES.getPath(), Maybe.Just(this.irComponent)));
    return testEntitiesCache.get();
  }

  @SuppressWarnings("ConstantConditions")
  @Override
  public Maybe<KBPScore> evaluate(Map<KBPOfficialEntity, Collection<KBPSlotFill>> fillsByEntity) {
    @SuppressWarnings("UnusedAssignment")
    Maybe<KBPScore> score = Maybe.Nothing();
    try {
      // -- Relevant Variables
      // Output strategy
      ListOutput listOutput = Props.TEST_LIST_OUTPUT;
      logger.log("Strategy for list slots is: " + listOutput);
      // Key File
     // assert Props.TEST_RESPONSES.isDefined();
      File keyFile = this.goldResponses.keyFile();
      System.out.println("done.");
      logger.log("using key file " + keyFile);


      // -- Slot thresholds
      startTrack("Setting Threshold[s]");
      // Define variables
      Map<RelationType, Double> relationNameToThresholds = new HashMap<RelationType, Double>();
      double singleThreshold = Props.TEST_THRESHOLD_MIN_GLOBAL;
      logger.log("when scoring, accept any doc: " + Props.TEST_ANYDOC);
      // Calculate thresholds
      switch (Props.TEST_THRESHOLD_TUNE) {
        case PER_RELATION:
          logger.log("thresholding slots per relation");
          if (props.containsKey(Props.TEST_THRESHOLD_MIN_GLOBAL)) {
            // load the thresholds from props:
            // the thresholds must be stored as a list of comma-separated values,
            // one threshold for each slot name.
            // the order of slot names is alphanumeric.
            int thresholdIndex = 0;
            RelationType[] relationsSorted = RelationType.values();
            Arrays.sort(relationsSorted, new Comparator<RelationType>() {
              public int compare(RelationType o1, RelationType o2) { return o1.canonicalName.compareTo(o2.canonicalName); }
            });
            for (RelationType relationName : relationsSorted) {
              relationNameToThresholds.put(relationName, Props.TEST_THRESHOLD_MIN_PERRELATION[thresholdIndex++]);
            }
          } else {
            // estimate a threshold for each slot name
            for (RelationType relationName : RelationType.values()) {
              double perSlotThreshold = tuneThreshold(fillsByEntity, Maybe.Just(relationName.canonicalName), keyFile);
              logger.log("threshold for " + relationName + " is " + perSlotThreshold);
              relationNameToThresholds.put(relationName, perSlotThreshold);
            }
          }
          break;
        case GLOBAL:
          singleThreshold = tuneThreshold(fillsByEntity, Maybe.<String>Nothing(), keyFile);
          // note: no break
        case NONE:
          for (RelationType relationName : RelationType.values()) {
            relationNameToThresholds.put(relationName, singleThreshold);
          }
          break;
        default: throw new IllegalStateException("Unknown tuning mode: " + Props.TEST_THRESHOLD_TUNE);
      }
      endTrack("Setting Threshold[s]");

      // -- Create Query File
      startTrack("Create Query File");
      officialOutputWriter.outputRelations(System.err, Props.KBP_RUNID, fillsByEntity, relationNameToThresholds);
      // Write Query File
      String outputFileName = Props.WORK_DIR.getPath() + File.separator + Props.KBP_RUNID + ".output";
      logger.log(FORCE, BLUE, BOLD, "writing query file to " + outputFileName);
      PrintStream os = new PrintStream(outputFileName, "UTF-8");
      officialOutputWriter.outputRelations(os, Props.KBP_RUNID, fillsByEntity, relationNameToThresholds);
      os.close();
      endTrack("Create Query File");

      // -- Scoring Mode
      startTrack("Scoring System");
      //choose values to score based on score mode param
      Maybe<Set<String>> scoreOnlyTheseValues;
      switch (Props.TEST_SCORE_MODE) {
        case OFFICIAL:
          logger.log(BLUE, BOLD, "using official score");
          scoreOnlyTheseValues = Maybe.Nothing();
          break;
        case IR_RECALL:
          logger.log(BLUE, BOLD, "using UNOFFICIAL score against our IR recall");
          scoreOnlyTheseValues = Maybe.Just(allSlotFillsSeen);  // score all candidates found in IR output
          break;
        default: throw new IllegalStateException("Unknown score mode: " + Props.TEST_SCORE_MODE);
      }

      // -- Score Model
      //
      // score using the official scorer
      // keyFile must be in 2010 format. If it's in 2009 format, use UpdateSFKey to convert to 2010
      //
      // Human readable score
      System.out.println("Official KBP score:");
      Set<String> queryIds = extractQueryIds(fillsByEntity.keySet());
      CustomSFScore.scoreByRelationName(System.out, outputFileName, keyFile.getPath(), Props.TEST_ANYDOC, scoreOnlyTheseValues, queryIds);

      // Write to file
      int threshold = (int) (100.0 * singleThreshold);
      String scoreFileName = Props.WORK_DIR.getPath() + File.separator + Props.KBP_RUNID + "." + Props.TEST_QUERYSCOREFILE
          + "_t" + threshold + ".txt";
      PrintStream sos = new PrintStream(new FileOutputStream(scoreFileName));
      Pair<Double, Double> precisionRecall = CustomSFScore.scoreByRelationName(sos, outputFileName, keyFile.getPath(), Props.TEST_ANYDOC, scoreOnlyTheseValues, queryIds);
      sos.close();
      endTrack("Scoring System");

      // -- PR Curve
      forceTrack("Generating PR Curve");
      // this generates the scores for the P/R curve
      Pair<double[], double[]> curve = generatePRCurveNonProbScores(fillsByEntity, scoreOnlyTheseValues, keyFile);
      endTrack("Generating PR Curve");

      // -- Write in 2013 format TODO(gabor) remove me after 2013 deadline
      try {
        Props.YEAR actualYear = Props.KBP_YEAR;
        Props.KBP_YEAR = Props.YEAR.KBP2013;
        os = new PrintStream(Props.WORK_DIR + File.separator + "kbp2013.output", "UTF-8");
        System.out.println(Props.WORK_DIR + File.separator + "kbp2013.output");
        new OfficialOutputWriter.OfficialOutputWriter2013((StandardIR) this.irComponent).outputRelations(os, Props.KBP_RUNID, fillsByEntity, relationNameToThresholds);
        Props.KBP_YEAR = actualYear;
      } catch (Exception e) { logger.log(e); }

      score = Maybe.Just(new KBPScore(precisionRecall.first, precisionRecall.second, curve.first, curve.second));
    } catch (Exception e) {
      throw new RuntimeException(e);
    }


    // -- Return (at last!)
    for (KBPScore s : score) { Redwood.channels("Result", GREEN).prettyLog("Score", s); }
    return score;
  }


  //
  // Utilities -- Evaluation
  //

  /**
   * Determine the best empirical tuning threshold for slots. If relationName is
   * null, we will determine the best threshold for all slots. Otherwise, we
   * will restrict ourselves only to those relations of type relationName.
   */
  private double tuneThreshold(Map<KBPOfficialEntity, Collection<KBPSlotFill>> relations,
                               Maybe<String> relationNameMaybe,
                               File keyFile) throws Exception {
    double bestThreshold = -1;
    double bestF1 = Double.MIN_VALUE;
    double bestP = Double.MIN_VALUE;
    double bestR = Double.MIN_VALUE;
    String prefix = "TUNING: (" + (relationNameMaybe.getOrElse("--") + ") ");
    logger.log(prefix + "started...");

    // run the system with a very inclusive threshold
    Maybe<Set<String>> allCandidates = Maybe.Just((Set<String>) new HashSet<String>());
    Set<String> queryIds = extractQueryIds(relations.keySet());

    // Set<String> slots = null;
    for (String relationName : relationNameMaybe) {
      // Filter relations by name
      // Keeps only the relations of a specific slot name. The original relations are not modified in any way.
      Set<KBPOfficialEntity> entities = relations.keySet();
      Map<KBPOfficialEntity, Collection<KBPSlotFill>> filteredRelations = new HashMap<KBPOfficialEntity, Collection<KBPSlotFill>>();
      for (KBPOfficialEntity entity : entities) {
        Collection<KBPSlotFill> myRelations = relations.get(entity);
        Collection<KBPSlotFill> myFilteredRels = new ArrayList<KBPSlotFill>();
        for (KBPSlotFill rel : myRelations) {
          if (rel.key.slotType.orCrash().name.equals(relationName)) {
            myFilteredRels.add(rel);
          }
        }
        filteredRelations.put(entity, myFilteredRels);
      }
      relations = filteredRelations;
      // relation specific scoring is busted for now
      // slots = new HashSet<String>();
      // slots.add(relationName.replaceAll("SLASH", "/"));
    }

    for (double singleThreshold = 0.00; singleThreshold <= 10.00; singleThreshold += 0.10) {
      // Filter Relations By Threshold
      // Keeps only the relations predicted with a score larger than threshold the original relations are not modified in any way.
      Set<KBPOfficialEntity> entities = relations.keySet();
      Map<KBPOfficialEntity, Collection<KBPSlotFill>> filteredRelations = new HashMap<KBPOfficialEntity, Collection<KBPSlotFill>>();
      for (KBPOfficialEntity entity : entities) {
        Collection<KBPSlotFill> myRelations = relations.get(entity);
        Collection<KBPSlotFill> myFilteredRels = new ArrayList<KBPSlotFill>();
        for (KBPSlotFill rel : myRelations) {
          if (rel.score.orCrash() >= singleThreshold) {
            myFilteredRels.add(rel);
          }
        }
        filteredRelations.put(entity, myFilteredRels);
      }

      // generate scorable output
      String workDir = Props.WORK_DIR.getPath();
      String outputFileName = workDir + File.separator + Props.KBP_RUNID + ".dev.output";
      File outputFile = new File(outputFileName);
      outputFile.deleteOnExit();
      PrintStream os = new PrintStream(outputFile);
      officialOutputWriter.outputRelations(os, Props.KBP_RUNID, filteredRelations, new HashMap<RelationType, Double>());
      os.close();

      // score using the official scorer
      String scoreFileName = workDir + File.separator + Props.KBP_RUNID + "." + Props.TEST_QUERYSCOREFILE
          + ".dev.txt";
      File scoreFile = new File(scoreFileName);
      scoreFile.deleteOnExit();
      PrintStream sos = new PrintStream(new FileOutputStream(scoreFile));
      Pair<Double, Double> score = CustomSFScore.score(sos, outputFileName, keyFile.getPath(), null, Props.TEST_ANYDOC, allCandidates, queryIds);
      double f1 = CustomSFScore.pairToFscore(score);
      sos.close();
      logger.log(prefix + "F1 score for threshold " + singleThreshold + " is " + f1 + "(P " + score.first + ", R "
          + score.second + ")");

      // best so far?
      if (f1 > bestF1) {
        bestThreshold = singleThreshold;
        bestF1 = f1;
        bestP = score.first();
        bestR = score.second();
        logger.log(prefix + "found current best F1: " + bestF1);
      }
    }

    String suffix = "";
    if (bestThreshold == -1) {
      suffix = " (didn't find any useful threshold settings, using default)";
      bestThreshold = Props.DEFAULT_SLOT_THRESHOLD;
    }

    logger.log(prefix + "selected final threshold " + bestThreshold + " with P " + bestP + " R " + bestR + " F1 "
        + bestF1 + suffix);
    return bestThreshold;
  }

  /**
   * Generate a PR curve
   * @param relations The relations to score
   * @param allCandidates All the candidate slot fills
   * @throws IOException If the PR file cannot be written
   */
  private Pair<double[], double[]> generatePRCurveNonProbScores(Map<KBPOfficialEntity, Collection<KBPSlotFill>> relations,
                                            Maybe<Set<String>> allCandidates,
                                            File keyFile) throws IOException {
    String workDir = Props.WORK_DIR.getPath();
    DecimalFormat df = new DecimalFormat("00.00");

    File dir = new File(workDir + File.separator + Props.KBP_RUNID + ".prcurve.tmp");
    if (!dir.mkdir()) { logger.err(RED, "Could not create directory for PR curves: " + dir); }

    Set<String> queryIds = extractQueryIds(relations.keySet());

    // Sort Relations
    String prFileName = workDir + File.separator + Props.KBP_RUNID + ".curve";
    PrintStream mos = new PrintStream(prFileName);
    List<Pair<KBPOfficialEntity, KBPSlotFill>> sorted = new ArrayList<Pair<KBPOfficialEntity, KBPSlotFill>>();
    for(KBPOfficialEntity e: relations.keySet()) {
      for(KBPSlotFill s: relations.get(e)) {
        sorted.add(new Pair<KBPOfficialEntity, KBPSlotFill>(e, s));
      }
    }
    Collections.sort(sorted, new Comparator<Pair<KBPOfficialEntity, KBPSlotFill>>() {
      @Override
      public int compare(Pair<KBPOfficialEntity, KBPSlotFill> s1, Pair<KBPOfficialEntity, KBPSlotFill> s2) {
        if(s1.second().score.orCrash() > s2.second().score.orCrash()) return -1;
        if(s1.second().score.orCrash() < s2.second().score.orCrash()) return 1;
        return 0;
      }
    });
    int START_OFFSET = 10;
    logger.log("generating PR curve with " + sorted.size() + " points");

    double[] precisions = new double[sorted.size()];
    double[] recalls = new double[sorted.size()];
    for(int i = START_OFFSET; i < sorted.size(); i ++) {
      // Keep Top
      Map<KBPOfficialEntity, Collection<KBPSlotFill>> filteredRels = new HashMap<KBPOfficialEntity, Collection<KBPSlotFill>>();
      for(int j = 0; j <= i; j ++) {
        KBPOfficialEntity e = sorted.get(j).first();
        KBPSlotFill s = sorted.get(j).second();
        if (!filteredRels.containsKey(e)) { filteredRels.put(e, new ArrayList<KBPSlotFill>()); }
        filteredRels.get(e).add(s);
      }
      String outputFileName = dir + File.separator + Props.KBP_RUNID + ".i" + i + ".output";
      PrintStream os = new PrintStream(outputFileName);
      officialOutputWriter.outputRelations(os, Props.KBP_RUNID, filteredRels, new HashMap<RelationType, Double>());
      os.close();

      String scoreFileName = dir + File.separator + Props.KBP_RUNID + ".i" + i + ".score";
      PrintStream sos = new PrintStream(new FileOutputStream(scoreFileName));
      Pair<Double, Double> pr = CustomSFScore.score(sos, outputFileName, keyFile.getPath(), Props.TEST_ANYDOC, allCandidates, queryIds);
      if (pr.first.isNaN() || pr.first.isInfinite() || pr.second.isNaN() || pr.second.isInfinite()) {
        logger.err(RED, "Infinite PR @ " + i + ": Precision: " + pr.first + " Recall: " + pr.second);
      }
      double f1 = (pr.first() != 0 && pr.second() != 0 ? 2*pr.first()*pr.second()/(pr.first()+pr.second()) : 0.0);
      sos.close();

      double ratio = (double) i / (double) sorted.size();
      logger.debug("PR@" + df.format(ratio) + ": F1 " + df.format(f1 * 100.0) + " P " + df.format(pr.first * 100.0) + " R " + df.format(pr.second * 100.0));
      precisions[i] = pr.first;
      recalls[i] = pr.second;
      mos.println(ratio + " P " + pr.first() + " R " + pr.second() + " F1 " + f1);
    }
    mos.close();
    logger.log("P/R curve data generated in file: " + prFileName);

    // let's remove the tmp dir with partial scores. we are unlikely to need all this data.
    File [] tmpFiles = dir.listFiles();
    boolean deleteSuccess = true;
    if (tmpFiles != null) {
      for(File f: tmpFiles) {
        deleteSuccess = deleteSuccess && f.delete();
      }
    }
    deleteSuccess = deleteSuccess && dir.delete();
    if(! deleteSuccess) {
      logger.warn("Tried to delete P/R tmp directory but failed: " + dir.getAbsolutePath());
    }
    return Pair.makePair(precisions, recalls);
  }

  //
  // Public Utilities
  //

  /**
   * Get the set of Query IDs from a set of KBP Entities
   * @param ents The KBP Entities to extract ids from
   * @return A set of ids
   */
  public static Set<String> extractQueryIds(Set<KBPOfficialEntity> ents) {
    Set<String> qids = new HashSet<String>();
    for(KBPOfficialEntity e: ents) {
      if (e.queryId.isDefined()) {
        qids.add(e.queryId.get());
      } else {
        logger.warn("No query id for entity: " + e);
      }
    }
    return qids;
  }

}
