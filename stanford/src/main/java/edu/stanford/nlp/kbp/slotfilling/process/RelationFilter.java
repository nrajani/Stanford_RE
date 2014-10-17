package edu.stanford.nlp.kbp.slotfilling.process;

import java.util.*;

import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.common.NERTag;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Function;
import edu.stanford.nlp.util.MetaClass;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.logging.Redwood;

/**
 * This class is responsible for performing filtering of candidate relations  
 * within a single sentence.
 *
 * @author kevinreschke
 *
 */
public class RelationFilter {

  protected static final Redwood.RedwoodChannels logger = Redwood.channels("RelFilter");

  private Function<Pair<SentenceGroup, Maybe<CoreMap[]>>, Counter<String>> classifier;
  private List<FilterComponent> filterComponents;

  /**
   * Constructs a RelationFilter.  Called by RelationFilterBuilder.
   */
  private RelationFilter(Function<Pair<SentenceGroup, Maybe<CoreMap[]>>, Counter<String>> classifier,
                         List<FilterComponent> filterComponents) {
    this.classifier = classifier;
    this.filterComponents = filterComponents;
  }

  /**
   * Apply this filter. There are three steps.
   *
   *  1) Predict relation labels.
   *  
   *      A RelationFilter takes a RelationClassifier as a member variable.  This classifier
   *      is used to predict labels and likelihood scores for each relation, enabling score-based
   *      filtering.
   *      
   *  2) Filter.
   *  
   *      A RelationFilter takes a list of FilterComponents as a member variable.  These
   *      filter components are applied sequentially to the set of all pairwise relations.
   *      
   *  3) Reduce to main entity relations.
   *  
   *      Since filtering is applied to all relation pairs, after filtering, our relations set
   *      may contain relations which don't involve the main entity of interest.  We reduce this 
   *      set to only include relations headed by the main entity.
   *
   * 
   * @param sentenceGroupsForEntity
   *          Input list of singleton sentence groups where each singleton represents
   *          a datum derived from a relation headed by our main entity of interest.
   * @param sentenceGroupsForAllPairs
   *          Input list of singleton sentence groups where each singleton represents
   *          a datum derived from each of the pairwise relations in the sentence.
   *          These singletons are grouped by the relation's slot value entity
   *          (a.k.a. entity2).
   * @param sentence
   *          CoreMap of sentence.  This is the domain within which we filter.
   * @return A list of singleton sentence groups.  This will be a subset of the input
   *          sentence group list.  It is the results of filtering. 
   */
  public List<SentenceGroup> apply(List<SentenceGroup> sentenceGroupsForEntity,
      List<SentenceGroup> sentenceGroupsForAllPairs,
      CoreMap sentence) {

    Redwood.startTrack("Applying filter...");

    logger.debug("Applying filter...\n" +
        "\tnSentenceGroupsForEntity = "+sentenceGroupsForEntity.size()+"\n"+
        "\tnSentenceGroupsForAllPairs = "+sentenceGroupsForAllPairs.size()+"\n"+
        "\tsentence = "+ CoreMapUtils.sentenceToMinimalString(sentence));
    
//    if(sentenceGroupsForAllPairs.size() > 20 ) {
//      logger.debug("sentenceGroupsForAllPairs too big.  Skipping Filtering.");
//      Redwood.endTrack("Applying filter...");
//      return sentenceGroupsForEntity;
//    }
    
    //1) Predict relation labels.
    Pair<Map<SentenceGroup,String>,Map<SentenceGroup,Double>> tmp2 =
        predictLabels(sentenceGroupsForAllPairs, sentence);
    Map<SentenceGroup,String> predictions = tmp2.first();
    Map<SentenceGroup,Double> scores = tmp2.second();

    logger.debug("Label predictions...");
    for(SentenceGroup sg : sentenceGroupsForAllPairs) {
      logger.debug("\tsentenceGroupKey = "+sg.key+"\n"+
          "\t\tprediction = "+predictions.get(sg)+"\n"+
          "\t\tscore = "+scores.get(sg));
    }

    //2) Filter.
    for(FilterComponent filterComponent : filterComponents) {
      sentenceGroupsForAllPairs = filterComponent.filter(sentenceGroupsForAllPairs, predictions, scores, sentenceGroupsForEntity); 
    }

    //3) Reduce to main entity relations.
    //    - Hash the singleton sentence groups which survived filtering
    HashSet<SentenceGroup> keepers = new HashSet<SentenceGroup>();
    for(SentenceGroup sg : sentenceGroupsForAllPairs) {
      keepers.add(sg);
    }
    //    - construct new sentence group list for the entity, preserving original ordering
    List<SentenceGroup> filteredSentenceGroupsForEntity = new ArrayList<SentenceGroup>();
    for(SentenceGroup sg : sentenceGroupsForEntity) {
      if(keepers.contains(sg)) {
        filteredSentenceGroupsForEntity.add(sg);
      }
    }

    logger.debug("SentenceGroupsForEntity...");
    logger.debug("\tBefore Filtering:");
    for(SentenceGroup sg : sentenceGroupsForEntity) {
      logger.debug("\t\tkey = "+sg.key);
    }
    logger.debug("\t\tTotal: "+sentenceGroupsForEntity.size());
    logger.debug("\tAfter Filtering:");
    for(SentenceGroup sg : filteredSentenceGroupsForEntity) {
      logger.debug("\t\tkey = "+sg.key);
    }
    logger.debug("\t\tTotal: "+filteredSentenceGroupsForEntity.size());

    if(sentenceGroupsForEntity.size() == filteredSentenceGroupsForEntity.size()) {
      logger.debug("RelationFilter had no effect.");
    }
    else {
      logger.debug("RelationFilter caused reduction.");
    }

    Redwood.endTrack("Applying filter...");

    return filteredSentenceGroupsForEntity;
  }

  private Pair<Map<SentenceGroup, String>, Map<SentenceGroup, Double>> predictLabels(
      List<SentenceGroup> sentenceGroups,
      CoreMap sentence) {
    Map<SentenceGroup,String> predictions = new HashMap<SentenceGroup,String>();
    Map<SentenceGroup,Double> scores = new HashMap<SentenceGroup,Double>();
    for(SentenceGroup sg : sentenceGroups) {
      CoreMap[] rawSentences = {sentence};
      Counter<String> result = classifier.apply(Pair.makePair(sg,Maybe.Just(rawSentences)));
      String hardPrediction = getHardPrediction(result,sg.key.entityType,sg.key.slotType);
      predictions.put(sg,hardPrediction);
      scores.put(sg, result.getCount(hardPrediction));
    }
    return Pair.makePair(predictions, scores);
  }
  
  // Get hard label prediction (argmax) subject to sanity check: correct entity
  //  types for the relation.
  // If no suitable predictions occur, return RelationMention.UNRELATED
  private String getHardPrediction(Counter<String> labelProbs, NERTag entityType, Maybe<NERTag> slotType) {
    List<String> labelsBestToWorst = Counters.toSortedList(labelProbs);
    String hardPrediction = null;
    for(int i = 0; hardPrediction == null && i < labelsBestToWorst.size(); i++) {
      hardPrediction = labelsBestToWorst.get(i);
      // check sanity
      RelationType relation = RelationType.fromString(hardPrediction).orCrash();
      if (entityType != relation.entityType) {
        hardPrediction = null;
      }
      else {
        for (NERTag type : slotType) {
          if (!relation.validNamedEntityLabels.contains(type)) {
            hardPrediction = null;
          }
        }
      }
    }
    if(hardPrediction == null) {
      hardPrediction = RelationMention.UNRELATED;
    }
         
    return hardPrediction;
  }

  /**
   * Builder class for collecting parameters and producing RelationFilter instances
   */
  public static class RelationFilterBuilder {
    private Function<Pair<SentenceGroup, Maybe<CoreMap[]>>, Counter<String>> classifier;
    final List<FilterComponent> filterComponents = new ArrayList<FilterComponent>();

    public RelationFilterBuilder(Function<Pair<SentenceGroup, Maybe<CoreMap[]>>, Counter<String>> classifier) {
      assert classifier != null;
      this.classifier = classifier;
    }

    @SuppressWarnings("UnusedDeclaration")
    public void addFilterComponentByName(String className) {
      // Try aliases
      if("coref".equals(className)) {
        className = CorefFilterComponent.class.getName();
      } else if("perRelTypeCompetition".equals(className)) {
        className = PerRelTypeCompetitionFilterComponent.class.getName();
      } else if("crossRelTypeCompetition".equals(className)) {
        className = CrossRelTypeCompetitionFilterComponent.class.getName();
      }
      // Construct filter
      try {
        //noinspection unchecked
        addFilterComponent((Class<FilterComponent>) Class.forName(className));
      } catch (ClassNotFoundException e) {
        throw new RuntimeException("[RelationFilterBuilder.addFilter] Unknown name: "+className);
      } catch (ClassCastException e) {
        throw new RuntimeException("[RelationFilterBuilder.addFilter] Not a relation filter: "+className);
      }
    }

    public void addFilterComponent(Class<FilterComponent> filter) {
      assert filter != null;
      filterComponents.add(MetaClass.create(filter).<FilterComponent>createInstance());
    }

    public RelationFilter make() {
      if(classifier == null) {
        throw new RuntimeException("[RelationFilterBuilder.make()] Classifier not set.");
      }
      else if(filterComponents.size() == 0) {
        throw new RuntimeException("[RelationFilterBuilder.make()] Must add at least one filter component.");
      }
      else {
        return new RelationFilter(classifier, filterComponents);
      }
    }
  }

  /**
   * Subclasses of FilterComponent handle the actual filtering logic.
   * 
   * @author kevinreschke
   *
   */
  public static abstract class FilterComponent {

    /**
     * @param sentenceGroupsForAllPairs
     *          List of singleton sentence groups, including relations headed
     *          by the main entity as well as other pairwise relations.
     *          This is the input list that undergoes filtering.
     * @param predictions
     *          Label predictions for each sentence group in sentenceGroupsForAllPairs
     * @param scores
     *          The score given by the classifier for each prediction
     * @param sentenceGroupsForEntity
     *          List of singleton sentence groups, including only
     *          relations headed by the main entity (or its coref
     *          chain).  This list is not the list that undergoes filtering.
     *          It is included because some filter components need to
     *          know which relations in sentenceGroupsForAllPairs pertain
     *          to the main entity.
     * @return A list of sentence groups, the result of filtering.  This
     *          will be a subest of sentenceGroupsForAllPairs.
     */
    public abstract List<SentenceGroup> filter(List<SentenceGroup> sentenceGroupsForAllPairs,
        Map<SentenceGroup,String> predictions,
        Map<SentenceGroup,Double> scores,
        List<SentenceGroup> sentenceGroupsForEntity);

    // group sentence groups by their slot values
    protected static Map<String, List<SentenceGroup>> groupBySlotValue(
        List<SentenceGroup> sentenceGroups) {
      Map<String, List<SentenceGroup>> rtn = new HashMap<String, List<SentenceGroup>>();
      for(SentenceGroup sg : sentenceGroups) {
        String slotValue = sg.key.slotValue;
        List<SentenceGroup> sgs = rtn.get(slotValue);
        if(sgs == null) {
          sgs = new ArrayList<SentenceGroup>();
          rtn.put(slotValue,sgs);
        }
        sgs.add(sg);
      }
      return rtn;
    }
  }

  /**
   * When an entity coref chain occurs within a sentence, we get multiple relation mentions
   * between the main entity and a single slotValue entity.  This filter picks only the 
   * highest scoring of these relations.
   *
   * @author kevinreschke
   * 
   */
  public static class CorefFilterComponent extends FilterComponent {

    @Override
    public List<SentenceGroup> filter(List<SentenceGroup> sentenceGroupsForAllPairs,
        Map<SentenceGroup, String> predictions,
        Map<SentenceGroup, Double> scores,
        List<SentenceGroup> sentenceGroupsForEntity) {

      List<SentenceGroup> rtn = new ArrayList<SentenceGroup>();

      Map<String,List<SentenceGroup>> slotValueToSentenceGroups =
          groupBySlotValue(sentenceGroupsForAllPairs);

      //get all entity mentions in main entity coref chain
      Set<KBPEntity> entityCorefChain = new HashSet<KBPEntity>();
      for(SentenceGroup sg : sentenceGroupsForEntity) {
        entityCorefChain.add(sg.key.getEntity());
      }

      for(String slotValue : slotValueToSentenceGroups.keySet()) {
        List<SentenceGroup> sgsForSlotValue = slotValueToSentenceGroups.get(slotValue);
        double bestScore = Double.NEGATIVE_INFINITY;
        for(SentenceGroup sg : sgsForSlotValue) {
          List<SentenceGroup> bestSgs = new ArrayList<SentenceGroup>();
          if(!entityCorefChain.contains(sg.key.getEntity())) {
            // This filter only affects relations on the entity coref chain.
            // All other pairwise relations are passed through.
            rtn.add(sg);
          }
          else {
            double score = scores.get(sg);
            if(score == bestScore) {
              bestSgs.add(sg);
            }
            else if(score > bestScore) {
              bestScore = score;
              bestSgs = new ArrayList<SentenceGroup>();
              bestSgs.add(sg);
            }
          }
          rtn.addAll(bestSgs);
        }
      }

      logger.debug("Entered CorefFilterComponent with " + sentenceGroupsForAllPairs.size() +
          " sentence groups.");
      logger.debug("Exiting CorefFilterComponent with " + rtn.size() +
          " sentence groups.");

      if(sentenceGroupsForAllPairs.size() == rtn.size()) {
        logger.debug("CorefFilterComponent had no effect.");
      }
      else {
        logger.debug("CorefFilterComponent caused reduction.");
      }

      return rtn;
    }
  }

  /**
   * For each slot value and relation type, choose only the relation with the highest score.
   * 
   * @author kevinreschke
   *
   */
  public static class PerRelTypeCompetitionFilterComponent extends FilterComponent  {

    @Override
    public List<SentenceGroup> filter(List<SentenceGroup> sentenceGroupsForAllPairs,
        Map<SentenceGroup, String> predictions,
        Map<SentenceGroup, Double> scores,
        List<SentenceGroup> sentenceGroupsForEntity) {

      List<SentenceGroup> rtn = new ArrayList<SentenceGroup>();

      Map<String,List<SentenceGroup>> slotValueToSentenceGroups =
          groupBySlotValue(sentenceGroupsForAllPairs);

      for(String slotValue : slotValueToSentenceGroups.keySet()) {
        List<SentenceGroup> sgsForSlotValue = slotValueToSentenceGroups.get(slotValue);

        // group by predicted label
        Map<String,List<SentenceGroup>> sgsByLabel = new HashMap<String,List<SentenceGroup>>();
        for(SentenceGroup sg : sgsForSlotValue) {
          String label = predictions.get(sg);
          List<SentenceGroup> sgsForLabel = sgsByLabel.get(label);
          if(sgsForLabel == null) {
            sgsForLabel = new ArrayList<SentenceGroup>();
            sgsByLabel.put(label, sgsForLabel);
          }
          sgsForLabel.add(sg);
        }

        // keep only top scoring relations
        for(String label : sgsByLabel.keySet()) {
          List<SentenceGroup> bestSgs = new ArrayList<SentenceGroup>();
          double bestScore = Double.NEGATIVE_INFINITY;
          for(SentenceGroup sg : sgsByLabel.get(label)) {
            double score = scores.get(sg);
            if(score == bestScore) {
              bestSgs.add(sg);
            }
            else if(score > bestScore) {
              bestSgs = new ArrayList<SentenceGroup>();
              bestSgs.add(sg);
              bestScore = score;
            }
          }
          rtn.addAll(bestSgs);
        }
      }

      logger.debug("Entered PerRelTypeCompetitionFilterComponent with " + sentenceGroupsForAllPairs.size() +
          " sentence groups.");
      logger.debug("Exiting PerRelTypeCompetitionFilterComponent with " + rtn.size() +
          " sentence groups.");

      if(sentenceGroupsForAllPairs.size() == rtn.size()) {
        logger.debug("PerRelTypeCompetitionFilterComponent had no effect.");
      }
      else {
        logger.debug("PerRelTypeCompetitionFilterComponent caused reduction.");
      }

      return rtn;
    }
  }

  /**
   * For each slot value, regardless of relation type, choose only the relation with the highest score.
   * 
   * @author kevinreschke
   *
   */
  public static class CrossRelTypeCompetitionFilterComponent extends FilterComponent  {

    @Override
    public List<SentenceGroup> filter(List<SentenceGroup> sentenceGroupsForAllPairs,
        Map<SentenceGroup, String> predictions,
        Map<SentenceGroup, Double> scores,
        List<SentenceGroup> sentenceGroupsForEntity) {

      List<SentenceGroup> rtn = new ArrayList<SentenceGroup>();

      Map<String,List<SentenceGroup>> slotValueToSentenceGroups =
          groupBySlotValue(sentenceGroupsForAllPairs);

      for(String slotValue : slotValueToSentenceGroups.keySet()) {
        List<SentenceGroup> sgsForSlotValue = slotValueToSentenceGroups.get(slotValue);
        List<SentenceGroup> bestSgs = new ArrayList<SentenceGroup>();
        double bestScore = Double.NEGATIVE_INFINITY;
        for(SentenceGroup sg : sgsForSlotValue) {
          double score = scores.get(sg);
          if(score == bestScore) {
            bestSgs.add(sg);
          }
          else if(score > bestScore) {
            bestSgs = new ArrayList<SentenceGroup>();
            bestSgs.add(sg);
            bestScore = score;
          }
        }
        rtn.addAll(bestSgs);
      }

      logger.debug("Entered CrossRelTypeCompetitionFilterComponent with " + sentenceGroupsForAllPairs.size() +
          " sentence groups.");
      logger.debug("Exiting CrossRelTypeCompetitionFilterComponent with " + rtn.size() +
          " sentence groups.");
      
      if(sentenceGroupsForAllPairs.size() == rtn.size()) {
        logger.debug("CrossRelTypeCompetitionFilterComponent had no effect.");
      }
      else if (rtn.size() < sentenceGroupsForAllPairs.size()){
        logger.debug("CrossRelTypeCompetitionFilterComponent caused reduction.");
      }
      else {
        // error: this shouldn't happen if the code above is correct
        logger.debug("WARNING: CrossRelTypeCompetitionFilterComponent caused increase.");
      }

      return rtn;
    }

  }
}
