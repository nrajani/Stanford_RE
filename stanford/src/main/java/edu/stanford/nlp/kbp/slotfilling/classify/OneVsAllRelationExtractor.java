package edu.stanford.nlp.kbp.slotfilling.classify;

import static edu.stanford.nlp.util.logging.Redwood.Util.BLUE;
import static edu.stanford.nlp.util.logging.Redwood.Util.BOLD;
import static edu.stanford.nlp.util.logging.Redwood.Util.endTrack;
import static edu.stanford.nlp.util.logging.Redwood.Util.startTrack;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.classify.Dataset;
import edu.stanford.nlp.classify.GeneralDataset;
import edu.stanford.nlp.classify.LinearClassifierFactory;
import edu.stanford.nlp.classify.ProbabilisticClassifier;
import edu.stanford.nlp.classify.LogPrior;
import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.ErasureUtils;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.logging.Redwood;

/**
 * Multi-class local LR classifier with incomplete negatives
 * This is what we used at KBP 2011
 */
public class OneVsAllRelationExtractor extends RelationClassifier {
  private Map<String, ProbabilisticClassifier<String, String>> classifiers = null;
  
  /** Regularization coefficient */
  private double sigma;
  
  /** Softmax parameter */
  public final double gamma;
  
  public final boolean useRobustLR;

  @SuppressWarnings("UnusedDeclaration") // Used via reflection
  public OneVsAllRelationExtractor(Properties props) {
    this(false);
  }

  @SuppressWarnings("UnusedDeclaration") // Used via reflection
  public OneVsAllRelationExtractor(Properties props, boolean useRobustLR) {
    this(useRobustLR);
  }
  
  public OneVsAllRelationExtractor(boolean useRobustLR) {
    this(useRobustLR, 1.0, Props.SOFTMAX_GAMMA);
  }
  
  public OneVsAllRelationExtractor(boolean useRobustLR, double sigma, double gamma) {
    super();
    this.useRobustLR = useRobustLR;
    this.sigma = sigma;
    this.gamma = gamma;
  }

  @Override
  public Counter<Pair<String, Maybe<KBPRelationProvenance>>> classifyRelations(SentenceGroup input, Maybe<CoreMap[]> rawSentence) {
    // TODO(gabor) A deeper rewrite than splicing in classifyMentions()
    return RelationClassifier.firstProvenance(classifyMentions(RelationClassifier.tupleToFeatureList(input)), input);
  }

  public Counter<String> classifyMentions(List<Collection<String>> relation) {
    assert(classifiers != null);

    Counter<String> labels = new ClassicCounter<String>();
    for(Collection<String> mention: relation) {
      // System.err.println("Classifying slot " + mention.mention().getArg(1).getExtentString());
      Datum<String, String> datum = new BasicDatum<String, String>(mention);
      Pair<String, Double> label = annotateDatum(datum);
      if(! label.first().equals(RelationMention.UNRELATED)) {
        // System.err.println("Classified slot " + mention.mention().getArg(1).getExtentString() + " with label " + label.first() + " with score " + label.second());
        labels.incrementCount(label.first(), label.second());
      }
    }
    
    Counters.normalize(labels);
    return labels;
  }

  private Pair<String, Double> annotateDatum(Datum<String, String> testDatum) {
    Set<String> knownLabels = classifiers.keySet();
    
    // fetch all scores 
    List<Pair<String, Double>> allLabelScores = new ArrayList<Pair<String,Double>>();
    List<Double> scores = new ArrayList<Double>();
    for(String knownLabel: knownLabels){
      ProbabilisticClassifier<String, String> labelClassifier = classifiers.get(knownLabel);
      Pair<String, Double> pred = classOf(testDatum, labelClassifier);  
      if (pred != null) { // null if No Relation has all the weight
        if(pred.second > 0.5) allLabelScores.add(pred);
        scores.add(pred.second);
      }
    }
    
    // convert scores to probabilities using softmax
    for(Pair<String, Double> ls: allLabelScores){
      ls.second = softmax(ls.second, scores, gamma);
    }
    
    Collections.sort(allLabelScores, new Comparator<Pair<String, Double>>() {
      @Override
      public int compare(Pair<String, Double> o1, Pair<String, Double> o2) {
        if(o1.second > o2.second) return -1;
        if(o1.second.equals(o2.second)) return 0;
        return 1;
      }
    });
    
    if(allLabelScores.size() > 0) return allLabelScores.iterator().next();
    return new Pair<String, Double>(RelationMention.UNRELATED, 1.0);
  }
  
  private Pair<String, Double> classOf(Datum<String, String> datum, ProbabilisticClassifier<String, String> classifier) {
    Counter<String> probs = classifier.probabilityOf(datum); 
    List<Pair<String, Double>> sortedProbs = Counters.toDescendingMagnitudeSortedListWithCounts(probs);
    for(Pair<String, Double> ls: sortedProbs){
      if(! ls.first.equals(RelationMention.UNRELATED)) return ls;
    }
    return null;
  }
  
  @Override
  public void save(ObjectOutputStream out) throws IOException {
    assert(classifiers != null);
    out.writeObject(classifiers);
    out.writeDouble(sigma);
  }

  @Override
  public void load(ObjectInputStream in) throws IOException, ClassNotFoundException {
    Map<String, ProbabilisticClassifier<String, String>> classifiers =
        ErasureUtils.uncheckedCast(in.readObject());
    double sigma = in.readDouble();
    this.classifiers = classifiers;
    this.sigma = sigma;
    in.close();
  }

  public static OneVsAllRelationExtractor load(String modelPath) throws IOException, ClassNotFoundException {
    return RelationClassifier.load(modelPath, new Properties(), OneVsAllRelationExtractor.class);
  }

  public TrainingStatistics train(Map<String, GeneralDataset<String, String>> trainSets){
    Set<String> labels = trainSets.keySet();
    startTrack(BOLD, BLUE, "Training " + labels.size() + " models");
    
    classifiers = new HashMap<String, ProbabilisticClassifier<String,String>>();
    for(String label: labels) {
      startTrack("Training classifier for label: " + label);
      Redwood.log("Train set size: " + trainSets.get(label).size());
      ProbabilisticClassifier<String,String> labelClassifier = trainOne(trainSets.get(label));
      classifiers.put(label, labelClassifier);
      endTrack("Training classifier for label: " + label);
    }
    endTrack("Training " + labels.size() + " models");
    // Return TODO(gabor) training statistics
    statistics = Maybe.Just(TrainingStatistics.undefined());
    return TrainingStatistics.undefined();
  }

  @Override
  public TrainingStatistics train(KBPDataset<String, String> dataset) {
    Map<String, GeneralDataset<String, String>> trainSets =
        new HashMap<String, GeneralDataset<String, String>>();

    // create a binary dataset for each relation
    for (String relation : dataset.labelIndex) {
      assert RelationType.fromString(relation).orCrash().canonicalName.equals(relation);
      Dataset<String, String> binaryDataset = new Dataset<String, String>();
      for (int i = 0; i < dataset.size(); i++) {
        List<Datum<String, String>> group = dataset.getDatumGroup(i);
        
        String label;
        if (dataset.getPositiveLabels(i).contains(relation)) {
          label = relation;
        } else if (Props.TRAIN_LR_ALLNEGATIVES || dataset.getNegativeLabels(i).contains(relation)) {
          label = RelationMention.UNRELATED;
        } else {
          continue;
        }
        
        for (Datum<String, String> datum : group) {
          binaryDataset.add(new BasicDatum<String, String>(datum.asFeatures(), label));
        }
      }
      if (binaryDataset.size() > 0) {
        trainSets.put(relation, binaryDataset);
      }
    }
    return train(trainSets);
  }

  private ProbabilisticClassifier<String, String> trainOne(GeneralDataset<String, String> trainSet) {
    if ( !useRobustLR) {
      LinearClassifierFactory<String, String> factory = 
          new LinearClassifierFactory<String, String>(1e-4, false, sigma);
      factory.setVerbose(false); 
      //factory.useHybridMinimizer();
      //factory.useInPlaceStochasticGradientDescent();
      return factory.trainClassifier(trainSet);
    }

    throw new IllegalStateException("Missing implementing robust classifier!");
    
//    LogPrior prior = new LogPrior(LogPrior.LogPriorType.QUADRATIC, 1.0, 0.1);
//    ShiftParamsLogisticClassifierFactory<String, String> factory =
//      new ShiftParamsLogisticClassifierFactory<String, String>(prior, 0.01);
//
//    return factory.trainClassifier(trainSet);
  }

}
