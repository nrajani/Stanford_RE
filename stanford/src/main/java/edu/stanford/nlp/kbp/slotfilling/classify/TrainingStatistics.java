package edu.stanford.nlp.kbp.slotfilling.classify;

import edu.stanford.nlp.kbp.slotfilling.common.Maybe;
import edu.stanford.nlp.kbp.slotfilling.common.Utils;
import edu.stanford.nlp.kbp.slotfilling.process.KBPProcess;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.*;

import java.io.Serializable;
import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * Relevant statistics returned from training a classifier.
 * Not all statistics must be returned by every classifier.
 *
 * @author Gabor Angeli
 */
public class TrainingStatistics implements Serializable {
  private static final long serialVersionUID = 1L;

  public static enum ActiveLearningSelectionCriterion {
    HIGH_KL_FROM_MEAN,
    LOW_AVERAGE_CONFIDENCE,
    RANDOM_UNIFORM
  }

  public static class SentenceKey implements Serializable {
    private static final long serialVersionUID = 1L;
    public final String sentenceHash;  // Really, it would be nice to store more information than this

    public SentenceKey(String sentenceHash) {
      this.sentenceHash = sentenceHash;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof SentenceKey)) return false;
      SentenceKey that = (SentenceKey) o;
      return sentenceHash.equals(that.sentenceHash);
    }

    @Override
    public int hashCode() {
      return sentenceHash.hashCode();
    }
  }



  public static class SentenceStatistics implements Serializable {
    private static final long serialVersionUID = 1L;
    /** The confidence of this classifier in the validity of the relation distribution */
    public final Maybe<Double> confidence;
    /** The distribution over relations as output by the classifier, p(z_s | x, y) */
    public final Counter<String> relationDistribution;

    // note that these constructors expect standard probabilities
    public SentenceStatistics(Counter<String> relationDistribution, double confidence) {
      this.confidence = Maybe.Just(confidence);
      this.relationDistribution = relationDistribution;
    }

    public SentenceStatistics(Counter<String> relationDistribution) {
      this.confidence = Maybe.Nothing();
      this.relationDistribution = relationDistribution;
    }
  }



  public static class EnsembleStatistics implements Serializable {
    private static final long serialVersionUID = 1L;
    /** The sentence statistics aggregated over potentially multiple classifiers */
    public final Collection<SentenceStatistics> statisticsForClassifiers;

    public EnsembleStatistics(Collection<SentenceStatistics> statisticsForClassifiers) {
      this.statisticsForClassifiers = statisticsForClassifiers;
    }

    public EnsembleStatistics(EnsembleStatistics toClone) {
      this.statisticsForClassifiers = new LinkedList<SentenceStatistics>(toClone.statisticsForClassifiers);
    }

    public void addInPlace(SentenceStatistics stats) {
      statisticsForClassifiers.add(stats);
    }

    public void addInPlace(EnsembleStatistics ensembleStatistics) {
      for (SentenceStatistics x : ensembleStatistics.statisticsForClassifiers) { addInPlace(x); }
    }

    public SentenceStatistics mean() {
      double sumConfidence = 0;
      int countWithConfidence = 0;
      Counter<String> avePredictions = new ClassicCounter<String>(MapFactory.<String, MutableDouble>linkedHashMapFactory());
      // Sum
      for (SentenceStatistics stat : this.statisticsForClassifiers) {
        for (Double confidence : stat.confidence) {
          sumConfidence += confidence;
          countWithConfidence += 1;
        }
        assert Math.abs(stat.relationDistribution.totalCount() - 1.0) < 1e-5;
        for (Map.Entry<String, Double> entry : stat.relationDistribution.entrySet()) {
          assert entry.getValue() >= 0.0;
          assert entry.getValue() == stat.relationDistribution.getCount(entry.getKey());
          avePredictions.incrementCount(entry.getKey(), entry.getValue());
          assert stat.relationDistribution.getCount(entry.getKey()) == stat.relationDistribution.getCount(entry.getKey());
        }
      }
      // Normalize
      double aveConfidence = sumConfidence / ((double) countWithConfidence);
      // Return
      if (this.statisticsForClassifiers.size() > 1) { Counters.divideInPlace(avePredictions, (double) this.statisticsForClassifiers.size()); }
      if (Math.abs(avePredictions.totalCount() - 1.0) > 1e-5) {
        throw new IllegalStateException("Mean relation distribution is not a distribution!");
      }
      assert this.statisticsForClassifiers.size() > 1 || this.statisticsForClassifiers.size() == 0 ||
          Counters.equals(avePredictions, statisticsForClassifiers.iterator().next().relationDistribution, 1e-5);
      return countWithConfidence > 0 ? new SentenceStatistics(avePredictions, aveConfidence)  : new SentenceStatistics(avePredictions);
    }

    public double averageKLFromMean() {
      Counter<String> mean = this.mean().relationDistribution;
      double sumKL = 0;
      for (SentenceStatistics stats : this.statisticsForClassifiers) {
        double kl = Counters.klDivergence(stats.relationDistribution, mean);
        if (kl < 0.0 && kl > -1e-12) { kl = 0.0; }  // floating point error.
        assert kl >= 0.0;
        sumKL += kl;
      }
      double val = sumKL / ((double) this.statisticsForClassifiers.size());
      if (Double.isInfinite(val) || Double.isNaN(val) || val < 0.0) {
        throw new AssertionError("Invalid average KL value: " + val);
      }
      assert val >= 0.0;  // KL lower bound
      assert this.statisticsForClassifiers.size() > 1 || val < 1e-5;
      if (val < 1e-10) { val = 0.0; }  // floating point error
      return val;
    }
  }



  private final Maybe<? extends Map<SentenceKey, EnsembleStatistics>> impl;

  public TrainingStatistics(Maybe<? extends Map<SentenceKey, EnsembleStatistics>> impl) {
    this.impl = impl;
  }

  public void addInPlace(SentenceKey key, SentenceStatistics sentenceStatistics) {
    for (Map<SentenceKey, EnsembleStatistics> impl : this.impl) {
      EnsembleStatistics stats = impl.get(key);
      if (stats == null) {
        stats = new EnsembleStatistics(new LinkedList<SentenceStatistics>());
        impl.put(key, stats);
      }
      stats.addInPlace(sentenceStatistics);
    }
  }
  
  public Maybe<Set<SentenceKey>> getSentenceKeys() {
    if (! impl.isDefined()) return Maybe.Nothing();
    return Maybe.Just(impl.get().keySet());
  }
  
  public TrainingStatistics merge(TrainingStatistics other) {
    Map<SentenceKey, EnsembleStatistics> newStats = new HashMap<SentenceKey, EnsembleStatistics>();
    // Add elements from this statistics
    for( Map<SentenceKey, EnsembleStatistics> map : this.impl) {
      for (SentenceKey key : map.keySet()) {
        newStats.put(key, new EnsembleStatistics(map.get(key)));
      }
    }
    // Add elements from other statistics
    for( Map<SentenceKey, EnsembleStatistics> map : other.impl) {
      for (SentenceKey key : map.keySet()) {
        EnsembleStatistics existing = newStats.get(key);
        if (existing == null) {
          existing = new EnsembleStatistics(new LinkedList<SentenceStatistics>());
          newStats.put(key, existing);
        }
        existing.addInPlace(map.get(key));
      }
    }
    // Return
    return new TrainingStatistics(Maybe.Just(newStats));
  }
  
  private IterableIterator<Pair<CoreMap,Counter<String>>> iterableFromList(final KBPProcess process,
      final List<String> hashes) {
    // Create Iterator
    return new IterableIterator<Pair<CoreMap,Counter<String>>>(new Iterator<Pair<CoreMap,Counter<String>>>() {
      Iterator<String> iter = hashes.iterator();
      Maybe<Pair<CoreMap,Counter<String>>> nextGloss = Maybe.Nothing();
      @Override
      public boolean hasNext() {
        if (nextGloss.isDefined()) { return true; }
        while (iter.hasNext() && !nextGloss.isDefined()) {
          String key = iter.next();
          Maybe<CoreMap> nextSentence = process.recoverSentenceGloss(key);
          if (nextSentence.isDefined()) {
            nextGloss = Maybe.Just(Pair.makePair(process.recoverSentenceGloss(key).get(), relationPredictionsForKey(key)));
          } else {
            nextGloss = Maybe.Nothing();
          }
          if (!nextGloss.isDefined()) { warn("No sentence for gloss key: " + key); }
        }
        return nextGloss.isDefined();
      }
      @Override
      public Pair<CoreMap,Counter<String>> next() {
        if (!hasNext()) { throw new NoSuchElementException(); }
        assert nextGloss.isDefined();
        Pair<CoreMap,Counter<String>> gloss = nextGloss.get();
        nextGloss = Maybe.Nothing();
        return gloss;
      }
      @Override
      public void remove() { throw new UnsupportedOperationException(); }
    });
  }

  private Counter<String> highKLFromMean() {
    // Get confidences
    Counter<String> kl = new ClassicCounter<String>(MapFactory.<String, MutableDouble>linkedHashMapFactory());
    for( Map<SentenceKey, EnsembleStatistics> impl : this.impl ) {
      for ( Map.Entry<SentenceKey, EnsembleStatistics> entry : impl.entrySet() ) {
        kl.setCount(entry.getKey().sentenceHash, entry.getValue().averageKLFromMean());
      }
    }

    return kl;
  }
  
  private Counter<String> lowAverageConfidence() {
    // Get confidences
    Counter<String> lowConfidence = new ClassicCounter<String>(MapFactory.<String, MutableDouble>linkedHashMapFactory());
    for( Map<SentenceKey, EnsembleStatistics> impl : this.impl ) {
      for ( Map.Entry<SentenceKey, EnsembleStatistics> entry : impl.entrySet() ) {
        SentenceStatistics average = entry.getValue().mean();
        for (double confidence : average.confidence) {
          lowConfidence.setCount(entry.getKey().sentenceHash, 1 - confidence);
        }
      }
    }
    return lowConfidence;
  }
  
  private Counter<String> uniformRandom() {
    Counter<String> uniformRandom = new ClassicCounter<String>(MapFactory.<String, MutableDouble>linkedHashMapFactory());
    for( Map<SentenceKey, EnsembleStatistics> impl : this.impl ) {
      for ( Map.Entry<SentenceKey, EnsembleStatistics> entry : impl.entrySet() ) {
        uniformRandom.setCount(entry.getKey().sentenceHash, 1.0);
      }
    }

    return uniformRandom;
  }
  
  public IterableIterator<Pair<CoreMap,Counter<String>>> selectExamples(KBPProcess process,
      ActiveLearningSelectionCriterion criterion) {
    List<String> examples = selectKeys(criterion);
    return iterableFromList(process, examples);
  }

  public List<String> selectKeys(ActiveLearningSelectionCriterion criterion) {
    Counter<String> weights = uncertainty(criterion);
    return Counters.toSortedList(weights);
  }

  public List<Pair<String,Double>> selectWeightedKeys(ActiveLearningSelectionCriterion criterion) {
    Counter<String> weights = uncertainty(criterion);
    return Counters.toSortedListWithCounts(weights);
  }

  // for each sentence, returns a non-negative weight representing its uncertainty (higher
  // means more uncertain)
  // (public only for testing)
  public Counter<String> uncertainty(ActiveLearningSelectionCriterion criterion) {
    switch (criterion) {
      case HIGH_KL_FROM_MEAN:
        return highKLFromMean();
      case LOW_AVERAGE_CONFIDENCE:
        return lowAverageConfidence();
      case RANDOM_UNIFORM:
        return uniformRandom();
      default:
        throw new IllegalArgumentException("Unknown selection criterion: " + criterion);
    }
  }
  
  // draws "count" number of weighted samples, with weights determined by the
  // given selection criterion
  // (public only for testing)
  public List<String> selectKeysWithSampling(ActiveLearningSelectionCriterion criterion, int numSamples, int seed) {
    final List<Pair<String, Double>> examples = selectWeightedKeysWithSampling(criterion, numSamples, seed);
    return new AbstractList<String>() {
      @Override
      public String get(int index) {
        return examples.get(index).first;
      }
      @Override
      public int size() {
        return examples.size();
      }
    };
  }

  public List<Pair<String, Double>> selectWeightedKeysWithSampling(ActiveLearningSelectionCriterion criterion, int numSamples, int seed) {
    List<Pair<String,Double>> result = new ArrayList<Pair<String,Double>>();
    forceTrack("Sampling Keys");
    log("" + numSamples + " to collect");

    // Get uncertainty
    forceTrack("Computing Uncertainties");
    Counter<String> weightCounter = uncertainty(criterion);
    assert weightCounter.equals(uncertainty(criterion));
    endTrack("Computing Uncertainties");
    // Compute some statistics
    double totalCount = weightCounter.totalCount();
    Random random = new Random(seed);

    // Flatten counter
    List<String> keys = new LinkedList<String>();
    List<Double> weights = new LinkedList<Double>();
    List<String> zeroUncertaintyKeys = new LinkedList<String>();
    for (Pair<String, Double> elem : Counters.toSortedListWithCounts(weightCounter, new Comparator<Pair<String, Double>>() {
          @Override
          public int compare(Pair<String, Double> o1, Pair<String, Double> o2) {
                int value = o1.compareTo(o2);
                if (value == 0) {
                  return o1.first.compareTo(o2.first);
                } else {
                  return value;
                }
              }
        })) {
      if (elem.second != 0.0) {  // ignore 0 probability weights
        keys.add(elem.first);
        weights.add(elem.second);
      } else {
        zeroUncertaintyKeys.add(elem.first);
      }
    }

    // Error check
    if (Utils.assertionsEnabled()) {
      for (Double elem : weights) {
        if (!(elem >= 0 && !Double.isInfinite(elem) && !Double.isNaN(elem))) {
          throw new IllegalArgumentException("Invalid weight: " + elem);
        }
      }
    }

    // Sample
    SAMPLE_ITER: for (int i = 1; i <= numSamples; ++i) {  // For each sample
      if (i % 1000 == 0) {
        // Debug log
        log("sampled " + (i/1000) + "k keys");
        // Recompute total count to mitigate floating point errors
        totalCount = 0.0;
        for (double val : weights) { totalCount += val; }
      }
      assert totalCount >= 0.0;
      assert weights.size() == keys.size();
      double target = random.nextDouble() * totalCount;
      Iterator<String> keyIter = keys.iterator();
      Iterator<Double> weightIter = weights.iterator();
      double runningTotal = 0.0;
      while (keyIter.hasNext()) {  // For each candidate
        String key = keyIter.next();
        double weight = weightIter.next();
        runningTotal += weight;
        if (target <= runningTotal) {  // Select that sample
          assert weight > 0.0;
          result.add(Pair.makePair(key, weight));
          keyIter.remove();
          weightIter.remove();
          totalCount -= weight;
          continue SAMPLE_ITER; // continue sampling
        }
      }
      // We should get here only if the keys list is empty
      warn("No more uncertain samples left to draw from! (target=" + target + " totalCount=" + totalCount + " size=" + keys.size());
      assert keys.size() == 0;
      assert zeroUncertaintyKeys.size() > 0;
      result.add(Pair.makePair(zeroUncertaintyKeys.remove(0), 0.0));
    }

    endTrack("Sampling Keys");
    return result;
  }

  @SuppressWarnings("UnusedDeclaration")
  public IterableIterator<Pair<CoreMap, Counter<String>>> selectExamplesWithSampling(KBPProcess process,
      ActiveLearningSelectionCriterion criterion, int numSamples, int seed) {
    List<String> hashes = selectKeysWithSampling(criterion, numSamples, seed);
    return iterableFromList(process, hashes);
  }

  public Counter<String> relationPredictionsForKey(String hexKey) {
    if (!impl.isDefined()) { throw new IllegalArgumentException("Training statistics is not defined"); }
    return impl.get().get(new SentenceKey(hexKey)).mean().relationDistribution;
  }

  /**
   * Run some sanity checks on the training statistics, to make sure they look valid.
   */
  public void validate() {
    for (Map<SentenceKey, EnsembleStatistics> map : impl) {
      for (EnsembleStatistics stats : map.values()) {
        for (SentenceStatistics component : stats.statisticsForClassifiers) {
          assert !Counters.isUniformDistribution(component.relationDistribution, 1e-5);
          Counters.normalize(component.relationDistribution);  // TODO(gabor) this shouldn't be necessary
          assert (Math.abs(component.relationDistribution.totalCount() - 1.0)) < 1e-5;
        }
        assert (Math.abs(stats.mean().relationDistribution.totalCount() - 1.0)) < 1e-5;
        assert !Counters.isUniformDistribution(stats.mean().relationDistribution, 1e-5);
      }
    }
  }

  public static TrainingStatistics empty() {
    return new TrainingStatistics(Maybe.Just(new HashMap<SentenceKey, EnsembleStatistics>()));
  }

  public static TrainingStatistics undefined() {
    return new TrainingStatistics(Maybe.<Map<SentenceKey, EnsembleStatistics>>Nothing());
  }

}
