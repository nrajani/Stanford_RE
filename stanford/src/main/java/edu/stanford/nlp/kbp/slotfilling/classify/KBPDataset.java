package edu.stanford.nlp.kbp.slotfilling.classify;

import edu.stanford.nlp.kbp.slotfilling.common.Maybe;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;
import edu.stanford.nlp.util.HashIndex;
import edu.stanford.nlp.util.Index;
import edu.stanford.nlp.util.Pair;

import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.err;

public class KBPDataset<L, F> implements Serializable {
  private static final long serialVersionUID = 1L;
  
  public final Index<L> labelIndex;
  private  Index<F> featureIndex;

  /** Stores the list of known positive labels for each datum group
   * Array indexes into the examples in the dataset; the Set is the set of positive labels
   * for a given example.
   */
  protected Set<Integer> [] posLabels;      // Positive labels used for training
  protected Set<Integer> [] origPosLabels;  // Original set of positive labels

  /** Stores the list of known negative labels for each datum group
  * Array indexes into the examples in the dataset; the Set is the set of negative labels
  * for a given example.
  */
  protected Set<Integer> [] negLabels;      // Negative labels used for training
  protected Set<Integer> [] origNegLabels;  // Original set of negative labels

  /** Stores the list of labels for each datum group for which we are not certain
   *   if it is actually positive or negative
   * Array indexes into the examples in the dataset; the Set is the set of unknown labels
   * for a given example.
   * Note that this set can overlap with the posLabels and negLabels if we want to have
   *   some initial guesses for what label it should actually be
   */
  protected Set<Integer> [] unkLabels;

  /** Stores the datum groups, where each group consists of a collection of datums
   * The indices are:
   * - For each example (group)...
   * - For each sentence in the example...
   * - For each feature in the sentence...
   *   -> Integer value of feature.
   */
  protected int[][][] data;

  /** Partial labels for the data, if known. For memory efficiency, this is stored as a sparse map
   * This is effectively a sparse matrix, keyed on the example index and the sentence index.
   */
  protected final Map<Pair<Integer,Integer>, L>  annotatedLabels = new HashMap<Pair<Integer,Integer>, L>();

  /**
   * Keys (in a compressed Hex Hash format)
   * for recovering the sentence gloss for a particular (example, sentence) pair.
   */
  public String[][] sentenceGlossKeys;

  /** The size of the dataset, in examples */
  protected int size;

  public KBPDataset() {
    this(10);
  }
  
  @SuppressWarnings("unchecked")
  public KBPDataset(int numGroups) {
    labelIndex = new HashIndex<L>();
    featureIndex = new HashIndex<F>();
    posLabels = new Set[numGroups];
    negLabels = new Set[numGroups];
    unkLabels = new Set[numGroups];
    data = new int[numGroups][][];
    sentenceGlossKeys = new String[numGroups][];
    size = 0;
  }
  
  @SuppressWarnings("unchecked")
  public KBPDataset(Index<F> featureIndex, Index<L> labelIndex) {
    this.featureIndex = featureIndex;
    this.labelIndex = labelIndex;
    posLabels = new Set[10];
    negLabels = new Set[10];
    unkLabels = new Set[10];
    data = new int[10][][];
    sentenceGlossKeys = new String[10][];
    size = 0;
  }
  
  public KBPDataset(int[][][] data,
                    Index<F> featureIndex,
                    Index<L> labelIndex,
                    Set<Integer>[] posLabels,
                    Set<Integer>[] negLabels,
                    Set<Integer>[] unkLabels,
                    Maybe<L>[][] annotatedLabels,
                    String[][] sentenceGlossKeys) {
    this.data = data;
    this.featureIndex = featureIndex;
    this.labelIndex = labelIndex;
    this.posLabels = posLabels;
    this.negLabels = negLabels;
    this.unkLabels = unkLabels;
    this.sentenceGlossKeys = sentenceGlossKeys;
    this.size = data.length;
    // Set Labels
    for (int i=0; i<annotatedLabels.length; ++i) {
      for (int j=0; j<annotatedLabels[i].length; ++j) {
        for (L label : annotatedLabels[i][j]) {
          this.annotatedLabels.put(Pair.makePair(i,j), label);
        }
      }
    }
  }
  
  public int size() { return size; }

  public Index<F> featureIndex() { return featureIndex; }
  
  public Index<L> labelIndex() { return labelIndex; }

  public int numFeatures() { return featureIndex.size(); }

  public int numClasses() { return labelIndex.size(); }
  
  public Set<Integer> [] getPositiveLabelsArray() {
    posLabels = trimToSize(posLabels);
    return posLabels;
  }
  
  public Set<L> getPositiveLabels(int i) {
    Set<Integer> positiveIndices = posLabels[i];
    return getLabels(positiveIndices, i);
  }
  
  public Set<Integer> [] getNegativeLabelsArray() {
    negLabels = trimToSize(negLabels);
    return negLabels;
  }
  
  public Set<L> getNegativeLabels(int i) {
    Set<Integer> negativeIndices = negLabels[i];
    return getLabels(negativeIndices, i);
  }

  public Set<Integer> [] getUnknownLabelsArray() {
    unkLabels = trimToSize(unkLabels);
    return unkLabels;
  }

  // Make a copy of the positive/negative labels and save it
  public void finalizeLabels() {
    posLabels = trimToSize(posLabels);
    negLabels = trimToSize(negLabels);
    origPosLabels = copyLabels(posLabels);
    origNegLabels = copyLabels(negLabels);
  }

  // Restores the positive/negative label from the original
  public void restoreLabels() {
    posLabels = copyLabels(origPosLabels);
    negLabels = copyLabels(origNegLabels);
  }

  public Set<Integer> [] getOrigPosLabelsArray() {
    // Make a copy of the positive labels
    if (origPosLabels == null) {
      origPosLabels = copyLabels(posLabels);
    }
    return origPosLabels;
  }

  public Set<Integer> [] getOrigNegLabelsArray() {
    // Make a copy of the negative labels
    if (origNegLabels == null) {
      origNegLabels = copyLabels(negLabels);
    }
    return origNegLabels;
  }

  public Set<L> getUnknownLabels(int i) {
    Set<Integer> unkIndices = unkLabels[i];
    return getLabels(unkIndices, i);
  }

  private Set<Integer>[] copyLabels(Set<Integer>[] labels) {
    Set<Integer>[] newLabels = new Set[labels.length];
    for (int i = 0; i < labels.length; i++) {
      if (labels[i] != null) {
        newLabels[i] = new HashSet<Integer>();
        newLabels[i].addAll(labels[i]);
      }
    }
    return newLabels;
  }

  private Set<L> getLabels(Set<Integer> indices, int i) {
    Set<L> result = new HashSet<L>();

    for (int index : indices)
      result.add(labelIndex.get(index));

    return result;
  }

  public int[][][] getDataArray() {
    data = trimToSize(data);
    return data;
  }

  public List<Datum<L, F>> getDatumGroup(int i) {
    List<Datum<L, F>> result = new ArrayList<Datum<L, F>>();
    int[][] sentences = data[i];
    for (int[] sentence : sentences) {
      Collection<F> features = featureIndex.objects(sentence);
      result.add(new BasicDatum<L, F>(features));
    }
    return result;
  }

  public int getNumSentencesInGroup(int i) {
    return data[i].length;
  }
  
  @SuppressWarnings("unchecked")
  protected <E> E[] trimToSize(E[] i) {
    if (i == null) { return null; }  // I hate null so very much.
    if(i.length == size) return i;
    E[] newI = (E[]) Array.newInstance(i.getClass().getComponentType(), size);  // Java is impossible without magic.
    System.arraycopy(i, 0, newI, 0, size);
    return newI;
  }

  private <T> void swap(T[] array, int i, int j) {
    T tmp = array[i];
    array[i] = array[j];
    array[j] = tmp;
  }

  /**
   * Randomizes the data array in place
   * @param randomSeed The random seed to randomize by.
   */
  public void randomize(int randomSeed) {
    Random rand = new Random(randomSeed);
    for(int j = size - 1; j > 0; j --){
      int randIndex = rand.nextInt(j);

      swap(data, randIndex, j);
      swap(posLabels, randIndex, j);
      swap(negLabels, randIndex, j);
      swap(unkLabels, randIndex, j);
      if (origPosLabels != null) swap(origPosLabels, randIndex, j);
      if (origNegLabels != null) swap(origNegLabels, randIndex, j);
    }
  }

  public void randomize(int [][] zLabels, int randomSeed) {
    Random rand = new Random(randomSeed);
    for(int j = size - 1; j > 0; j --){
      int randIndex = rand.nextInt(j);

      swap(zLabels, randIndex, j);
      swap(data, randIndex, j);
      swap(posLabels, randIndex, j);
      swap(negLabels, randIndex, j);
      swap(unkLabels, randIndex, j);
      if (origPosLabels != null) swap(origPosLabels, randIndex, j);
      if (origNegLabels != null) swap(origNegLabels, randIndex, j);
    }
  }
  
  /**
   * Get the total count (over all data instances) of each feature
   *
   * @return an array containing the counts (indexed by index)
   */
  public float[] getFeatureCounts() {
    float[] counts = new float[featureIndex.size()];
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < data[i].length; j++) {
        for(int k = 0; k < data[i][j].length; k ++) {
          counts[data[i][j][k]] += 1.0;
        }
      }
    }
    return counts;
  }

  /**
   * Applies a feature count threshold to the KBPDataset.
   * All features that occur fewer than <i>threshold</i> times are expunged.
   */
  public void applyFeatureCountThreshold(int threshold) {
    data = trimToSize(data);
    float[] counts = getFeatureCounts();
    
    //
    // rebuild the feature index
    //
    Index<F> newFeatureIndex = new HashIndex<F>();
    int[] featMap = new int[featureIndex.size()];
    for (int i = 0; i < featMap.length; i++) {
      F feat = featureIndex.get(i);
      if (counts[i] >= threshold) {
        int newIndex = newFeatureIndex.size();
        newFeatureIndex.add(feat);
        featMap[i] = newIndex;
      } else {
        featMap[i] = -1;
      }
    }

    featureIndex = newFeatureIndex;

    //
    // rebuild the data
    //
    for (int i = 0; i < size; i++) {
      for(int j = 0; j < data[i].length; j ++){
        List<Integer> featList = new ArrayList<Integer>(data[i][j].length);
        for (int k = 0; k < data[i][j].length; k++) {
          if (featMap[data[i][j][k]] >= 0) {
            featList.add(featMap[data[i][j][k]]);
          }
        }
        data[i][j] = new int[featList.size()];
        for(int k = 0; k < data[i][j].length; k ++) {
          data[i][j][k] = featList.get(k);
        }
      }
    }
  }
  
//  public void addDatum(Set<L> yPos, Set<L> yNeg, List<Datum<L, F>> group, Maybe<? extends List<String>> sentenceGlossKeys) {
//    addDatum(yPos, yNeg, new HashSet<L>(0), group, sentenceGlossKeys);
//  }
//
  public void addDatum(Set<L> yPos, Set<L> yNeg, Set<L> yUnk, List<Datum<L, F>> group, Maybe<? extends List<String>> sentenceGlossKeys) {
    List<Collection<F>> features = new ArrayList<Collection<F>>();
    for(Datum<L, F> datum: group){
      features.add(datum.asFeatures());
    }
    add(yPos, yNeg, yUnk, features, sentenceGlossKeys);
  }

  private void add(Set<L> yPos, Set<L> yNeg, Set<L> yUnk, List<Collection<F>> group,
                   Maybe<? extends List<String>> sentenceGlossKeys) {
    if (sentenceGlossKeys.isDefined() && group.size() != sentenceGlossKeys.get().size()) {
      err("Sentence gloss keys don't match datum keys length!");
      return;
    }

    ensureSize();
    
    addPosLabels(yPos);
    addNegLabels(yNeg);
    addUnkLabels(yUnk);
    addFeatures(group);
    for (List<String> keys : sentenceGlossKeys) {
      assert (keys.size() == group.size());
      addSentenceGlossKeys(keys);
    }

    size++;
  }

  /**
   * Add a datum from its already interned form. This is generally quicker than adding it datum by datum,
   * though a copy of the datum still has to be made (to avoid conflicts when shuffling the data in parallel
   * classifiers).
   * @param yPos The positive labels to add, already indexed by this dataset's {@link KBPDataset#labelIndex}.
   * @param yNeg The negative labels to add, already indexed by this dataset's {@link KBPDataset#labelIndex}.
   * @param yUnk The unlabeled labels to add, already indexed by this dataset's {@link KBPDataset#labelIndex}.
   * @param group The datum group to add, already featurized and indexed by this dataset's {@link KBPDataset#featureIndex}.
   * @param keys The sentence gloss keys associated with this datum.
   */
  public void addDatum(Set<Integer> yPos, Set<Integer> yNeg, Set<Integer> yUnk, int[][] group, String[] keys) {
    ensureSize();
    
    addPosLabelIndices(new HashSet<Integer>(yPos));
    addNegLabelIndices(new HashSet<Integer>(yNeg));
    addUnkLabelIndices(new HashSet<Integer>(yUnk));
    int[][] copiedGroup = new int[group.length][];
    for (int i = 0; i < group.length; ++i) {
      copiedGroup[i] = new int[group[i].length];
      System.arraycopy(group[i], 0, copiedGroup[i], 0, group[i].length);
    }
    addFeatureIndices(copiedGroup);
    String[] copiedKeys = new String[keys.length];
    System.arraycopy(keys, 0, copiedKeys, 0, keys.length);
    addSentenceGlossKeys(copiedKeys);
    
    size++;
  }

  protected void addFeatures(List<Collection<F>> group) {
    int [][] groupFeatures = new int[group.size()][];
    int datumIndex = 0;
    for(Collection<F> features: group){
      int[] intFeatures = new int[features.size()];
      int j = 0;
      for (F feature : features) {
        featureIndex.add(feature);
        int index = featureIndex.indexOf(feature);
        if (index >= 0) {
          intFeatures[j] = featureIndex.indexOf(feature);
          j++;
        }
      }
      
      int [] trimmedFeatures = new int[j];
      System.arraycopy(intFeatures, 0, trimmedFeatures, 0, j);
      groupFeatures[datumIndex] = trimmedFeatures;
      datumIndex ++;
    }
    assert(datumIndex == group.size());
    data[size] = groupFeatures;
  }
  
  protected void addFeatureIndices(int[][] group) {
    data[size] = group;
  }
  
  protected void addPosLabels(Set<L> labels) {
    labelIndex.addAll(labels);
    Set<Integer> newLabels = new HashSet<Integer>();
    for(L l: labels) {
      newLabels.add(labelIndex.indexOf(l));
    }
    posLabels[size] = newLabels;
  }
  
  protected void addPosLabelIndices(Set<Integer> labels) {
    posLabels[size] = labels;
  }
  
  protected void addNegLabels(Set<L> labels) {
    labelIndex.addAll(labels);
    Set<Integer> newLabels = new HashSet<Integer>();
    for(L l: labels) {
      newLabels.add(labelIndex.indexOf(l));
    }
    negLabels[size] = newLabels;
  }

  protected void addNegLabelIndices(Set<Integer> labels) {
    negLabels[size] = labels;
  }

  protected void addUnkLabels(Set<L> labels) {
    labelIndex.addAll(labels);
    Set<Integer> newLabels = new HashSet<Integer>();
    for(L l: labels) {
      newLabels.add(labelIndex.indexOf(l));
    }
    unkLabels[size] = newLabels;
  }

  protected void addUnkLabelIndices(Set<Integer> labels) {
    unkLabels[size] = labels;
  }

  protected void addSentenceGlossKeys(List<String> keys) {
    addSentenceGlossKeys(keys.toArray(new String[keys.size()]));
  }
  
  protected void addSentenceGlossKeys(String[] keys) {
    sentenceGlossKeys[size] = keys;
  }

  @SuppressWarnings("unchecked")
  protected void ensureSize() {
    if (posLabels.length == size) {
      Set<Integer> [] newLabels = new Set[size * 2];
      System.arraycopy(posLabels, 0, newLabels, 0, size);
      this.posLabels = newLabels;
      
      newLabels = new Set[size * 2];
      System.arraycopy(negLabels, 0, newLabels, 0, size);
      this.negLabels = newLabels;

      newLabels = new Set[size * 2];
      System.arraycopy(unkLabels, 0, newLabels, 0, size);
      this.unkLabels = newLabels;

      int[][][] newData = new int[size * 2][][];
      System.arraycopy(data, 0, newData, 0, size);
      this.data = newData;

      String[][] newKeys = new String[size * 2][];
      System.arraycopy(this.sentenceGlossKeys, 0, newKeys, 0, size);
      this.sentenceGlossKeys = newKeys;
    }
  }

//  @SuppressWarnings("unchecked")
//  public Maybe<L>[][] getAnnotatedLabels() {
//    // Setup Variables
//    Pair<Integer, Integer> key = Pair.makePair(-1, -1);
//    Maybe<L>[][] annotatedLabels = (Maybe<L>[][]) new Maybe[size][];
//    // Create Structure
//    for (int groupI = 0; groupI < annotatedLabels.length; ++groupI) {
//      annotatedLabels[groupI] = (Maybe<L>[]) new Maybe[data[groupI].length];
//      for (int sentI = 0; sentI < annotatedLabels[groupI].length; ++sentI) {
//        // Get label (if it exists)
//        key.first = groupI;
//        key.second = sentI;
//        L label = this.annotatedLabels.get(key);
//        // Set label in array
//        if (label != null) {
//          annotatedLabels[groupI][sentI] = Maybe.Just(label);
//        } else {
//          annotatedLabels[groupI][sentI] = Maybe.Nothing();
//        }
//      }
//    }
//    // Return
//    return annotatedLabels;
//  }

  @SuppressWarnings("unchecked")
  public Maybe<L>[] getAnnotatedLabels(int groupI) {
    // Setup Variables
    Pair<Integer, Integer> key = Pair.makePair(-1, -1);
    // Create Structure
    Maybe<L>[] annotatedLabels = (Maybe<L>[]) new Maybe[data[groupI].length];
    for (int sentI = 0; sentI < annotatedLabels.length; ++sentI) {
      // Get label (if it exists)
      key.first = groupI;
      key.second = sentI;
      L label = this.annotatedLabels.get(key);
      // Set label in array
      if (label != null) {
        annotatedLabels[sentI] = Maybe.Just(label);
      } else {
        annotatedLabels[sentI] = Maybe.Nothing();
      }
    }
    // Return
    return annotatedLabels;
  }

  public String[] getSentenceGlossKey(int i) {
    return sentenceGlossKeys[i];
  }

  public String getSentenceGlossKey(int group, int sentence) {
    return sentenceGlossKeys[group][sentence];
  }

  public int countLabels(Set<Integer>[] labels) {
    int count = 0;
    for (int i = 0; i < labels.length; i++) {
      count+=labels[i].size();
    }
    return count;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof KBPDataset)) return false;

    KBPDataset that = (KBPDataset) o;

    if (size != that.size) return false;
    if (!annotatedLabels.equals(that.annotatedLabels))
      return false;
    if (!labelIndex.equals(that.labelIndex)) return false;
    if (!featureIndex.equals(that.featureIndex)) return false;
    if (!Arrays.equals(trimToSize(negLabels), trimToSize(that.negLabels))) return false;
    if (!Arrays.equals(trimToSize(posLabels), trimToSize(that.posLabels))) return false;
    if (!Arrays.equals(trimToSize(unkLabels), trimToSize(that.unkLabels))) return false;
    if (!Arrays.equals(trimToSize(origNegLabels), trimToSize(that.origNegLabels))) return false;
    if (!Arrays.equals(trimToSize(origPosLabels), trimToSize(that.origPosLabels))) return false;

    if (this.size != that.size) { return false; }
    for (int i = 0; i < this.size; ++i) {
      if (data[i].length != that.data[i].length) { return false; }
      for (int j = 0; j < data[i].length; ++j) {
        if (data[i][j].length != that.data[i][j].length) { return false; }
        for (int k = 0; k < data[i][j].length; ++k) {
          if (data[i][j][k] != that.data[i][j][k]) { return false; }
        }
      }
    }

    return true;
  }

  @Override
  public int hashCode() {
    int result = labelIndex.hashCode();
    result = 31 * result + featureIndex.hashCode();
    result = 31 * result + Arrays.hashCode(posLabels);
    result = 31 * result + (origPosLabels != null ? Arrays.hashCode(origPosLabels) : 0);
    result = 31 * result + Arrays.hashCode(negLabels);
    result = 31 * result + (origNegLabels != null ? Arrays.hashCode(origNegLabels) : 0);
    result = 31 * result + (unkLabels != null ? Arrays.hashCode(unkLabels) : 0);
    result = 31 * result + (annotatedLabels.hashCode());
    result = 31 * result + size;
    return result;
  }
}
