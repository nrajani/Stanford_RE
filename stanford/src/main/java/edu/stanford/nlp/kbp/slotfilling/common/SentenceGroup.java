package edu.stanford.nlp.kbp.slotfilling.common;

import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.ling.BasicDatum;
import edu.stanford.nlp.ling.Datum;

import java.io.PrintStream;
import java.io.Serializable;
import java.security.NoSuchAlgorithmException;
import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.debug;

/**
 * Stores all the datums that have identified <entity, slot> pairs.
 * This is used to construct a dataset.
 * It also keeps track of the provenance information
 */
@SuppressWarnings("UnusedDeclaration")
public class SentenceGroup extends AbstractList<Datum<String,String>> implements Serializable, Comparable<SentenceGroup> {
  private static final long serialVersionUID = 1L;

  /**
   * Identifying (entity, slot) value
   */
  public final KBPair key;

  /**
   * A list of mentions retrieved from IR, and featurized into Datums.
   */
  private final List<Datum<String,String>> datums;
  /**
   * A list of (index,docid) that tracks the source of the above datums
   */
  public final List<KBPRelationProvenance> provenances;

  public final Maybe<? extends List<String>> sentenceGlossKeys;

  /** For reflection only! */
  @SuppressWarnings("UnusedDeclaration")
  private SentenceGroup() {
    this.key = null;
    this.datums = new ArrayList<Datum<String, String>>();
    this.provenances = new ArrayList<KBPRelationProvenance>();
    this.sentenceGlossKeys = Maybe.Nothing();
    assert !this.sentenceGlossKeys.isDefined() || datums.size() == this.sentenceGlossKeys.get().size();
  }

  public static SentenceGroup empty(KBPair key) {
    List<Datum<String, String>> datums = new ArrayList<Datum<String, String>>();
    ArrayList<KBPRelationProvenance> provenances = new ArrayList<KBPRelationProvenance>();
    ArrayList<String> sentenceGlossKeys = new ArrayList<String>();
    return new SentenceGroup(key, datums, provenances, sentenceGlossKeys );
  }

  public SentenceGroup(KBPair key, List<Datum<String, String>> datums, List<KBPRelationProvenance> provenances) {
    this.key = key;
    this.datums = datums;
    this.provenances = provenances;
    this.sentenceGlossKeys = Maybe.Nothing();
    assert !this.sentenceGlossKeys.isDefined() || datums.size() == this.sentenceGlossKeys.get().size();
  }
  public SentenceGroup(KBPair key, Datum<String, String> datum, KBPRelationProvenance provenance) {
    this.key = key;
    this.datums = new ArrayList<Datum<String,String>>();
    this.datums.add( datum );
    this.provenances = new ArrayList<KBPRelationProvenance>();
    this.provenances.add( provenance );
    this.sentenceGlossKeys = Maybe.Nothing();
    assert !this.sentenceGlossKeys.isDefined() || datums.size() == this.sentenceGlossKeys.get().size();
  }
  public SentenceGroup(KBPair key, List<Datum<String, String>> datums, List<KBPRelationProvenance> provenances, List<String> sentenceGlossKeys) {
    assert(sentenceGlossKeys.size() == datums.size());

    this.key = key;
    this.datums = datums;
    this.provenances = provenances;
    this.sentenceGlossKeys =  Maybe.Just(sentenceGlossKeys);
    assert !this.sentenceGlossKeys.isDefined() || datums.size() == this.sentenceGlossKeys.get().size();
  }
  public SentenceGroup(KBPair key, Datum<String, String> datum, KBPRelationProvenance provenance, String sentenceGlossKey) {
    this.key = key;
    this.datums = new ArrayList<Datum<String,String>>();
    this.datums.add( datum );
    this.provenances = new ArrayList<KBPRelationProvenance>();
    this.provenances.add( provenance );
    this.sentenceGlossKeys = Maybe.Just(new ArrayList<String>());
    this.sentenceGlossKeys.get().add( sentenceGlossKey );
    assert !this.sentenceGlossKeys.isDefined() || datums.size() == this.sentenceGlossKeys.get().size();
  }


  public String getSentenceGlossKey( int idx ) {
    assert !sentenceGlossKeys.isDefined() || datums.size() == sentenceGlossKeys.get().size();
    if (this.sentenceGlossKeys.isDefined())
      return sentenceGlossKeys.get().get(idx);
    else
      return null;
  }

  public String setSentenceGlossKey( int idx, String hexKey ) throws NoSuchAlgorithmException {
    assert (this.sentenceGlossKeys.isDefined());
    assert !sentenceGlossKeys.isDefined() || datums.size() == sentenceGlossKeys.get().size();
    return sentenceGlossKeys.get().set( idx, hexKey);
  }

  @Override
  public Datum<String,String> get( int idx ) {
    return datums.get( idx );
  }
  @Override
  public Datum<String,String> set( int idx, Datum<String,String> datum ) {
    return datums.set( idx, datum );
  }

  @Override
  public int size() {
    return datums.size();
  }
  @Override
  public void add( int idx, Datum<String,String> datum ) {
    assert !sentenceGlossKeys.isDefined();
    datums.add(idx, datum);
    assert !sentenceGlossKeys.isDefined() || datums.size() == sentenceGlossKeys.get().size();
  }

  public void add(Datum<String, String> datum, KBPRelationProvenance provenance, String hexKey) {
    assert (this.sentenceGlossKeys.isDefined());
    datums.add(datum);
    provenances.add(provenance);
    sentenceGlossKeys.get().add(hexKey);
    assert !sentenceGlossKeys.isDefined() || datums.size() == sentenceGlossKeys.get().size();
  }

  public void add(Datum<String, String> datum, KBPRelationProvenance provenance) {
    assert (this.sentenceGlossKeys.isDefined());
    datums.add(datum);
    provenances.add(provenance);
    if (sentenceGlossKeys.isDefined()) {
      sentenceGlossKeys.get().add("--no sentencegloss key--");
    }
    assert !sentenceGlossKeys.isDefined() || datums.size() == sentenceGlossKeys.get().size();
  }

  @Override
  public Datum<String,String> remove( int idx ) {
    if (this.sentenceGlossKeys.isDefined()) {
      sentenceGlossKeys.get().remove(idx);
    }
    Datum<String, String> rtn =  datums.remove( idx );
    assert !sentenceGlossKeys.isDefined() || datums.size() == sentenceGlossKeys.get().size();
    return rtn;
  }

  public KBPRelationProvenance getProvenance( int idx ) {
    assert !sentenceGlossKeys.isDefined() || datums.size() == sentenceGlossKeys.get().size();
    return provenances.get( idx );
  }

  public void merge( SentenceGroup other ) {
    //assert other.key.equals( this.key );
    if (sentenceGlossKeys.isDefined() && datums.size() != sentenceGlossKeys.get().size()) {
      throw new IllegalStateException("Sentence gloss key size doesn't match datums size (for this)!");
    }
    if (other.sentenceGlossKeys.isDefined() && other.sentenceGlossKeys.get().size() != other.datums.size()) {
      throw new IllegalStateException("Sentence gloss key size doesn't match datums size (for argument)!");
    }

    this.datums.addAll( other.datums );
    this.provenances.addAll( other.provenances );

    if (this.sentenceGlossKeys.isDefined()) {
      assert other.sentenceGlossKeys.isDefined();
      assert other.sentenceGlossKeys.get().size() == other.datums.size();
      this.sentenceGlossKeys.get().addAll(other.sentenceGlossKeys.get());
      assert datums.size() == sentenceGlossKeys.get().size();
    }
  }

  public SentenceGroup removeDuplicateDatums() {
    // Variables
    KBPair key = this.key;
    List<Datum<String,String>> datums = new ArrayList<Datum<String, String>>(this.datums.size());
    List<KBPRelationProvenance> provenances = new ArrayList<KBPRelationProvenance>(this.datums.size());
    Maybe<? extends List<String>> sentenceGlossKeys = Maybe.Nothing();
    if (this.sentenceGlossKeys.isDefined()) {
      sentenceGlossKeys = Maybe.Just(new ArrayList<String>(this.datums.size()));
    }

    // Copy
    Set<Datum<String, String>> datumSet = new HashSet<Datum<String, String>>();
    for (int i = 0; i < size(); ++i) {
      if (!datumSet.contains(this.datums.get(i))) {
        datums.add(this.datums.get(i));
        if (this.provenances.size() == this.datums.size()) {
          provenances.add(this.provenances.get(i));
        }
        if (this.sentenceGlossKeys.isDefined()) {
          assert sentenceGlossKeys.isDefined();
          sentenceGlossKeys.get().add(this.sentenceGlossKeys.get().get(i));
        }
      }
      datumSet.add(this.datums.get(i));
    }

    // Create
    if (this.provenances.size() != this.datums.size()) {
      provenances = this.provenances;
    }
    if (sentenceGlossKeys.isDefined()) {
      return new SentenceGroup(key, datums, provenances, sentenceGlossKeys.get());
    } else {
      return new SentenceGroup(key, datums, provenances);
    }
  }

  /**
   * Adapted from MinimalDatum.saveDatum() so that we can save to the old datum format
   */
  @Deprecated
  public void writeToStream(PrintStream os) {
    String id = key.entityId.orCrash();

    for (Datum<String, String> datum : this.datums) {
      os.print("{" + id + "} " +
          key.entityType.toXMLRepresentation().replaceAll("\\s+", "") + " " +
          key.slotType.orCrash().name.replaceAll("\\s+", "") + " " +
          key.slotValue.replaceAll("\\s+", "_") + " " +
          datum.label());
      Collection<String> feats = datum.asFeatures();
      for(String feat: feats){
        feat = feat.replaceAll("\\s+", "_");
        os.print(" " + feat);
      }
      os.println();
    }
  }

  /**
   * Adapted from MinimalDatum.lineToDatum() so that we can read the old datum format.
   * IMPORTANT NOTE: Don't ask for entity names or provenances from the resulting sentence group!
   */
  @Deprecated
  public static SentenceGroup readFromLine(String line) {
    // the entity id is stored at the beginning within "{...} "
    int entEnd = line.indexOf("} ");
    if(entEnd < 2) throw new RuntimeException("Invalid datum line: " + line);
    String eid = line.substring(1, entEnd);
    line = line.substring(entEnd + 2);

    String [] bits = line.split("\\s+");
    String entType = bits[0];
    String neType = bits[1];
    String slotValue = bits[2];
    String concatenatedLabel = bits[3];
    String [] labels = concatenatedLabel.split("\\|");
    if(labels.length > 1){
      debug("Found concatenated label: " + concatenatedLabel);
    }
    Collection<String> feats = new LinkedList<String>();
    feats.addAll(Arrays.asList(bits).subList(4, bits.length));
    List<Datum<String,String>> datums = new ArrayList<Datum<String, String>>();
    List<KBPRelationProvenance> provenances = new ArrayList<KBPRelationProvenance>();
    for(String label: labels){
      datums.add(new BasicDatum<String, String>(feats, label));
      provenances.add(null);
    }
    KBPair key = KBPNew.entName("???").entType(NERTag.fromString(entType).orCrash()).entId(eid).slotValue(slotValue).KBPair();
    return new SentenceGroup(key, datums, provenances);
  }

  @Override
  public String toString() {
    return "SentenceGroup [key=" + key + ", datums=" + datums
        + ", provenances=" + provenances + ", sentenceGlossKeys="
        + sentenceGlossKeys + "]";
  }

  public SentenceGroup filterFeature(String featureToExclude) {
    SentenceGroup toReturn = SentenceGroup.empty(key);
    for (int i = 0; i < this.size(); ++i) {
      Datum datum = this.get(i);
      if (!datum.asFeatures().contains(featureToExclude)) {
        if (sentenceGlossKeys.isDefined()) {
          toReturn.add(datums.get(i), provenances.get(i), sentenceGlossKeys.get().get(i));
        } else {
          toReturn.add(datums.get(i), provenances.get(i));
        }
      }
    }
    return toReturn;
  }

  @Override
  public int compareTo(SentenceGroup o) {
    int firstPass = this.key.compareTo(o.key);
    if (firstPass == 0) {
      return new Integer(this.datums.hashCode()).compareTo(o.datums.hashCode());
    } else {
      return firstPass;
    }
  }
}
