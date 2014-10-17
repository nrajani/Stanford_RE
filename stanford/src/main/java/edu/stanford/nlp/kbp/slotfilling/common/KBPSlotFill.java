package edu.stanford.nlp.kbp.slotfilling.common;

//import adept.common.Sentence;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;

import java.io.Serializable;

/** Data structure representing a relation mention  */
public class KBPSlotFill implements Serializable, Comparable<KBPSlotFill> {

  private static final long serialVersionUID = 1L;

  /**
   * The core of the slot fill, containing the entity, relation, and slot value.
   */
  public final KBTriple key;

  /**
   * An optional provenance associated iwth the slot fill. This must be filled by the time
   * evaluation comes along, but can be blank at any intermediate time.
   */
  public  Maybe<KBPRelationProvenance> provenance;

  /**
   * Score assigned to this relation by the predictor
   * This score is assigned after conflict resolution so it may not be a probability anymore
   */
  public final Maybe<Double> score;

 // public Sentence sourceDEFTSentence = null;

  /** Private constructor for reflection only! */
  @SuppressWarnings("UnusedDeclaration")
  private KBPSlotFill() {
    this.key = null;
    this.provenance = Maybe.Nothing();
    this.score = Maybe.Nothing();
  }

  /** The (internal) constructor for a KBPSlotFill. This is really a "family" constructor */
  protected KBPSlotFill(KBTriple key, Maybe<KBPRelationProvenance> provenance,
                     Maybe<Double> score) {
    this.key = key;
    this.provenance = provenance;
    this.score = score;
    assert this.provenance != null;
    assert this.score != null;
    assert !this.provenance.isDefined() || this.provenance.get() != null;
    assert !this.score.isDefined() || this.score.get() != null;
  }

  @Override
  public String toString() {
    return key.entityName + ", " + key.relationName + ": " + key.slotValue;
  }

  @Override
  public int compareTo(KBPSlotFill o) {
    for (Double myScore : score) {
      //noinspection LoopStatementThatDoesntLoop
      for (Double theirScore : o.score) {
        if (myScore < theirScore) return 1;
        else if (myScore > theirScore) return -1;
        else {
          String[] thisSlotValueTokens = CoreMapUtils.tokenizeToStrings(this.key.slotValue);
          String[] otherSlotValueTokens = CoreMapUtils.tokenizeToStrings(o.key.slotValue);
          // Same score. This is particularly relevant for rule-based extractions, which are not scored
          if (thisSlotValueTokens.length > otherSlotValueTokens.length) { return -1; }
          else if (thisSlotValueTokens.length < otherSlotValueTokens.length) { return 1; }
          else {
            // Same score, same (or similar slot fill)
            if (this.key.entityName.length() > o.key.entityName.length()) { return -1; }
            else if (this.key.entityName.length() < o.key.entityName.length()) { return 1; }
            else return this.key.toString().compareTo(o.key.toString());  // Stable tiebreaking
          }
        }
      }
    }
    throw new IllegalStateException("Sorting KBPSlotFills without scores!");
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof KBPSlotFill)) return false;

    KBPSlotFill that = (KBPSlotFill) o;

    return key.equals(that.key);

  }

  @Override
  public int hashCode() {
    return key.hashCode();
  }

  public String fillToString() {
    return key.relationName + ": " + key.slotValue;
  }
}
