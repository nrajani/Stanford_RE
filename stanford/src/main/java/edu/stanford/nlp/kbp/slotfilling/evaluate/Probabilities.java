package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.kbp.slotfilling.common.KBPOfficialEntity;
import edu.stanford.nlp.kbp.slotfilling.common.RelationType;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;

import java.util.Collection;

import static edu.stanford.nlp.util.logging.Redwood.Util.debug;
import static edu.stanford.nlp.util.logging.Redwood.Util.warn;

/**
 * A class for manipulating probabilities.
 * Naive approximations are marked with TODOs
 *
 * @author Gabor Angeli
 */
public class Probabilities {

  public final KBPIR querier;
  public final double ofRelationGivenEntityAndSlotValue;
  @SuppressWarnings("FieldCanBeLocal")
  private final Collection<String> allSlotValues;

  public Probabilities(KBPIR querier,
                       Collection<String> allSlotValues,
                       double ofRelationGivenEntityAndSlotValue) {
    this.querier = querier;
    this.allSlotValues = allSlotValues;
    this.ofRelationGivenEntityAndSlotValue = ofRelationGivenEntityAndSlotValue;
    // Error checks
    if (this.ofRelationGivenEntityAndSlotValue < 0 || this.ofRelationGivenEntityAndSlotValue > 1.0) {
      warn("Probability of relation given entity and slot fill is not between 0 and 1: " + this.ofRelationGivenEntityAndSlotValue);
    }
    if (this.allSlotValues.isEmpty()) {
      throw new IllegalArgumentException("Of course there are slot fills. How else did you get P(r | e1. *slot_fill*) ?");

    }
  }

  public double ofSlotValueGivenRelationAndEntity(String slotFill, RelationType rel, KBPOfficialEntity entity) {
    // TODO(gabor) this is still ignoring the retreived sentences entirely...
    double prob = ofRelationGivenEntityAndSlotValue * ofSlotValueGivenEntity(slotFill, entity) / ofRelationGivenEntity(rel, entity);
    if (prob < 0.0 || prob > 1.0) {
      debug("Math failed: " +
          ofRelationGivenEntityAndSlotValue + " * " + ofSlotValueGivenEntity(slotFill, entity) + " / " + ofRelationGivenEntity(rel, entity));
    }
    return prob;
  }

  public double ofRelation(RelationType rel) {
    return rel.priorProbability;
  }

  public double ofRelationGivenEntity(RelationType rel, KBPOfficialEntity entity) {
    // TODO(gabor) Could probably be better -- currently it's basically just the prior
    switch (entity.type) {
      case PERSON:
        return ofRelation(rel) / ofRelationTypePERSON;
      case ORGANIZATION:
        return ofRelation(rel) / ofRelationTypeORGANIZATION;
      default:
        throw new IllegalArgumentException("Unknown entity type: " + entity);
    }
  }

  public double ofSlotValueGivenEntity(String slotFill, KBPOfficialEntity entity) {
    // TODO(gabor) Naive approximation...
//    return 1.0 / ((double) allSlotValues.size());
    return 1.0;
  }


  /** The total prior probability of relations of for the PERSON type */
  public static final double ofRelationTypePERSON;
  /** The total prior probability of relations of for the ORGANIZATION type */
  public static double ofRelationTypeORGANIZATION;

  static {
    double sumPerProb = 0.0;
    double sumOrgProb = 0.0;

    for (RelationType rel : RelationType.values()) {
      switch (rel.entityType) {
        case PERSON:
          sumPerProb += rel.priorProbability;
          break;
        case ORGANIZATION:
          sumOrgProb += rel.priorProbability;
          break;
        default:
          throw new IllegalStateException("Unknown relation entity type: " + rel.entityType);
      }
    }

    ofRelationTypePERSON = sumPerProb;
    ofRelationTypeORGANIZATION = sumOrgProb;
  }



}
