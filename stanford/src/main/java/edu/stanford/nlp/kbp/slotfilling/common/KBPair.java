package edu.stanford.nlp.kbp.slotfilling.common;

import java.io.Serializable;

/**
 * A Knowledge-Base Pair stores (entity, entity) pairs with auxiliary IDs and types for the first entity.
 * This "key" is used in several places, e.g. to index things in a KBPDataset.
 * For example, this would store pairs such as (Julie, Canada) where Julie (e_1) is the entity and
 * Canada (e_2) is the slotValue.
 */
public class KBPair implements Serializable, Comparable<KBPair> {
  private static final long serialVersionUID = 1L;

  /** The cannonical entity name */
  public final String entityName;
  /** The entity's NER type */
  public final NERTag entityType;
  /**
   * The entity ID.
   * This is distinct from the entity's {@link KBPOfficialEntity#queryId}, which is provided only for test-time entities
   * and corresponds to the id given by LDC for the scoring scripts.
   */
  public final Maybe<String> entityId;
  /** The longest candidate slot value */
  public final String slotValue;

  /**
   * <p>The NER _TYPE_ of the slotValue.</p>
   *
   * <p>
   *   Although formally optional, certain components of the code (e.g., {@link edu.stanford.nlp.kbp.slotfilling.evaluate.InferentialSlotFiller})
   *   expect this to be filled, and will likely discard this item if it is not found.
   *
   *   The only real reason this should not be provided is if the NER in question does not fit into the types enumerated in
   *   {@link NERTag}, for example CURRENCY. These are datums that should really eventually be discarded anyways, as they can never
   *   be valid slot fills.
   * </p>
   *
   */
  public final Maybe<NERTag> slotType;

  /** For reflection only! */
  @SuppressWarnings("UnusedDeclaration")
  protected KBPair() {
    this.entityId = Maybe.Nothing();
    this.entityName = "(null)";
    this.entityType = null;
    this.slotType = Maybe.Nothing();
    this.slotValue = "(null)";
  }

  /** The (internal) constructor for a KBTriple */
  protected KBPair(
      Maybe<String> entityId,
      String entityName,
      NERTag entityType,
      String slotValue,
      Maybe<NERTag> slotType
  ) {
    this.entityId = entityId;
    this.entityName = entityName;
    this.entityType = entityType;
    this.slotValue = slotValue;
    this.slotType = slotType;
    assert this.entityId != null;
    assert this.entityName != null;
    assert this.entityType != null;
    assert this.slotValue != null;
    assert this.slotType != null;
  }

  public boolean equals(KBPair other) {
    return this.getEntity().equals(other.getEntity())
        && other.slotValue.equals(slotValue);
  }

  @Override
  public boolean equals(Object other) {
    if (other instanceof KBTriple) { return equals(KBPNew.from((KBTriple) other).KBPair()); }
    else { return equals((KBPair) other); }
  }

  @Override
  public int hashCode() {
    if (entityId.isDefined()) {
      return entityId.get().hashCode() + 31 * slotValue.hashCode();
    } else {
      return entityName.hashCode() + 31 * entityType.toXMLRepresentation().hashCode() + 961 * slotValue.hashCode();
    }
  }

  @Override
  public String toString() {
    return "(" + (entityId.isDefined() ? "*" : "") + entityName + ":" + entityType + ", " + slotValue + ":" + slotType.getOrElse(null) + ")";
  }


  /**
   * @see edu.stanford.nlp.kbp.slotfilling.common.KBPair#getEntity()
   */
  private volatile KBPEntity cachedEntityView = null;
  /**
   * @see edu.stanford.nlp.kbp.slotfilling.common.KBPair#getSlotEntity()
   */
  private volatile KBPEntity cachedSlotEntityView = null;

  /**
   * Returns the KBP Entity associated with this tuple.
   * Note that it in reality returns an official entity when possible.
   * However, it should not be treated as such as some of the fields may have gotten
   * lost (e.g., the query id).
   * The entity id is maintained if possible.
   */
  public KBPEntity getEntity() {
    KBPEntity rtn;
    // Try cache
    if (cachedEntityView != null) {
      return cachedEntityView;
    }
    // Recreate entity
    if ((this.entityType.equals(NERTag.PERSON) || this.entityType.equals(NERTag.ORGANIZATION)) &&
        this.entityId.isDefined()){  // case: likely an official entity
      rtn = KBPNew.from(this).KBPOfficialEntity();
    } else {
      rtn = KBPNew.from(this).KBPEntity();
    }
    // Return
    cachedEntityView = rtn;
    return rtn;
  }

  /**
   * Return the KBP Slot Fill associated with this tuple, as an Entity.
   * This will always return a {@link KBPEntity} rather than a {@link KBPOfficialEntity}.
   * @return The slot fill corresponding to this KBP slot fill, or Nothing if the slot type is undefined.
   */
  public Maybe<KBPEntity> getSlotEntity() {
    KBPEntity rtn = null;
    // Try Cache
    if (cachedSlotEntityView != null) {
      return Maybe.Just(cachedSlotEntityView);
    }
    // Recreate entity
    if (this.slotType.isDefined()) {
      rtn = KBPNew.entName(this.slotValue).entType(this.slotType.get()).KBPEntity();
    }
    // Return
    if (rtn != null) { cachedSlotEntityView = rtn; }
    return Maybe.fromNull(rtn);
  }

  /** For stable ordering only; no meaningful natural order otherwise */
  @Override
  public int compareTo(KBPair o) {
    if (this.entityId.isDefined() && o.entityId.isDefined()) { return this.entityId.get().compareTo(o.entityId.get()); }
    int entityCompare = this.entityName.compareTo(o.entityName);
    if (entityCompare != 0) { return entityCompare; }
    int entityTypeCompare = this.entityType.ordinal() - o.entityType.ordinal();
    if (entityTypeCompare != 0) { return entityTypeCompare; }
    int slotValueCompare = slotValue.compareTo(o.slotValue);
    if (slotValueCompare != 0) { return slotValueCompare; }
    if (slotType.isDefined() && o.slotType.isDefined()) { return slotType.get().ordinal() - o.slotType.get().ordinal(); }
    if (slotType.isDefined() && !o.slotType.isDefined()) { return 1; }
    if (!slotType.isDefined() && o.slotType.isDefined()) { return -1; }
    return 0; // give up -- they're the same entity
  }
}
