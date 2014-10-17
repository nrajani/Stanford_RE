package edu.stanford.nlp.kbp.slotfilling.common;

/**
 * A Knowledge-Base Triple stores (entity, relation, entity) triples with auxillary IDs and types.
 * This class should be used as a key wherever datums for such a triple need to be referenced.
 * E.g. (Julie, country_of_birth, Canada).
 */
public class KBTriple extends KBPair {
  private static final long serialVersionUID = 2L;

  /** The most common slot fill type */
  public final String relationName;

  /** For reflection only! */
  @SuppressWarnings("UnusedDeclaration")
  private KBTriple() {
    super();
    this.relationName = "(null)";
  }

  /** The (internal) constructor for a KBTriple */
  protected KBTriple(
      Maybe<String> entityId,
      String entityName,
      NERTag entityType,
      String relationName,
      String slotValue,
      Maybe<NERTag> slotType) {
    super(entityId, entityName, entityType, slotValue, slotType);
    this.relationName = relationName;
    assert this.entityId != null;
    assert this.entityName != null;
    assert this.entityType != null;
    assert this.relationName != null;
    assert this.slotType != null;
    assert this.slotValue != null;
  }

  public RelationType kbpRelation() {
    return RelationType.fromString(relationName).orCrash(relationName);
  }

  public Maybe<RelationType> tryKbpRelation() {
    return RelationType.fromString(relationName);
  }

  public boolean hasKBPRelation() {
    return RelationType.isKBPRelation(relationName);
  }

  public boolean equals(KBTriple other) {
    return this.getEntity().equals(other.getEntity())
        && relationName.equals(other.relationName)
        && other.slotValue.equals(slotValue);
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof KBTriple && equals((KBTriple) other);
  }

  @Override
  public int hashCode() {
    if (entityId.isDefined()) {
      return entityId.get().hashCode() + 31 * slotValue.hashCode() + 691 * relationName.hashCode();
    } else {
      return entityName.hashCode() + 31 * entityType.toXMLRepresentation().hashCode()
          + 961 * slotValue.hashCode() + 29791 * relationName.hashCode();
    }
  }

  @Override
  public String toString() {
    return relationName.replaceAll("\\s+", "_") + "(" + (entityId.isDefined() ? "*" : "") + entityName + ":" + entityType + ", " + slotValue + ":" + slotType.getOrElse(null) +")";
  }

  /** For stable ordering only; no meaningful natural order otherwise */
  @Override
  public int compareTo(KBPair o) {
    int superJudgment = super.compareTo(o);
    if (o instanceof KBTriple && superJudgment == 0) {
      return relationName.compareTo(((KBTriple) o).relationName);
    } else {
      return superJudgment;
    }
  }

}
