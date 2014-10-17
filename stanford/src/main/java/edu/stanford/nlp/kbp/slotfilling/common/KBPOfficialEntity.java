package edu.stanford.nlp.kbp.slotfilling.common;

import edu.stanford.nlp.util.ArrayUtils;

import java.util.Comparator;
import java.util.HashSet;
import java.util.Set;

/**
 * <p>An official KBP query entity -- that is, the first argument to a triple (entity, relation, slotValue).</p>
 *
 * <p>
 *   Important Implementation Notes:
 *   <ul>
 *     <li><b>Equality:</b> To be equal to another KBPEntity, it must have the same name and type.
 *                          To be equal to another KBPOfficialEntity, it must have the same ID.
 *                          BUT: for hashing purposes, it must also have the same name, or else the hash code will not bucket
 *                               the entity correctly.</li>
 *   </ul>
 * </p>
 */
public class KBPOfficialEntity extends KBPEntity {
  private static final long serialVersionUID = 2L;

  public final Maybe<String> id;
  public final Maybe<String> queryId;
  public final Maybe<Set<RelationType>> ignoredSlots;
  public final Maybe<String> representativeDocument;

  protected KBPOfficialEntity(String name,
                           NERTag type,
                           Maybe<String> id,
                           Maybe<String> queryId,
                           Maybe<Set<RelationType>> ignoredSlots,
                           Maybe<String> representativeDocument) {
    super(name, type);
    if (!type.isEntityType()) { throw new IllegalArgumentException("Invalid entity type for official entity: " + type); }
    assert id != null;
    assert id.getOrElse("") != null;
    this.id = id;
    assert queryId != null;
    assert queryId.getOrElse("") != null;
    this.queryId = queryId;
    assert ignoredSlots != null;
    assert ignoredSlots.getOrElse(new HashSet<RelationType>()) != null;
    this.ignoredSlots = ignoredSlots;
    assert representativeDocument != null;
    assert queryId.getOrElse("") != null;
    this.representativeDocument = representativeDocument;
  }

  @Override
  public int hashCode() {
    return name.toLowerCase().hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if(obj instanceof KBPOfficialEntity){
      KBPOfficialEntity em = (KBPOfficialEntity) obj;
      // NOTE (arun): This used to assert that id != null earlier. I'm relaxing the constraint
      // to first check on id then others.
      if (id != null && em.id != null && id.isDefined() && em.id.isDefined()) {
        // Note: names must match if id's match; this is a safety check to make sure hashing works correctly
        if (Utils.assertionsEnabled() && id.get().equals(em.id.get()) && !name.toLowerCase().equals(em.name.toLowerCase())) {
          throw new AssertionError("Official entity with same id has different name: " + name + " vs " + em.name);
        }
        return id.get().equals(em.id.get());
      } else {
        if (!name.equals(em.name)) { return false; }
        if (!type.toXMLRepresentation().equals(em.type.toXMLRepresentation())) { return false; }
      }
      return true;
    } else if (obj instanceof KBPEntity) {
      KBPEntity em = (KBPEntity) obj;
      return type == em.type && name.equals(em.name);
    }
    return false;
  }

  @Override
  public String toString() {
    String s = type + ":" + name;
    if(id != null) s += " (" + id.getOrElse("<no id>") + "," + queryId.getOrElse("<no query id>") + ")";
    return s;
  }

  public static class QueryIdSorter implements Comparator<KBPOfficialEntity> {
    public int compare(KBPOfficialEntity first, KBPOfficialEntity second) {
      // queryId should be unique, so there shouldn't be a need for a
      // fallback comparison
      return first.queryId.orCrash().compareTo(second.queryId.orCrash());
    }
  }
  
  /**
   * Sort first alphabetically by entity name, then by query ID, then by entity ID, then by type.
   */
  @SuppressWarnings("UnusedDeclaration")
  public static class AlphabeticSorter implements Comparator<KBPOfficialEntity> {
    public int compare(KBPOfficialEntity first, KBPOfficialEntity second) {
      return ArrayUtils.compareArrays(extractFields(first), extractFields(second));
    }

    private String[] extractFields(KBPOfficialEntity entity) {
      String queryId = entity.queryId.getOrElse("(null)");
      String id = entity.id.getOrElse("(null)");
      return new String[] { entity.name, queryId, id, entity.type.toString() };
    }
  }
}
