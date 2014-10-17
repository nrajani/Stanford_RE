
package edu.stanford.nlp.kbp.slotfilling.common;

import java.util.HashMap;
import java.util.Map;

/**
 * Different kinds of entities you might want to search for in the KBP
 * data.  Other packages may behave differently depending on the
 * entity type you request.
 */
public enum EntityType {
  PERSON          ("PER"), 
  ORGANIZATION    ("ORG"),
  NIL             ("NIL");

  private final String xmlRepresentation;

  EntityType(String xml) {
    this.xmlRepresentation = xml;
  }

  private static final Map<String, EntityType> xmlToEntity = 
    new HashMap<String, EntityType>();
  static {
    for (EntityType entity : values()) {
      xmlToEntity.put(entity.xmlRepresentation, entity);
    }
  }

  static public EntityType fromXmlRepresentation(String xml) {
    return xmlToEntity.get(xml);
  }

  public static Maybe<EntityType> fromRelation(String relation) {
    assert( relation.length() >= 3 );
    relation = relation.toLowerCase();
    if (relation.startsWith("per:")) return Maybe.Just(EntityType.PERSON);
    else if (relation.startsWith("org:")) return Maybe.Just(EntityType.ORGANIZATION);
    else return Maybe.Nothing();
  }

  public static EntityType fromRelation(RelationType relation) {
    return fromRelation(relation.toString()).orCrash(); // Can safely assume it won't crash
  }

}
