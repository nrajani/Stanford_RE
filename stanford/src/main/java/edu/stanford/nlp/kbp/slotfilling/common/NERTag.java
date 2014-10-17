package edu.stanford.nlp.kbp.slotfilling.common;

/**
* An expanded set of NER tags
*
* @author Gabor Angeli
*/
public enum NERTag {
  CAUSE_OF_DEATH    ("CAUSE_OF_DEATH",    "COD"), // note: these names must be upper case
  CITY              ("CITY",              "CIT"), //       furthermore, DO NOT change the short names, or else serialization may break
  COUNTRY           ("COUNTRY",           "CRY"),
  CRIMINAL_CHARGE   ("CRIMINAL_CHARGE",   "CC"),
  DATE              ("DATE",              "DT"),
  IDEOLOGY          ("IDEOLOGY",          "IDY"),
  LOCATION          ("LOCATION",          "LOC"),
  MISC              ("MISC",              "MSC"),
  MODIFIER          ("MODIFIER",          "MOD"),
  NATIONALITY       ("NATIONALITY",       "NAT"),
  NUMBER            ("NUMBER",            "NUM"),
  ORGANIZATION      ("ORGANIZATION",      "ORG"),
  PERSON            ("PERSON",            "PER"),
  RELIGION          ("RELIGION",          "REL"),
  STATE_OR_PROVINCE ("STATE_OR_PROVINCE", "ST"),
  TITLE             ("TITLE",             "TIT"),
  URL               ("URL",               "URL"),
  DURATION          ("DURATION",          "DUR"),
  ;

  public final String name;
  public final String shortName;
  NERTag(String name, String shortName){ this.name = name; this.shortName = shortName; }

  /** Find the slot for a given name */
  public static Maybe<NERTag> fromString(String name) {
    // Early termination
    if (name == null || name.equals("")) { return Maybe.Nothing(); }
    // Cycle known NER tags
    name = name.toUpperCase();
    for (NERTag slot : NERTag.values()) {
      if (slot.name.equals(name)) return Maybe.Just(slot);
    }
    for (NERTag slot : NERTag.values()) {
      if (slot.shortName.equals(name)) return Maybe.Just(slot);
    }
    // DEFT types
    // Some quick fixes
    return Maybe.Nothing();
  }

  public static Maybe<NERTag> fromDEFTString(String name) {
    Maybe<NERTag> tag = fromString(name);
    if (tag.isDefined()) { return tag; }
    if (name.equalsIgnoreCase("NAME")) { return Maybe.Just(NERTag.PERSON); }
    return Maybe.Nothing();
  }

  /** Find the slot for a given name */
  public static Maybe<NERTag> fromShortName(String name) {
    // Early termination
    if (name == null) { return Maybe.Nothing(); }
    // Cycle known NER tags
    name = name.toUpperCase();
    for (NERTag slot : NERTag.values()) {
      if (slot.shortName.startsWith(name)) return Maybe.Just(slot);
    }
    // Some quick fixes
    return Maybe.Nothing();
  }


  public String toXMLRepresentation() {
    switch (this) {
      case PERSON: return "PER";
      case ORGANIZATION: return "ORG";
      default: return name;
    }
  }

  public static Maybe<NERTag> fromRelation(String relation) {
    assert( relation.length() >= 3 );
    relation = relation.toLowerCase();
    if (relation.startsWith("per:")) return Maybe.Just(PERSON);
    else if (relation.startsWith("org:")) return Maybe.Just(ORGANIZATION);
    else return Maybe.Nothing();
  }

  public boolean isEntityType() {
    return this.equals(PERSON) || this.equals(ORGANIZATION);
  }
}
