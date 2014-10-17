package edu.stanford.nlp.kbp.slotfilling.common;

import edu.stanford.nlp.kbp.slotfilling.evaluate.OfficialOutputWriter;

import java.util.*;

import static edu.stanford.nlp.kbp.slotfilling.common.NERTag.*;
import static edu.stanford.nlp.util.logging.Redwood.Util.fatal;

/**
 * Known relation types (last updated for the 2013 shared task).
 *
 * @author Gabor Angeli
 */
public enum RelationType {
  PER_ALTERNATE_NAMES                   ("per:alternate_names",                     NERTag.PERSON,       Cardinality.LIST,   new NERTag[]{ PERSON, MISC },                                                    new String[]{ "NNP" },      0.0353027270308107100 ),
  PER_CHILDREN                          ("per:children",                            NERTag.PERSON,       Cardinality.LIST,   new NERTag[]{ PERSON },                                                          new String[]{ "NNP" },      0.0058428110284504410 ),
  PER_CITIES_OF_RESIDENCE               ("per:cities_of_residence",                 NERTag.PERSON,       Cardinality.LIST,   new NERTag[]{ CITY, LOCATION },                                                  new String[]{ "NNP" },      0.0136105679675116560 ),
  PER_CITY_OF_BIRTH                     ("per:city_of_birth",                       NERTag.PERSON,       Cardinality.SINGLE, new NERTag[]{ CITY, LOCATION, STATE_OR_PROVINCE },                               new String[]{ "NNP" },      0.0358146961159769100 ),
  PER_CITY_OF_DEATH                     ("per:city_of_death",                       NERTag.PERSON,       Cardinality.SINGLE, new NERTag[]{ CITY, LOCATION, STATE_OR_PROVINCE },                               new String[]{ "NNP" },      0.0102003332137774650 ),
  PER_COUNTRIES_OF_RESIDENCE            ("per:countries_of_residence",              NERTag.PERSON,       Cardinality.LIST,   new NERTag[]{ COUNTRY, LOCATION },                                               new String[]{ "NNP" },      0.0107788293552082020 ),
  PER_COUNTRY_OF_BIRTH                  ("per:country_of_birth",                    NERTag.PERSON,       Cardinality.SINGLE, new NERTag[]{ COUNTRY, LOCATION, NATIONALITY },                                  new String[]{ "NNP" },      0.0223444134627622040 ),
  PER_COUNTRY_OF_DEATH                  ("per:country_of_death",                    NERTag.PERSON,       Cardinality.SINGLE, new NERTag[]{ COUNTRY, LOCATION, NATIONALITY },                                  new String[]{ "NNP" },      0.0060626395621941200 ),
  PER_EMPLOYEE_OF                       ("per:employee_of",                         NERTag.PERSON,       Cardinality.LIST,   new NERTag[]{ ORGANIZATION, COUNTRY, STATE_OR_PROVINCE },                        new String[]{ "NNP" },      0.0335281901169719200 ),
  PER_MEMBER_OF                         ("per:member_of",                           NERTag.PERSON,       Cardinality.LIST,   new NERTag[]{ ORGANIZATION },                                                    new String[]{ "NNP" },      0.0521716745149309900 ),
  PER_ORIGIN                            ("per:origin",                              NERTag.PERSON,       Cardinality.LIST,   new NERTag[]{ NATIONALITY, COUNTRY, LOCATION },                                  new String[]{ "NNP" },      0.0069795559463618380 ),
  PER_OTHER_FAMILY                      ("per:other_family",                        NERTag.PERSON,       Cardinality.LIST,   new NERTag[]{ PERSON },                                                          new String[]{ "NNP" },      2.7478566717959990E-5 ),
  PER_PARENTS                           ("per:parents",                             NERTag.PERSON,       Cardinality.LIST,   new NERTag[]{ PERSON },                                                          new String[]{ "NNP" },      0.0032222235077692030 ),
  PER_SCHOOLS_ATTENDED                  ("per:schools_attended",                    NERTag.PERSON,       Cardinality.LIST,   new NERTag[]{ ORGANIZATION },                                                    new String[]{ "NNP" },      0.0054696810172276150 ),
  PER_SIBLINGS                          ("per:siblings",                            NERTag.PERSON,       Cardinality.LIST,   new NERTag[]{ PERSON },                                                          new String[]{ "NNP" },      1.000000000000000e-99 ),
  PER_SPOUSE                            ("per:spouse",                              NERTag.PERSON,       Cardinality.LIST,   new NERTag[]{ PERSON },                                                          new String[]{ "NNP" },      0.0164075968113292680 ),
  PER_STATE_OR_PROVINCES_OF_BIRTH       ("per:stateorprovince_of_birth",            NERTag.PERSON,       Cardinality.SINGLE, new NERTag[]{ STATE_OR_PROVINCE, LOCATION, COUNTRY },                            new String[]{ "NNP" },      0.0165825918941120660 ),
  PER_STATE_OR_PROVINCES_OF_DEATH       ("per:stateorprovince_of_death",            NERTag.PERSON,       Cardinality.SINGLE, new NERTag[]{ STATE_OR_PROVINCE, LOCATION, COUNTRY },                            new String[]{ "NNP" },      0.0050083303444366030 ),
  PER_STATE_OR_PROVINCES_OF_RESIDENCE   ("per:stateorprovinces_of_residence",       NERTag.PERSON,       Cardinality.LIST,   new NERTag[]{ STATE_OR_PROVINCE, LOCATION, COUNTRY },                            new String[]{ "NNP" },      0.0066787379528178550 ),
  // Phrases like "one-year-old" is marked by SUTime as DURATION (TODO: Mark as age?)
  PER_AGE                               ("per:age",                                 NERTag.PERSON,       Cardinality.SINGLE, new NERTag[]{ NUMBER, DURATION },                                                new String[]{ "CD", "NN" }, 0.0483159977322951300 ),
  PER_DATE_OF_BIRTH                     ("per:date_of_birth",                       NERTag.PERSON,       Cardinality.SINGLE, new NERTag[]{ DATE },                                                            new String[]{ "CD", "NN" }, 0.0743584477791533200 ),
  PER_DATE_OF_DEATH                     ("per:date_of_death",                       NERTag.PERSON,       Cardinality.SINGLE, new NERTag[]{ DATE },                                                            new String[]{ "CD", "NN" }, 0.0189819046406960460 ),
  PER_CAUSE_OF_DEATH                    ("per:cause_of_death",                      NERTag.PERSON,       Cardinality.SINGLE, new NERTag[]{ CAUSE_OF_DEATH },                                                  new String[]{ "NN" },       1.0123682475037891E-5 ),
  PER_CHARGES                           ("per:charges",                             NERTag.PERSON,       Cardinality.LIST,   new NERTag[]{ CRIMINAL_CHARGE },                                                 new String[]{ "NN" },       3.8614617440501670E-4 ),
  PER_RELIGION                          ("per:religion",                            NERTag.PERSON,       Cardinality.SINGLE, new NERTag[]{ RELIGION },                                                        new String[]{ "NN" },       7.6650738739572610E-4 ),
  PER_TITLE                             ("per:title",                               NERTag.PERSON,       Cardinality.LIST,   new NERTag[]{ TITLE, ORGANIZATION, MODIFIER },                                   new String[]{ "NN" },       0.0334283995325751200 ),
  ORG_ALTERNATE_NAMES                   ("org:alternate_names",                     NERTag.ORGANIZATION, Cardinality.LIST,   new NERTag[]{ ORGANIZATION, MISC },                                              new String[]{ "NNP" },      0.0552058867767352000 ),
  ORG_CITY_OF_HEADQUARTERS              ("org:city_of_headquarters",                NERTag.ORGANIZATION, Cardinality.SINGLE, new NERTag[]{ CITY, LOCATION, STATE_OR_PROVINCE },                               new String[]{ "NNP" },      0.0555949254318473740 ),
  ORG_COUNTRY_OF_HEADQUARTERS           ("org:country_of_headquarters",             NERTag.ORGANIZATION, Cardinality.SINGLE, new NERTag[]{ COUNTRY, LOCATION, NATIONALITY },                                  new String[]{ "NNP" },      0.0580217167451493100 ),
  ORG_FOUNDED_BY                        ("org:founded_by",                          NERTag.ORGANIZATION, Cardinality.LIST,   new NERTag[]{ PERSON, ORGANIZATION },                                            new String[]{ "NNP" },      0.0050806423621154450 ),
  ORG_MEMBER_OF                         ("org:member_of",                           NERTag.ORGANIZATION, Cardinality.LIST,   new NERTag[]{ ORGANIZATION, LOCATION, COUNTRY, STATE_OR_PROVINCE },              new String[]{ "NNP" },      0.0396298781687126140 ),
  ORG_MEMBERS                           ("org:members",                             NERTag.ORGANIZATION, Cardinality.LIST,   new NERTag[]{ ORGANIZATION, COUNTRY },                                           new String[]{ "NNP" },      0.0012220730987724312 ),
  ORG_PARENTS                           ("org:parents",                             NERTag.ORGANIZATION, Cardinality.LIST,   new NERTag[]{ ORGANIZATION, LOCATION, COUNTRY, STATE_OR_PROVINCE },              new String[]{ "NNP" },      0.0550048593675880200 ),
  ORG_POLITICAL_RELIGIOUS_AFFILIATION   ("org:political/religious_affiliation",     NERTag.ORGANIZATION, Cardinality.LIST,   new NERTag[]{ IDEOLOGY, RELIGION },                                              new String[]{ "NN", "JJ" }, 0.0059266929689578970 ),
  ORG_SHAREHOLDERS                      ("org:shareholders",                        NERTag.ORGANIZATION, Cardinality.LIST,   new NERTag[]{ PERSON, ORGANIZATION },                                            new String[]{ "NNP" },      1.1569922828614734E-5 ),
  ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS("org:stateorprovince_of_headquarters",     NERTag.ORGANIZATION, Cardinality.SINGLE, new NERTag[]{ STATE_OR_PROVINCE, LOCATION, COUNTRY },                            new String[]{ "NNP" },      0.0312619314829170100 ),
  ORG_SUBSIDIARIES                      ("org:subsidiaries",                        NERTag.ORGANIZATION, Cardinality.LIST,   new NERTag[]{ ORGANIZATION },                                                    new String[]{ "NNP" },      0.0162412791706679320 ),
  ORG_TOP_MEMBERS_SLASH_EMPLOYEES       ("org:top_members/employees",               NERTag.ORGANIZATION, Cardinality.LIST,   new NERTag[]{ PERSON },                                                          new String[]{ "NNP" },      0.0907168724184609800 ),
  ORG_DISSOLVED                         ("org:dissolved",                           NERTag.ORGANIZATION, Cardinality.SINGLE, new NERTag[]{ DATE },                                                            new String[]{ "CD", "NN" }, 0.0023877428237553656 ),
  ORG_FOUNDED                           ("org:founded",                             NERTag.ORGANIZATION, Cardinality.SINGLE, new NERTag[]{ DATE },                                                            new String[]{ "CD", "NN" }, 0.0796314401082944800 ),
  ORG_NUMBER_OF_EMPLOYEES_SLASH_MEMBERS ("org:number_of_employees/members",         NERTag.ORGANIZATION, Cardinality.SINGLE, new NERTag[]{ NUMBER },                                                          new String[]{ "CD", "NN" }, 0.0366274831946870950 ),
  ORG_WEBSITE                           ("org:website",                             NERTag.ORGANIZATION, Cardinality.SINGLE, new NERTag[]{ URL },                                                             new String[]{ "NNP" },      0.0051544006201478640 ),
  ;

  public boolean sameEquivalenceClass(RelationType relationType) {
    return relationType == this || OfficialOutputWriter.officialRelationName(this).equals(OfficialOutputWriter.officialRelationName(relationType));
  }


  public static enum Cardinality {
    SINGLE,
    LIST
  }


  /**
   * A canonical name for this relation type. This is the official 2010 relation name,
   * that has since changed (see {@link OfficialOutputWriter#officialRelationName(RelationType, edu.stanford.nlp.kbp.slotfilling.common.Props.YEAR)}).
   */
  public final String canonicalName;
  /**
   * The entity type (left arg type) associated with this relation. That is, either a PERSON or an ORGANIZATION "slot".
   */
  public final NERTag entityType;
  /**
   * The cardinality of this entity. That is, can multiple right arguments participate in this relation (born_in vs. lived_in)
   */
  public final Cardinality cardinality;
  /**
   * Valid named entity labels for the right argument to this relation
   */
  public final Set<NERTag> validNamedEntityLabels;
  /**
   * Valid POS [prefixes] for the right argument to this relation (e.g., can only take nouns, or can only take numbers, etc.)
   */
  public final Set<String> validPOSPrefixes;
  /**
   * The prior for how often this relation occurs in the training data.
   * Note that this prior is not necessarily accurate for the test data.
   */
  public final double priorProbability;


  RelationType(String canonicalName, NERTag type, Cardinality cardinality, NERTag[] validNamedEntityLabels, String[] validPOSPrefixes,
               double priorProbability) {
    this.canonicalName          = canonicalName;
    this.entityType             = type;
    this.cardinality            = cardinality;
    this.validNamedEntityLabels = new HashSet<NERTag>(Arrays.asList(validNamedEntityLabels));
    this.validPOSPrefixes       = new HashSet<String>(Arrays.asList(validPOSPrefixes));
    this.priorProbability       = priorProbability;
  }

  /** Returns whether this relation type takes a person as a slot value */
  public boolean isPersonNameRelation(){ return validNamedEntityLabels.contains(PERSON); }
  /** Returns whether this relation type takes a date as a slot value */
  public boolean isDateRelation() { return validNamedEntityLabels.contains(DATE); }
  /** Returns whether this relation type takes a country as a slot value */
  public boolean isCountryNameRelation(){ return validNamedEntityLabels.contains(COUNTRY); }

  public boolean isCityRegionCountryRelation() {
    return
        this == ORG_CITY_OF_HEADQUARTERS ||
            this == RelationType.ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS ||
            this == RelationType.ORG_COUNTRY_OF_HEADQUARTERS ||
            this == RelationType.PER_CITIES_OF_RESIDENCE ||
            this == RelationType.PER_STATE_OR_PROVINCES_OF_RESIDENCE ||
            this == RelationType.PER_COUNTRIES_OF_RESIDENCE ||
            this == RelationType.PER_CITY_OF_BIRTH ||
            this == RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH ||
            this == RelationType.PER_COUNTRY_OF_BIRTH ||
            this == RelationType.PER_CITY_OF_DEATH ||
            this == RelationType.PER_STATE_OR_PROVINCES_OF_DEATH ||
            this == RelationType.PER_COUNTRY_OF_DEATH;
  }


  /** Returns whether this relation encodes a city/region/country of birth */
  public boolean isBirthRelation() {
    return this == PER_COUNTRY_OF_BIRTH || this == PER_STATE_OR_PROVINCES_OF_BIRTH || this == PER_CITY_OF_BIRTH;
  }

  /** Returns whether this relation denotes alternate names for a person/organization */
  public boolean isAlternateName() {
    return this == PER_ALTERNATE_NAMES || this == ORG_ALTERNATE_NAMES;
  }

  /**
   * Returns whether the given slot fill type is consistent with relations which are expecting a date.
   * @param slotType The type of the slot fill (e.g., COUNTRY).
   * @return true if a relation which accepts slotType also accepts the DATE type.
   */
  @SuppressWarnings("UnusedDeclaration")
  public static boolean isRelationTypeForDateRelation(NERTag slotType) {
    for (RelationType rel : slotTypeToRelations.get(slotType)) {
      if (rel.isDateRelation()) return true;
    }
    return false;
  }
  /**
   * Returns whether the given slot fill type is consistent with relations which are expecting a person.
   * @param slotType The type of the slot fill (e.g., COUNTRY).
   * @return true if a relation which accepts slotType also accepts the PERSON type.
   */
  @SuppressWarnings("UnusedDeclaration")
  public static boolean isRelationTypeForPersonRelation(NERTag slotType) {
    for (RelationType rel : slotTypeToRelations.get(slotType)) {
      if (rel.isPersonNameRelation()) return true;
    }
    return false;
  }
  /**
   * Returns whether the given slot fill type is consistent with relations which are expecting a country.
   * @param slotType The type of the slot fill (e.g., COUNTRY).
   * @return true if a relation which accepts slotType also accepts the COUNTRY type.
   */
  @SuppressWarnings("UnusedDeclaration")
  public static boolean isRelationTypeForCountryRelation(NERTag slotType) {
    for (RelationType rel : slotTypeToRelations.get(slotType)) {
      if (rel.isCountryNameRelation()) return true;
    }
    return false;
  }


  /**
   * Returns whether a classification of both of these arguments is consistent with our knowledge of relations
   * which co-occur in the knowledge base.
   * For example, the same entities (e.g., Obama and Hawaii) could express both stateorprovince_of_birth and statorprovince_of_residence
   * but could not express spouse.
   * @param otherRelationWithSameArguments Another relation which could occur with the same arguments
   * @return True if these relations could plausibly (but not necessarily or even likely) co-occur.
   */
  public boolean plausiblyCooccursWith(RelationType otherRelationWithSameArguments) {
    return plausibleOverlappingRelations.containsKey(this) && plausibleOverlappingRelations.containsKey(otherRelationWithSameArguments) && (plausibleOverlappingRelations.get(this).contains(otherRelationWithSameArguments) || plausibleOverlappingRelations.get(otherRelationWithSameArguments).contains(this));
  }

  //
  // Static Methods
  //

  /** An array of the single valued relations */
  public static final RelationType[] singleValuedRelations;
  /** An array of the list valued relations */
  public static final RelationType[] listValuedRelations;
  /** An array of the person centric relations */
  public static final RelationType[] PERRelations;
  /** An array of the organization centric relations */
  public static final RelationType[] ORGRelations;
  /** Valid POS Tags **/
  public static final Set<String> POSPrefixes;
  /**
   * Compatible relations.
   * That is to say, for a given e_1, e_2, it is conceivable that both of these relations co-occur
   */
  public static final Map<RelationType, Set<RelationType>> plausibleOverlappingRelations = new HashMap<RelationType, Set<RelationType>>();
  /**
   * A map from a named entity type to the relations that take that type as a possible fill.
   * For example, DATE maps to per:date, org:founded, and org:dissolved.
   */
  public static final Map<NERTag, Set<RelationType>> slotTypeToRelations = new HashMap<NERTag, Set<RelationType>>();

  /** Static initialization of fields */
  static {
    // -- Count relations
    int singleCount = 0;
    int listCount   = 0;
    int PERCount    = 0;
    int ORGCount    = 0;
    for (RelationType rel : values()) {
      // Cardinality
      switch(rel.cardinality) {
        case SINGLE: singleCount += 1; break;
        case LIST:   listCount   += 1; break;
        default:     fatal("Unknown cardinality: " + rel.cardinality);
      }
      // PER / ORG
      String relationName = rel.canonicalName.toLowerCase();
      if (relationName.startsWith("per:")) PERCount += 1;
      else if (relationName.startsWith("org:")) ORGCount += 1;
      else fatal("Unknown prefix: " + rel.toString());
    }
    // -- Route relations
    // Create arrays
    singleValuedRelations = new RelationType[singleCount];
    listValuedRelations   = new RelationType[listCount];
    PERRelations          = new RelationType[PERCount];
    ORGRelations          = new RelationType[ORGCount];
    // Route
    for (RelationType rel : values()) {
      // Cardinality
      switch(rel.cardinality) {
        case SINGLE: singleCount -= 1; singleValuedRelations[singleCount] = rel; break;
        case LIST:   listCount   -= 1; listValuedRelations[listCount]     = rel; break;
        default:     fatal("Unknown cardinality: " + rel.cardinality);
      }
      // PER / ORG
      String relationName = rel.canonicalName.toLowerCase();
      if (relationName.startsWith("per:")) {
        PERCount -= 1;
        PERRelations[PERCount] = rel;
      } else if (relationName.startsWith("org:")) {
        ORGCount -= 1;
        ORGRelations[ORGCount] = rel;
      } else {
        fatal("Unknown prefix: " + rel.toString());
      }
    }
    // -- Populate Plausible Overlapping Relations
    plausibleOverlappingRelations.put(PER_STATE_OR_PROVINCES_OF_DEATH, new HashSet<RelationType>(Arrays.asList(new RelationType[]{PER_STATE_OR_PROVINCES_OF_RESIDENCE})));
    plausibleOverlappingRelations.put(PER_STATE_OR_PROVINCES_OF_BIRTH, new HashSet<RelationType>(Arrays.asList(new RelationType[]{PER_STATE_OR_PROVINCES_OF_RESIDENCE, PER_STATE_OR_PROVINCES_OF_DEATH})));
    plausibleOverlappingRelations.put(PER_SPOUSE, new HashSet<RelationType>(Arrays.asList(new RelationType[]{PER_TITLE, PER_STATE_OR_PROVINCES_OF_BIRTH})));
    plausibleOverlappingRelations.put(PER_SCHOOLS_ATTENDED, new HashSet<RelationType>(Arrays.asList(new RelationType[]{PER_STATE_OR_PROVINCES_OF_RESIDENCE, PER_STATE_OR_PROVINCES_OF_BIRTH})));
    plausibleOverlappingRelations.put(PER_PARENTS,new HashSet<RelationType>(Arrays.asList(new RelationType[]{PER_SPOUSE})));
    plausibleOverlappingRelations.put(PER_MEMBER_OF, new HashSet<RelationType>(Arrays.asList(new RelationType[]{PER_STATE_OR_PROVINCES_OF_RESIDENCE, PER_STATE_OR_PROVINCES_OF_DEATH, PER_STATE_OR_PROVINCES_OF_BIRTH, PER_SPOUSE, PER_SCHOOLS_ATTENDED, PER_ORIGIN})));
    plausibleOverlappingRelations.put(PER_EMPLOYEE_OF, new HashSet<RelationType>(Arrays.asList(new RelationType[]{PER_STATE_OR_PROVINCES_OF_RESIDENCE, PER_STATE_OR_PROVINCES_OF_DEATH, PER_STATE_OR_PROVINCES_OF_BIRTH, PER_SCHOOLS_ATTENDED, PER_MEMBER_OF})));
    plausibleOverlappingRelations.put(PER_DATE_OF_BIRTH,new HashSet<RelationType>(Arrays.asList(new RelationType[]{PER_DATE_OF_DEATH})));
    plausibleOverlappingRelations.put(PER_COUNTRY_OF_DEATH, new HashSet<RelationType>(Arrays.asList(new RelationType[]{PER_STATE_OR_PROVINCES_OF_BIRTH, PER_MEMBER_OF, PER_EMPLOYEE_OF})));
    plausibleOverlappingRelations.put(PER_COUNTRY_OF_BIRTH, new HashSet<RelationType>(Arrays.asList(new RelationType[]{PER_STATE_OR_PROVINCES_OF_BIRTH, PER_SCHOOLS_ATTENDED, PER_MEMBER_OF, PER_EMPLOYEE_OF, PER_COUNTRY_OF_DEATH})));
    plausibleOverlappingRelations.put(PER_COUNTRIES_OF_RESIDENCE, new HashSet<RelationType>(Arrays.asList(new RelationType[]{PER_STATE_OR_PROVINCES_OF_BIRTH, PER_SCHOOLS_ATTENDED, PER_MEMBER_OF, PER_EMPLOYEE_OF, PER_COUNTRY_OF_DEATH, PER_COUNTRY_OF_BIRTH})));
    plausibleOverlappingRelations.put(PER_CITY_OF_DEATH, new HashSet<RelationType>(Arrays.asList(new RelationType[]{PER_STATE_OR_PROVINCES_OF_DEATH, PER_STATE_OR_PROVINCES_OF_BIRTH, PER_SCHOOLS_ATTENDED, PER_MEMBER_OF, PER_EMPLOYEE_OF, PER_COUNTRY_OF_DEATH, PER_COUNTRY_OF_BIRTH})));
    plausibleOverlappingRelations.put(PER_CITY_OF_BIRTH, new HashSet<RelationType>(Arrays.asList(new RelationType[]{PER_STATE_OR_PROVINCES_OF_DEATH, PER_STATE_OR_PROVINCES_OF_BIRTH, PER_SPOUSE, PER_SCHOOLS_ATTENDED, PER_MEMBER_OF, PER_EMPLOYEE_OF, PER_COUNTRY_OF_BIRTH, PER_COUNTRIES_OF_RESIDENCE, PER_CITY_OF_DEATH})));
    plausibleOverlappingRelations.put(PER_CITIES_OF_RESIDENCE, new HashSet<RelationType>(Arrays.asList(new RelationType[]{PER_STATE_OR_PROVINCES_OF_RESIDENCE, PER_STATE_OR_PROVINCES_OF_BIRTH, PER_SCHOOLS_ATTENDED, PER_MEMBER_OF, PER_EMPLOYEE_OF, PER_CITY_OF_DEATH, PER_CITY_OF_BIRTH})));
    plausibleOverlappingRelations.put(PER_CHILDREN, new HashSet<RelationType>(Arrays.asList(new RelationType[]{PER_SPOUSE, PER_PARENTS, PER_CITY_OF_DEATH})));
    plausibleOverlappingRelations.put(PER_ALTERNATE_NAMES, new HashSet<RelationType>(Arrays.asList(new RelationType[]{PER_STATE_OR_PROVINCES_OF_BIRTH, PER_SPOUSE, PER_PARENTS, PER_ORIGIN, PER_MEMBER_OF, PER_COUNTRY_OF_BIRTH, PER_CITY_OF_DEATH, PER_CITY_OF_BIRTH, PER_CHILDREN})));
    plausibleOverlappingRelations.put(ORG_SUBSIDIARIES, new HashSet<RelationType>(Arrays.asList(new RelationType[]{ORG_TOP_MEMBERS_SLASH_EMPLOYEES})));
    plausibleOverlappingRelations.put(ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS, new HashSet<RelationType>(Arrays.asList(new RelationType[]{ORG_TOP_MEMBERS_SLASH_EMPLOYEES, ORG_SUBSIDIARIES})));
    plausibleOverlappingRelations.put(ORG_PARENTS, new HashSet<RelationType>(Arrays.asList(new RelationType[]{ORG_TOP_MEMBERS_SLASH_EMPLOYEES, ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS})));
    plausibleOverlappingRelations.put(ORG_MEMBER_OF, new HashSet<RelationType>(Arrays.asList(new RelationType[]{ORG_TOP_MEMBERS_SLASH_EMPLOYEES, ORG_SUBSIDIARIES, ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS, ORG_POLITICAL_RELIGIOUS_AFFILIATION, ORG_PARENTS, ORG_MEMBERS})));
    plausibleOverlappingRelations.put(ORG_FOUNDED_BY, new HashSet<RelationType>(Arrays.asList(new RelationType[]{ORG_TOP_MEMBERS_SLASH_EMPLOYEES, ORG_SUBSIDIARIES, ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS, ORG_POLITICAL_RELIGIOUS_AFFILIATION, ORG_PARENTS})));
    plausibleOverlappingRelations.put(ORG_FOUNDED,new HashSet<RelationType>(Arrays.asList(new RelationType[]{ORG_NUMBER_OF_EMPLOYEES_SLASH_MEMBERS})));
    plausibleOverlappingRelations.put(ORG_DISSOLVED, new HashSet<RelationType>(Arrays.asList(new RelationType[]{ORG_NUMBER_OF_EMPLOYEES_SLASH_MEMBERS, ORG_FOUNDED})));
    plausibleOverlappingRelations.put(ORG_COUNTRY_OF_HEADQUARTERS, new HashSet<RelationType>(Arrays.asList(new RelationType[]{ORG_TOP_MEMBERS_SLASH_EMPLOYEES, ORG_SUBSIDIARIES, ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS, ORG_PARENTS, ORG_MEMBERS, ORG_MEMBER_OF, ORG_FOUNDED_BY})));
    plausibleOverlappingRelations.put(ORG_CITY_OF_HEADQUARTERS, new HashSet<RelationType>(Arrays.asList(new RelationType[]{ORG_TOP_MEMBERS_SLASH_EMPLOYEES, ORG_SUBSIDIARIES, ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS, ORG_PARENTS, ORG_MEMBER_OF, ORG_FOUNDED_BY, ORG_COUNTRY_OF_HEADQUARTERS})));
    plausibleOverlappingRelations.put(ORG_ALTERNATE_NAMES, new HashSet<RelationType>(Arrays.asList(new RelationType[]{ORG_TOP_MEMBERS_SLASH_EMPLOYEES, ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS, ORG_POLITICAL_RELIGIOUS_AFFILIATION, ORG_FOUNDED_BY, ORG_COUNTRY_OF_HEADQUARTERS, ORG_CITY_OF_HEADQUARTERS})));
    // -- Populate Slot Type to Relations
    for (NERTag ne : NERTag.values()) {
      slotTypeToRelations.put(ne, new HashSet<RelationType>());
    }
    for (RelationType rel : RelationType.values()) {
      for (NERTag validSlotFillType : rel.validNamedEntityLabels) {
        slotTypeToRelations.get(validSlotFillType).add(rel);
      }
    }

    POSPrefixes = new HashSet<String>();
    for( RelationType ty : RelationType.values() ) {
      POSPrefixes.addAll(ty.validPOSPrefixes);
    }
  }

  /** A small cache of names to relation types; we call fromString() a lot in the code, usually expecting it to be very fast */
  private static final Map<String, RelationType> cachedFromString = new HashMap<String, RelationType>();

  /** Find the slot for a given name */
  public static Maybe<RelationType> fromString(String name) {
    if (name == null) { return Maybe.Nothing(); }
    String originalName = name;
    if (cachedFromString.get(name) != null) { return Maybe.Just(cachedFromString.get(name)); }
    if (cachedFromString.containsKey(name)) { return Maybe.Nothing(); }
    // Try naive
    for (RelationType slot : RelationType.values()) {
      if (slot.canonicalName.equals(name) || slot.name().equals(name)) {
        cachedFromString.put(originalName, slot);
        return Maybe.Just(slot);
      }
    }
    // Replace slashes
    name = name.toLowerCase().replaceAll("[Ss][Ll][Aa][Ss][Hh]", "/");
    for (RelationType slot : RelationType.values()) {
      if (slot.canonicalName.equalsIgnoreCase(name)) {
        cachedFromString.put(originalName, slot);
        return Maybe.Just(slot);
      }
    }
    // Canonicalize to year
    for (RelationType slot : RelationType.values()) {
      if (name.equals(OfficialOutputWriter.officialRelationName(slot))) {
        cachedFromString.put(originalName, slot);
        return Maybe.Just(slot);
      }
    }
    // Canonicalize to any year
    for (RelationType slot : RelationType.values()) {
      for(Props.YEAR y : Props.YEAR.values()) {
        if (name.equals(OfficialOutputWriter.officialRelationName(slot, y))) {
          cachedFromString.put(originalName, slot);
          return Maybe.Just(slot);
        }
      }
    }
    cachedFromString.put(originalName, null);
    return Maybe.Nothing();
  }

  public static boolean isKBPRelation(String relation) {
    //noinspection ConstantConditions
    return (cachedFromString != null && cachedFromString.get(relation) != null) || fromString(relation).isDefined();
  }

  /** Determine if relation is single valued */
  public static boolean singleValuedRelation(RelationType rel) { return rel.cardinality == Cardinality.SINGLE; }
  /** Determine if relation is single valued */
  public static boolean singleValuedRelation(String rel) { 
    rel = rel.replace("/", "SLASH" );
    return singleValuedRelation(fromString(rel).orCrash());
  }
  /** Determine if relation is list valued */
  public static boolean listValuedRelation(RelationType rel) { return rel.cardinality == Cardinality.LIST; }
  /** Determine if relation is list valued */
  public static boolean listValuedRelation(String rel) { 
    rel = rel.replace("/", "SLASH" );
    return listValuedRelation(fromString(rel).orCrash());
  }
}
