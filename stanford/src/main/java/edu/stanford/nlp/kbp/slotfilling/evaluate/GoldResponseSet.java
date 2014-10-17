package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.logging.Color;
import edu.stanford.nlp.util.logging.PrettyLoggable;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.*;
import java.text.DecimalFormat;
import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * A utility to help track which slot fills we are missing, and not
 * just which slot fills we are filling incorrectly
 *
 * @author Gabor Angeli
 */
@SuppressWarnings("UnusedDeclaration")
public class GoldResponseSet {

  /** A consistent decimal format for scores */
  private static final java.text.DecimalFormat df = new java.text.DecimalFormat("0.000");

  /**
   * <p>The embodiment of a gold response; that is, each object embodies the set of <b>equivalent</b>
   * slot fills. So, (Obama, title, president) and (Obama, title, Pres.) would be in the same category,
   * but (Obama, title, senator) would not.</p>
   *
   * <p>Note that the hashCode() and equals() methods are tuned so that a correct GuessResponse will be
   * equal to the corresponding GoldResponse.</p>
   */
  public static class GoldResponse {
    /** The entity for the response (e.g., Obama) */
    public final KBPEntity entity;
    /** The relation for the response (e.g., born_in) */
    public final RelationType relation;
    /** The correct slot values, along with their provenance (e.g., ["hawaii", "doc7"]) */
    public final Set<Pair<String,String>> correctSlotValues;
    /** Known incorrect slot values, along with their provenance (e.g., ["kenya", "doc8"]) */
    public final Set<Pair<String,String>> incorrectSlotValues;
    /** The equivalence class of this slot fill, used for checking equality and clustering slot responses */
    public final int equivalenceClass;

    public GoldResponse(int equivalenceClass, KBPEntity entity, RelationType relation, Set<Pair<String, String>> correctSlotValues, Set<Pair<String, String>> incorrectSlotValues) {
      this.equivalenceClass = equivalenceClass;
      this.entity = entity;
      this.relation = relation;
      this.correctSlotValues = correctSlotValues;
      this.incorrectSlotValues = incorrectSlotValues;
    }

    public GoldResponse(int equivalenceClass, KBPEntity entity, RelationType relation) {
      this(equivalenceClass, entity, relation, new HashSet<Pair<String, String>>(), new HashSet<Pair<String,String>>());
    }

    // Implementation note: equals on GoldResponse must be defined on entity, relation, and equivalence class only.
    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o instanceof GoldResponse) {
        GoldResponse that = (GoldResponse) o;
        return equivalenceClass == that.equivalenceClass && entity.equals(that.entity) && relation == that.relation;
      } else if (o instanceof GuessResponse) {
        GuessResponse that = (GuessResponse) o;
        // Enforce entity and relation match
        if (!this.entity.equals(that.entity) || !this.relation.equals(that.relation)) {
          return false;
        }
        if (that.provenance.isDefined()) {
          if (!that.provenance.get().isOfficial()) { return false; }
          // If there's a provenance, enforce it
          Pair<String, String> key = Pair.makePair(that.slotValue, that.provenance.get().docId);
          return correctSlotValues.contains(key) || incorrectSlotValues.contains(key);
        } else {
          // Else, check only on the slot value
          for (Pair<String, String> key : correctSlotValues) {
            if (key.first.equals(that.slotValue)) { return true; }
          }
          for (Pair<String, String> key : incorrectSlotValues) {
            if (key.first.equals(that.slotValue)) { return true; }
          }
          return false;
        }
      } else {
        return false;
      }
    }

    // Implementation note: must be defined on only entity and relation
    @Override
    public int hashCode() {
      int result = entity.hashCode();
      result = 31 * result + relation.hashCode();
      return result;
    }

    @Override
    public String toString() {
      Set<String> onlySlotValues = new HashSet<String>();
      for (Pair<String, String> slotValue : correctSlotValues) {
        onlySlotValues.add(slotValue.first);
      }
      return entity.name + " " + relation.canonicalName + " { " + StringUtils.join(onlySlotValues, " | ") + " }";
    }
  }

  /**
   * <p>An embodiment of a guessed response, with the bare minimum necessary information to determine
   * if it's right or not.</p>
   *
   * <p>Note that the hashCode() and equals() methods are tuned so that a correct GuessResponse will be
   * equal to the corresponding GoldResponse.</p>
   *
   */
  private static class GuessResponse implements Comparable<GuessResponse> {
    public final KBPEntity entity;
    public final RelationType relation;
    public final String slotValue;
    public final Maybe<KBPRelationProvenance> provenance;
    public final double score;

    private GuessResponse(KBPEntity entity, RelationType relation, String slotValue, Maybe<KBPRelationProvenance> provenance, double score) {
      this.entity = entity;
      this.relation = relation;
      this.slotValue = slotValue;
      this.provenance = provenance;
      this.score = score;
    }
    private GuessResponse(KBPSlotFill fill) {
      this(fill.key.getEntity(), fill.key.kbpRelation(), fill.key.slotValue, fill.provenance, fill.score.getOrElse(-1.0));
    }

    public GuessResponse withNoProvenance() {
      return new GuessResponse(entity, relation, slotValue, Maybe.<KBPRelationProvenance>Nothing(), score);
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o instanceof GoldResponse) {
        GoldResponse that = (GoldResponse) o;
        // Enforce entity and relation match
        if (!this.entity.equals(that.entity) || !this.relation.equals(that.relation)) {
          return false;
        }
        if (provenance.isDefined()) {
          // If there's a provenance, enforce it
          if (!provenance.get().isOfficial()) { return false; }
          Pair<String, String> key = Pair.makePair(slotValue, provenance.get().docId);
          return that.correctSlotValues.contains(key) || that.incorrectSlotValues.contains(key);
        } else {
          // Else, check only on the slot value
          for (Pair<String, String> key : that.correctSlotValues) {
            if (key.first.equals(this.slotValue)) { return true; }
          }
          for (Pair<String, String> key : that.incorrectSlotValues) {
            if (key.first.equals(this.slotValue)) { return true; }
          }
          return false;
        }
      } else if (o instanceof GuessResponse) {  // Provenance-insensitive...
        GuessResponse that = (GuessResponse) o;
        return entity.equals(that.entity) && relation.equals(that.relation) && slotValue.equals(that.slotValue);
      } else {
        return false;
      }
    }

    @Override
    public int hashCode() {
      int result = entity.hashCode();
      result = 31 * result + relation.hashCode();
      return result;
    }

    @Override
    public String toString() {
      return "[" + df.format(score) + "] " + entity.name + " " + relation.canonicalName + " " + slotValue;
    }

    @Override
    public int compareTo(GuessResponse o) {
      if (this.score < o.score) return 1;
      else if (this.score > o.score) return -1;
      else return 0;
    }
  }

  /**
   * The collection of possible causes for discarding a slot.
   * Note that {@link edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet.ErrorType#WRONG_PROVENANCE} is used primarily internally to denote a slot which
   * would be correct if anydoc were set to true.
   */
  public static enum ErrorType {
    WRONG_PROVENANCE(GREEN),
    FAILED_CONSISTENCY(MAGENTA),
    NO_PROVENANCE(MAGENTA),
    REWRITTEN(MAGENTA);

    public final Color color;

    ErrorType(Color color) {
      this.color = color;
    }
  }

  /** Just a function, which in Java-land means a class */
  private static interface ResponseProcessor {
    /**
     * We read a response from an official output file, and have grok'd it into its components, and
     * this function is called for each line (e.g., each single response) in the gold file.
     *
     * @param lineId The "id" of this slot fill among slot fills with the same entity and relation.
     *               This is <b>not</b> unique for all slot fills, but is only unique among slot fills
     *               that share an entity and a relation.
     * @param entity The official entity for the slot fill.
     * @param relation The relation proposed by this slot fill.
     * @param slotValue The textual value of this slot fill.
     * @param docid The document id registered in the key file, likely aligned to that particular year's source docs.
     * @param judgement Whether the slot fill was deemed incorrect, inexact, etc.
     * @param equivalenceClass The equivalence class registered for the slot fill.
     */
    public void addGoldResponse(int lineId, KBPOfficialEntity entity, String relation, String slotValue, String docid, int judgement,
                                 int equivalenceClass);
  }

  /** The usual function for processing responses -- add them to the response set */
  private class StandardResponseProcessor implements ResponseProcessor {
    private int nextIncorrectEquivalenceClass = -1;  // Implementation Note: This forces only one StandardResponseProcessor to be created when reading a response directory

    @Override
    public void addGoldResponse(int lineId, KBPOfficialEntity entity, String relation, String slotValue, String docid, int judgement,
                                int equivalenceClass) {
      if (docid.startsWith("noidxxxx")) {
        warn("adding gold without proper docid: " + docid);
      }
      if (entity != null && entity.name != null) { // Ignore entities without mapping to test entity set
        // Error checks
        if (!RelationType.fromString(relation).isDefined()) {
          err("Could not map relation in " + Props.KBP_YEAR + ": " + relation);
          return;
        }
        if (!OfficialOutputWriter.officialRelationName(RelationType.fromString(relation).get()).equals(relation)) {
          warn("Official output for " + relation + " in " + Props.KBP_YEAR + " is not the same as the output in the results file");
        }
        // Tweak equivalence class
        if (judgement != CustomSFScore.CORRECT) {
          equivalenceClass = nextIncorrectEquivalenceClass;
          nextIncorrectEquivalenceClass -= 1;
        }

        GoldResponse dummy = new GoldResponse(equivalenceClass, entity, RelationType.fromString(relation).orCrash());
        if (!goldResponses.containsKey(dummy)) {
          goldResponses.put(dummy, dummy);
        }
        if (judgement == CustomSFScore.CORRECT) {
          goldResponses.get(dummy).correctSlotValues.add(Pair.makePair(slotValue, docid));
        } else {
          goldResponses.get(dummy).incorrectSlotValues.add(Pair.makePair(slotValue, docid));
        }
      }
    }
  }

  /** Write each response to a key file, in a format that SFScore can read */
  private class KeyFileResponseProcessor implements ResponseProcessor {
    public final PrintWriter toWriteTo;
    public KeyFileResponseProcessor(PrintWriter toWriteTo) { this.toWriteTo = toWriteTo; }

    @Override
    public void addGoldResponse(int lineId, KBPOfficialEntity entity, String relation, String slotValue, String docid, int judgement,
                                int equivalenceClass) {
    	
      StringBuilder line = new StringBuilder();
      line.append(lineId).append("\t")
          .append(entity.queryId.orCrash()).append("\t")
          .append("LDC").append("\t")
          .append(relation).append("\t")
          .append(docid).append("\t")
          .append("0\t0\t")
          .append(slotValue).append("\t")
          .append(slotValue).append("\t")
          .append(equivalenceClass).append("\t")
          .append(judgement);
      toWriteTo.println(line.toString());
    }
  }



  /** Just an interner -- take the ValueSet and treat it as a set */
  private Map<GoldResponse, GoldResponse> goldResponses = new HashMap<GoldResponse, GoldResponse>();
  /** The set of guessed responses (for the checklist functionality */
  private Set<GuessResponse> guessedResponses = new HashSet<GuessResponse>();
  /** The set of discarded responses, that we've changed our mind on during consistency */
  private Set<Pair<GuessResponse,ErrorType>> discardedResponses = new HashSet<Pair<GuessResponse,ErrorType>>();
  /** Remember the set of entities we're supposed to have read responses for */
  private final Maybe<Collection<KBPOfficialEntity>> entities;

  public GoldResponseSet() {
    this.entities = Maybe.Nothing();
    readGoldResponses(DataUtils.testEntities(Props.TEST_QUERIES.getPath(), Maybe.<KBPIR>Nothing()),
        new StandardResponseProcessor());
  }

  public GoldResponseSet(HashMap<GoldResponse, GoldResponse> goldResponses) {
    this.entities = Maybe.Nothing();
    this.goldResponses = goldResponses;
  }

  public GoldResponseSet(Collection<KBPOfficialEntity> entities) {
    this.entities = Maybe.Just(entities);
    readGoldResponses(entities, new StandardResponseProcessor());
  }





  private void grokBefore2011(
      ResponseProcessor processor,
      Map<String, KBPOfficialEntity> queryIdToEntity,
      String line) {
    // Read line
    String[] fields = line.split("\t", 11);
    assert fields.length == 11;
    // Get fields
    int lineId = Integer.parseInt(fields[0]);
    KBPOfficialEntity entity = queryIdToEntity.get(fields[1]);
    String relation = fields[3];
    String slotValue = fields[8];
    String docId = fields[4];
    int equivalenceClass = Integer.parseInt(fields[9]);
    int judgement = Integer.parseInt(fields[10]);
    // Add response
    processor.addGoldResponse(lineId, entity, relation, slotValue, docId, judgement, equivalenceClass);
  }

  private void grok2011(
      ResponseProcessor processor,
      Map<String, KBPOfficialEntity> queryIdToEntity,
      String line) {
    // Read line
    String[] fields = line.split("\\s+", 11);
    assert fields.length >= 6;
    // Get fields
    int lineId = Integer.parseInt(fields[0]);
    String[] slotInfo = fields[1].split(":");
    assert slotInfo.length == 3;
    KBPOfficialEntity entity = queryIdToEntity.get(slotInfo[0]);
    String relation = slotInfo[1] + ":" + slotInfo[2];
    String slotValue = StringUtils.join(Arrays.asList(fields).subList(5, fields.length), " ");
    String docId = fields[2];
    int equivalenceClass = Integer.parseInt(fields[4]);
    int judgement = Integer.parseInt(fields[3]);
    // Add response
    processor.addGoldResponse(lineId, entity, relation, slotValue, docId, judgement, equivalenceClass);
  }

  private void grok2012(
      ResponseProcessor processor,
      Map<String, KBPOfficialEntity> queryIdToEntity,
      String line) {
    // Read line
    String[] fields = line.split("\t");
    assert fields.length == 10;
    // Get fields
    int lineId = Integer.parseInt(fields[0]);
    String[] slotInfo = fields[1].split(":");
    assert slotInfo.length == 3;
    KBPOfficialEntity entity = queryIdToEntity.get(slotInfo[0]);
    String relation = slotInfo[1] + ":" + slotInfo[2];
    String slotValue = fields[5];
    String docId = fields[2];
    int equivalenceClass = Integer.parseInt(fields[4]);
    int judgement = Integer.parseInt(fields[3]);
    // Add response
    processor.addGoldResponse(lineId, entity, relation, slotValue, docId, judgement, equivalenceClass);
  }

  private void grok2013(
      ResponseProcessor processor,
      Map<String, KBPOfficialEntity> queryIdToEntity,
      String line) {
    // Read line
    String[] fields = line.split("\t");
    if (fields.length != 12) {
      err("Gold response line didn't have 12 entries: " + line);
      return;
    }
    // Get fields
    int lineId = Integer.parseInt(fields[0]);
    KBPOfficialEntity entity = queryIdToEntity.get(fields[1].substring(0, fields[1].indexOf(':')).trim());
    String relation = fields[1].substring(fields[1].indexOf(':') + 1).trim();
    String docId = fields[2].trim();
    String slotValue = fields[3].trim();
    int equivalenceClass = Integer.parseInt(fields[11]);
    int judgement = CustomSFScore.WRONG; // "W"
    if (fields[10].equals("C")) { judgement = CustomSFScore.CORRECT; }
    if (fields[10].equals("X")) { judgement = CustomSFScore.INEXACT; }
    if (fields[10].equals("R")) { judgement = CustomSFScore.REDUNDANT; }
    // Add response
    if (!fields[10].equals("I")) {
    //	System.out.println("adding..");
      processor.addGoldResponse(lineId, entity, relation, slotValue, docId, judgement, equivalenceClass);
    }
  }

  /**
   * An intelligible format for the gold responses we are expecting -- for debugging
   * purposes only! To actually score the system, use SFScore or CustomSFScore.
   */
  private void readGoldResponses(Collection<KBPOfficialEntity> queries, ResponseProcessor processor) {
    startTrack("Getting gold responses");
    try {
      Map<String, KBPOfficialEntity> queryIdToEntity = new HashMap<String, KBPOfficialEntity>();
      for (KBPOfficialEntity entity : queries) {
    	//  System.out.println("adding entity "+entity);
        queryIdToEntity.put(entity.queryId.orCrash(), entity);
      }
      //System.out.println("hello "+Props.TEST_RESPONSES==null);
      for (File queryFile : DataUtils.fetchFiles(Props.TEST_RESPONSES.getPath(), "", false)) {
        BufferedReader keyReader = null;
        try {
          keyReader = new BufferedReader(new FileReader(queryFile));
          String line = keyReader.readLine();
          int nextId = 1;
          while (line != null) {
        	  //System.out.println("reading line.."+line);
            switch (Props.KBP_YEAR) {
              case KBP2009: case KBP2010: grokBefore2011(processor, queryIdToEntity, line); break;
              case KBP2011: grok2011(processor, queryIdToEntity, line); break;
              case KBP2012: grok2012(processor, queryIdToEntity, line); break;
              case KBP2013: grok2013(processor, queryIdToEntity, line); break;
              default: throw new IllegalArgumentException("Unknown year: " + Props.KBP_YEAR);
            }
            line = keyReader.readLine();
          }
        } catch (IOException e) {
          warn(e);
        } finally {
          try {
            if (keyReader != null) { keyReader.close(); }
          } catch (IOException ignored) { }
        }
      }
    } finally {
      endTrack("Getting gold responses");
    }
  }

  //
  // UTILITIES FOR GETTING GOLD SLOT INFORMATION
  //

  /**
   * Returns whether this gold response set has been populated with responses.
   */
  public boolean isEmpty() {
    return goldResponses.isEmpty();
  }

  /**
   * Returns the number of <b>correct</b> fills in the gold response set.
   */
  public int size() {
    return correctFills().size();
  }

  /**
   * Get the set of correct slot fills, without de-duplicating equivalent fills.
   * @return The set of correct slot fills.
   */
  public Set<KBPSlotFill> correctFills() {
    return correctProvenances().keySet();
  }

  /**
   * A cache to prevent re-computing the flat provenance view of the gold slot fills multiple times
   * @see edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet#correctProvenances()
   */
  private Map<KBPSlotFill, Set<String>> correctProvenanceCached = null;

  /**
   * Get the set of provenances for any given slot fill.
   * @return A mapping from slot fills correct provenances for that fill.
   */
  public Map<KBPSlotFill, Set<String>> correctProvenances() {
    if (correctProvenanceCached == null) {
      correctProvenanceCached = new HashMap<KBPSlotFill, Set<String>>();
      for (GoldResponse response : goldResponses.values()) {
        for (Pair<String, String> slotValue : response.correctSlotValues) {
          KBPSlotFill key = KBPNew.from(response.entity).slotValue(slotValue.first).rel(response.relation)
              .provenance(new KBPRelationProvenance(slotValue.second, Props.INDEX_OFFICIAL.getPath()))
              .score(Double.POSITIVE_INFINITY)
              .KBPSlotFill();
          if (!correctProvenanceCached.containsKey(key)) { correctProvenanceCached.put(key, new HashSet<String>()); }
          correctProvenanceCached.get(key).add(slotValue.second);
        }
      }
    }
    return correctProvenanceCached;
  }

  //
  // UTILITIES FOR SCORING
  //

  public synchronized File keyFile() {

    File keyFile = new File(Props.WORK_DIR + File.separator + "key.tab");
    // Don't write it multiple times
    if (keyFile.exists()) {
      return keyFile;
    }
    try {
    	
    	//System.out.println("hello : ");
  	  

      PrintWriter writer = new PrintWriter(new FileWriter(keyFile));
      if (entities.isDefined()) {
        // Case: only read certain entities
        readGoldResponses(entities.get(), new KeyFileResponseProcessor(writer));
      } else {
        // Case: read all entities
        readGoldResponses(DataUtils.testEntities(Props.TEST_QUERIES.getPath(), Maybe.<KBPIR>Nothing()),
            new KeyFileResponseProcessor(writer));
      }
      writer.close();
      return keyFile;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }


  //
  // REGISTER OUR OWN RESPONSES
  //

  /**
   * Register that the system guessed a response BEFORE CONSISTENCY.
   * In particular, correct slots discarded with {@link GoldResponse#discardResponse(edu.stanford.nlp.kbp.slotfilling.common.KBPSlotFill, edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet.ErrorType)}
   * will be tracked if this is called on every slot output by the slot filler.
   * @param fill The slot fill to register as guessed, before consistency is applied.
   */
  public void registerResponse(KBPSlotFill fill) {
    if (fill.key.tryKbpRelation().isDefined()) {
      GuessResponse response = new GuessResponse(fill);
      guessedResponses.remove(response);
      guessedResponses.add(response);
    }
  }

  /**
   * Discard a slot fill for some reason or another -- most commonly, from consistency or provenance checks.
   * This can be undone with {@link GoldResponse#undoDiscardResponse(edu.stanford.nlp.kbp.slotfilling.common.KBPSlotFill, edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet.ErrorType)}.
   * @param fill The slot fill to discard.
   * @param cause The reason to discard this slot fill, e.g., no provenance or consistency failed.
   */
  public void discardResponse(KBPSlotFill fill, ErrorType cause) {
    if (fill.key.tryKbpRelation().isDefined()) {
      discardedResponses.add(Pair.makePair(new GuessResponse(fill), cause));
    }
  }

  /**
   * Often consistency is easiest implemented by discarding slots and re-adding them later. This is to undo a discard operation
   * from {@link GoldResponseSet#discardResponse(edu.stanford.nlp.kbp.slotfilling.common.KBPSlotFill, edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet.ErrorType)}.
   * @param fill The slot fill to re-enter as valid (at least according to us).
   * @param cause The cause of this slot being discarded in the first place, and the context in which it should
   *              be re-added. For example, if a slot is discarded both from consistency and provenance, and the consistency
   *              discard is undone, it will still be registered as discarded from provenance.
   */
  public void undoDiscardResponse(KBPSlotFill fill, ErrorType cause) {
    if (fill.key.tryKbpRelation().isDefined()) {
      discardedResponses.remove(Pair.makePair(new GuessResponse(fill), cause));
    }
  }

  /**
   * Alias for {@link edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet#discardResponse(edu.stanford.nlp.kbp.slotfilling.common.KBPSlotFill, edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet.ErrorType)}
   * with a cause of {@link edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet.ErrorType#NO_PROVENANCE}.
   */
  public void discardNoProvenance(KBPSlotFill fill) {
    discardResponse(fill, ErrorType.NO_PROVENANCE);
  }

  /**
   * Alias for {@link edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet#undoDiscardResponse(edu.stanford.nlp.kbp.slotfilling.common.KBPSlotFill, edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet.ErrorType)}
   * with a cause of {@link edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet.ErrorType#NO_PROVENANCE}.
   */
  public void undoDiscardNoProvenance(KBPSlotFill fill) {
    undoDiscardResponse(fill, ErrorType.NO_PROVENANCE);
  }

  /**
   * Alias for {@link edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet#discardResponse(edu.stanford.nlp.kbp.slotfilling.common.KBPSlotFill, edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet.ErrorType)}
   * with a cause of {@link edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet.ErrorType#FAILED_CONSISTENCY}.
   */
  public void discardInconsistent(KBPSlotFill fill) {
    discardResponse(fill, ErrorType.FAILED_CONSISTENCY);
  }

  /**
   * Alias for {@link edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet#undoDiscardResponse(edu.stanford.nlp.kbp.slotfilling.common.KBPSlotFill, edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet.ErrorType)}
   * with a cause of {@link edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet.ErrorType#FAILED_CONSISTENCY}.
   */
  public void undoDiscardInconsistent(KBPSlotFill fill) {
    undoDiscardResponse(fill, ErrorType.FAILED_CONSISTENCY);
  }

  /**
   * Alias for {@link edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet#discardResponse(edu.stanford.nlp.kbp.slotfilling.common.KBPSlotFill, edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet.ErrorType)}
   * with a cause of {@link edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet.ErrorType#REWRITTEN}.
   */
  public void discardRewritten(KBPSlotFill fill) {
    discardResponse(fill, ErrorType.REWRITTEN);
  }

  /**
   * Alias for {@link edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet#undoDiscardResponse(edu.stanford.nlp.kbp.slotfilling.common.KBPSlotFill, edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet.ErrorType)}
   * with a cause of {@link edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet.ErrorType#REWRITTEN}.
   */
  public void undoDiscardRewritten(KBPSlotFill fill) {
    undoDiscardResponse(fill, ErrorType.REWRITTEN);
  }


  public PrettyLoggable loggableForEntity(final KBPOfficialEntity entity, final Maybe<KBPIR> irComponent) {
    return new PrettyLoggable(){
      @SuppressWarnings("SuspiciousMethodCalls")  // this comes about from the strange equals() semantics of the Gold/Guess responses
      @Override
      public void prettyLog(Redwood.RedwoodChannels channels, String description) {

        // Collect responses we should get
        Set<GoldResponse> missingGoldResponses = new HashSet<GoldResponse>();
        Set<GoldResponse> knownIncorrectResponses = new HashSet<GoldResponse>();
        for (GoldResponse gold : goldResponses.values()) {
          if (!gold.correctSlotValues.isEmpty() && gold.entity.equals(entity)) {
            missingGoldResponses.add(gold);
          } else if (!gold.incorrectSlotValues.isEmpty() && gold.entity.equals(entity)) {
            knownIncorrectResponses.add(gold);
          }
        }

        // Useful variables to keep track of
        int totalCorrect = 0;
        int totalGold = missingGoldResponses.size();
        int totalGuessed = 0;
        List<Pair<GuessResponse,Maybe<Boolean>>> responses = new ArrayList<Pair<GuessResponse,Maybe<Boolean>>>();
        Map<ErrorType, Set<GuessResponse>> filteredButCorrect = new HashMap<ErrorType, Set<GuessResponse>>();
        Set<GuessResponse> correctlyDiscardedResponses = new HashSet<GuessResponse>();
        for (ErrorType type : ErrorType.values()) { filteredButCorrect.put(type, new HashSet<GuessResponse>()); }
        Map<GuessResponse, Integer> clusterIds = new IdentityHashMap<GuessResponse, Integer>();

        // Get Correct Slots
        for (GuessResponse guess : guessedResponses) {
          if (!guess.entity.equals(entity)) { continue; }
          // Determine if the slot is correct
          Maybe<Boolean> isCorrect = Maybe.Nothing();
          if (missingGoldResponses.contains(guess)) { isCorrect = Maybe.Just(true); }
          if (knownIncorrectResponses.contains(guess)) {
            isCorrect = Maybe.Just(false);
          }
          // Find and register discarded responses
          boolean isDiscarded = false;
          boolean isDiscardedForProvenance = false;
          for (Pair<GuessResponse, ErrorType> discarded : discardedResponses) {
            if (discarded.first.equals(guess)) {
              isDiscarded = true;
              if (discarded.second == ErrorType.NO_PROVENANCE) {
                isDiscardedForProvenance = true;
              }
            }
          }
          if (!isDiscarded) {
            // Everything we haven't discarded is a guess
            totalGuessed += 1;
            responses.add(Pair.makePair(guess, isCorrect));
          }
          if (isCorrect.getOrElse(false)) {
            // Everything we guessed at all and is correct is no longer "missing"
            missingGoldResponses.remove(guess);
          }
          if (isCorrect.getOrElse(false) && !isDiscarded) {
            // Mark correct guesses
            totalCorrect += 1;
            GoldResponse goldForGuess = goldResponses.get(guess);
            if (goldForGuess != null) {
              clusterIds.put(guess, goldForGuess.equivalenceClass);
            }
          }
          if (!isCorrect.getOrElse(false) && isDiscarded && !isDiscardedForProvenance) {
            // Mark correctly discarded responses
            correctlyDiscardedResponses.add(guess);
          }
        }

        // Get Missed [Correct] Slots
        for (Pair<GuessResponse, ErrorType> discarded : discardedResponses) {
          if (missingGoldResponses.contains(discarded.first)) {
            filteredButCorrect.get(discarded.second).add(discarded.first);
          }
        }

        // Get Correct Slots With Wrong Provenance
        for (GuessResponse guess : guessedResponses) {
          // Determine if the slot is correct
          boolean isCorrect = missingGoldResponses.contains(guess.withNoProvenance());
          if (isCorrect) {
            boolean isDiscarded = false;
            for (Pair<GuessResponse, ErrorType> discarded : discardedResponses) {
              if (discarded.first.equals(guess)) {
                isDiscarded = true;
              }
            }
            if (!isDiscarded) {
              // Case: correct slot ignoring provenance
              filteredButCorrect.get(ErrorType.WRONG_PROVENANCE).add(guess);
              responses.remove(Pair.makePair(guess, Maybe.Just(true)));
              responses.remove(Pair.makePair(guess, Maybe.Just(false)));
              responses.remove(Pair.makePair(guess, Maybe.<Boolean>Nothing()));
              missingGoldResponses.remove(guess);
            }
          }
        }

        // Print
        startTrack("Score For " + entity.name);

        // Responses
        startTrack("Responses");
        Collections.sort(responses, new Comparator<Pair<GuessResponse, Maybe<Boolean>>>() {
          @Override
          public int compare(Pair<GuessResponse, Maybe<Boolean>> o1, Pair<GuessResponse, Maybe<Boolean>> o2) {
            return o1.first.compareTo(o2.first);
          }
        });
        for (Pair<GuessResponse, Maybe<Boolean>> response : responses) {
          Color color = YELLOW;
          String sym = "✘ ";
          if (response.second.isDefined()) {
            if (response.second.get()) {
              sym = "✔ ";
              color = GREEN;
            } else {
              sym = "✘ ";
              color = RED;
            }
          }
          channels.log(color, sym + response.first);
          for (KBPRelationProvenance provenance : response.first.provenance) {
            for (KBPIR ir : irComponent) {
              if (provenance.isOfficial()) { channels.prettyLog("Provenance", provenance.loggable(ir)); }
            }
          }
          channels.log("        8\t" + entity.queryId.getOrElse("(null)") + "\tGaborKBP\t" + response.first.relation.canonicalName + "\t" + response.first.provenance.getOrElse(new KBPRelationProvenance("doc???", Props.INDEX_OFFICIAL.getPath())).docId + "\t0\t0\t" + response.first.slotValue + "\t" + response.first.slotValue + "\t" + Maybe.fromNull(clusterIds.get(response)).getOrElse(-1) + "\t1");
        }
        endTrack("Responses");

        // Filtered
        for (Map.Entry<ErrorType, Set<GuessResponse>> filteredEntries : filteredButCorrect.entrySet()) {
          startTrack("Correct but " + filteredEntries.getKey().name().replaceAll("_", " ").toLowerCase());
          for (GuessResponse response : filteredEntries.getValue()) {
            // Print missed response
            channels.log(filteredEntries.getKey().color, response.toString());
            // Print provenance (if the error was incorrect provenance)
            if (filteredEntries.getKey() == ErrorType.WRONG_PROVENANCE && response.provenance.isDefined() && irComponent.isDefined() &&
                response.provenance.get().isOfficial()) {
              channels.prettyLog("Provenance", response.provenance.get().loggable(irComponent.get()));
            }
          }
          endTrack("Correct but " + filteredEntries.getKey().name().replaceAll("_", " ").toLowerCase());
        }

        // Missed
        startTrack("Missing");
        for (GoldResponse gold : missingGoldResponses) {
          channels.log(CYAN, gold.toString());
        }
        endTrack("Missing");

        // Correctly Filtered
        startTrack("Correctly Filtered");
        for (GuessResponse discarded : correctlyDiscardedResponses) {
          channels.debug(discarded.toString());
        }
        endTrack("Correctly Filtered");

        // Score
        DecimalFormat df = new DecimalFormat("00.000%");
        double p = totalGuessed == 0 ? 1.0 : ((double) totalCorrect) / ((double) totalGuessed);
        double pAnyDoc = totalGuessed == 0 ? 1.0 : ((double) (totalCorrect + filteredButCorrect.get(ErrorType.WRONG_PROVENANCE).size())) / ((double) totalGuessed);
        double r = ((double) totalCorrect) / ((double) totalGold);
        double rAnyDoc = ((double) (totalCorrect + filteredButCorrect.get(ErrorType.WRONG_PROVENANCE).size())) / ((double) totalGold);
        double f1 = 2.0 * p * r / (p + r);
        double f1AnyDoc = 2.0 * pAnyDoc * rAnyDoc / (pAnyDoc + rAnyDoc);
        log(BLUE, "NORMAL: P " + df.format(p) + " R " + df.format(r) + " F1 " + df.format(f1));
        log(BLUE, "ANYDOC: P " + df.format(pAnyDoc) + " R " + df.format(rAnyDoc) + " F1 " + df.format(f1AnyDoc));
        endTrack("Score For " + entity.name);
      }
    };
  }

  public static GoldResponseSet empty() {
    return new GoldResponseSet(new HashMap<GoldResponse, GoldResponse>());
  }

}
