package edu.stanford.nlp.kbp.slotfilling.ir;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.*;
import java.util.regex.Pattern;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * An interface for querying documents and sentences, with many utility methods, and a few
 * key methods which should be implemented.
 *
 * @author Gabor Angeli
 */
@SuppressWarnings("UnusedDeclaration")
public abstract class KBPIR {

  //
  // To Override
  //

  /**
   * The all-encompasing query method -- most things are really syntactic sugar on top
   * of this method.
   *
   * Note that from here on, the convention of putting relation after slotValue disappears, as every method will take
   * both a relation and a slot fill.
   *
   * @param tableName The table to write to (either document or sentence)
   * @param expectedOutput The class to output to. This is either a CoreMap (for sentences) or Annotation (for documents)
   * @param entityName The name of the entity to query (e.g., Susan Boyle)
   * @param entityType The type of the entity to query (e.g., PERSON)
   * @param relation   The relation we are querying for (e.g., per:country_of_birth
   * @param slotValue   The slot fill value we are querying for (e.g., Scotland)
   * @param slotValueType The type of the slot fill (e.g., COUNTRY)
   * @param docidsToForce A set of docids to always query
   * @param maxDocuments The maximum number of documents to search over. This is not necessarily the same as the number
   * @param officialIndexOnly If set to true, the query will only run on the official index.
   * @return A sorted list of the top sentences
   */
  protected abstract <E extends CoreMap> List<E> queryCoreMaps(final String tableName, final Class<E> expectedOutput,
                                                               final String entityName, final Maybe<NERTag> entityType, final Maybe<String> relation,
                                                               final Maybe<String> slotValue, final Maybe<NERTag> slotValueType,
                                                               final Set<String> docidsToForce,
                                                               final int maxDocuments, final boolean officialIndexOnly);

  /**
   * Fetch a document from all the indices.
   */
  public abstract Annotation fetchDocument(String docId, boolean officialIndexOnly);

  /**
   * Get the total number of documents containing a set of search terms, in any document, as defined by its
   * corresponding querier.
   *
   * @param terms The query phrases to search for.
   * @return The number of results in the index.
   */
  public abstract int queryNumHits(Collection<String> terms);



  /** The top level function to query for sentences */
  protected List<CoreMap> querySentences(String entityName,
                                         Maybe<NERTag> entityType,
                                         Maybe<String> relation,
                                         Maybe<String> slotValue,
                                         Maybe<NERTag> slotValueType,
                                         Set<String> docidsToForce,
                                         int maxDocuments,
                                         boolean officialIndexOnly) {
    return queryCoreMaps(Props.DB_TABLE_SENTENCE_CACHE, CoreMap.class, entityName, entityType, relation, slotValue, slotValueType, docidsToForce, maxDocuments, officialIndexOnly);
  }

  /** The top level function to query for entire documents */
  protected List<Annotation> queryDocuments(String entityName,
                                            Maybe<NERTag> entityType,
                                            Maybe<String> relation,
                                            Maybe<String> slotValue,
                                            Maybe<NERTag> slotValueType,
                                            Set<String> docidsToForce,
                                            int maxDocuments,
                                            boolean officialIndexOnly) {
    return queryCoreMaps(Props.DB_TABLE_DOCUMENT_CACHE, Annotation.class, entityName, entityType, relation, slotValue, slotValueType, docidsToForce, maxDocuments, officialIndexOnly);
  }

  /**
   * Similar to querySentences, but only returns KBP DocIDs, and doesn't hit the CoreMap.
   *
   * @see KBPIR#querySentences(String, Maybe, Maybe, Maybe, Maybe, Set, int, boolean) querySentences
   */
  protected abstract List<String> queryDocIDs(String entityName,
                                              Maybe<NERTag> entityType,
                                              Maybe<String> relation,
                                              Maybe<String> slotValue,
                                              Maybe<NERTag> slotValueType,
                                              int maxDocuments,
                                              boolean officialIndexOnly);

  public abstract Set<String> getKnownRelationsForPair(KBPair pair);
  public abstract List<KBPSlotFill> getKnownSlotFillsForEntity(KBPEntity entity);


  //
  // Shared Data
  //

  /**
   * A cached version of the knowledge base.
   * This is to allow reading it from a simple serialized file, rather than having to parse
   * the XML each time.
   */
  protected static KnowledgeBase knowledgeBase = new KnowledgeBase();

  public KnowledgeBase getKnowledgeBase() {
    if (knowledgeBase == null || knowledgeBase.isEmpty()) {
      trainingData();
    }
    return knowledgeBase;
  }

  //
  // Helper if not already there
  //

  /**
   * Get the list of training tuples, from a list of files
   * Once the limit is reached, the tuples are returned
   * @param limit The number of tuples to read, or -1 to read all available tupels
   * @param files The TSV files to read the tuples from.
   * @return A list of KBTriple objects, corresponding to the (entity, relation, slotValue) triples found in the TSV file.
   */
  public List<KBTriple> trainingDataFromTSV(int limit, String... files) {
    List<KBTriple> tuples = new ArrayList<KBTriple>();
    for (String file:files) {
      readTuplesFromTSV(tuples, file, limit);
    }
    return tuples;
  }

  /** The tab character */
  private static final Pattern TAB_PATTERN = Pattern.compile("\\t");
  /**
   * Get the list of training tuples, from one TSV file.
   * Format of the file is tab delimited with fields: entityId, entityName, relationName, and slotValue
   * @return A list of KBTriple objects, corresponding to the (entity, relation, slotValue) triples found in the TSV file.
   */
  private List<KBTriple> readTuplesFromTSV(List<KBTriple> tuples, String filename, int limit) {
    try {
      BufferedReader bufferedReader = IOUtils.getBufferedFileReader(filename);
      String line;
      int count = 0;
      boolean tuplesCountReached = false;
      while ((line = bufferedReader.readLine()) != null) {
        String[] fields = TAB_PATTERN.split(line);
        count++;
        if (fields.length == 4) {
          // 0 is entityId, 1 is entityName, 2 is relationName, 3 is slotValue
          String entityId = fields[0];
          String entityName = fields[1];
          String relationName = fields[2];
          String slotValue = fields[3];
          NERTag entType = NERTag.fromRelation(relationName).orCrash("Unknown relation " + relationName);
          KBTriple t = KBPNew.entName(entityName).entType(entType).entId(entityId).slotValue(slotValue).rel(relationName).KBTriple();
          tuples.add(t);
          if (limit > 0 && tuples.size() >= limit) {
            tuplesCountReached = true;
            break;
          }
        } else {
          throw new RuntimeException("Error reading tuples from TSV: Invalid line at " + filename + ":" + count);
        }
      }
      log("Read " + tuples.size() + " from " + filename + ((tuplesCountReached)? " reached tupled count":""));
      bufferedReader.close();
      return tuples;
    } catch (IOException ex) {
      throw new RuntimeException("Error reading tuples from " + filename, ex);
    }
  }

  /**
   * Get the triples in the training data.
   * @return A list of KBTriples corresponding to the training triples (e.g., Obama, born_in, Hawaii)
   */
  public List<KBTriple> trainingData() {
    if (knowledgeBase.isEmpty()) {
      // Knowledge base is not cached -- start caching it
      List<KBTriple> tuples = new ArrayList<KBTriple>(Props.TRAIN_TUPLES_COUNT);
      tuples.addAll(trainingDataFromTSV(Props.TRAIN_TUPLES_COUNT, Props.TRAIN_TUPLES_FILES));
      for (KBTriple tuple: tuples) {
        knowledgeBase.put(KBPNew.from(tuple).KBPSlotFill());
      }
    }
    return knowledgeBase.triples();
  }

  /**
   * Returns known slot values for a given entity and relation. For example, Barack Obama and born_in should return Set(Hawaii).
   * @param entity The entity to query
   * @param rel The relation to fill with the slot fills
   * @return A set of known slot fills for the given entity and relation.
   */
  public Set<String> getKnownSlotValuesForEntityAndRelation(KBPEntity entity, RelationType rel) {
    List<KBPSlotFill> triples = getKnownSlotFillsForEntity(entity);
    Set<String> slotValues = new HashSet<String>();
    for (KBPSlotFill triple : triples) {
      if (triple.key.relationName.equals(rel.canonicalName)) {
        slotValues.add(triple.key.slotValue);
      }
    }
    return slotValues;
  }

  public List<CoreMap> querySentences(String entityName,
                                      NERTag entityType,
                                      String slotValue,
                                      String relation,
                                      NERTag slotValueType,
                                      int maxDocuments,
                                      boolean officialIndexOnly) {
    return querySentences(entityName, Maybe.Just(entityType),
        Maybe.Just(relation), Maybe.Just(slotValue),
        Maybe.Just(slotValueType), new HashSet<String>(),
        maxDocuments, officialIndexOnly);
  }

  public List<CoreMap> querySentences(String entityName, String slotValue, String reln, int n, boolean officialIndexOnly) {
    // Get IR retreivals
    List<CoreMap> sentences = querySentences(entityName, Maybe.<NERTag>Nothing(),
        Maybe.Just(reln), Maybe.Just(slotValue),
        Maybe.<NERTag>Nothing(), new HashSet<String>(), n, officialIndexOnly);
    // TODO(gabor) this is kind of a hack
    // In case the cache is stale, force the source index to be the correct index
    if (officialIndexOnly) {
      for (CoreMap sentence : sentences) {
        if (sentence.containsKey(CoreAnnotations.SentenceIndexAnnotation.class)) {
          sentence.set(KBPAnnotations.SourceIndexAnnotation.class, Props.INDEX_OFFICIAL.getPath());
        }
      }
    }
    // Return
    return sentences;
  }

  public List<CoreMap> querySentences(String entityName,
                                      NERTag entityType,
                                      String slotValue,
                                      String relation,
                                      NERTag slotValueType,
                                      int maxDocuments) {
    return querySentences(entityName, Maybe.Just(entityType),
        Maybe.Just(relation), Maybe.Just(slotValue),
        Maybe.Just(slotValueType), new HashSet<String>(),
        maxDocuments, false);
  }

  public List<CoreMap> querySentences( String entityName, NERTag entityType, int n  ) {
    return querySentences(entityName, Maybe.Just(entityType),
        Maybe.<String>Nothing(), Maybe.<String>Nothing(),
        Maybe.<NERTag>Nothing(), new HashSet<String>(), n, false);
  }

  public List<CoreMap> querySentences( String entityName, NERTag entityType, Set<String> docidsToForce, int n  ) {
    return querySentences(entityName, Maybe.Just(entityType),
        Maybe.<String>Nothing(), Maybe.<String>Nothing(),
        Maybe.<NERTag>Nothing(), docidsToForce, n, false);
  }

  public List<CoreMap> querySentences( String entityName, String slotValue, int n  ) {
    return querySentences(entityName, Maybe.<NERTag>Nothing(),
        Maybe.<String>Nothing(), Maybe.Just(slotValue),
        Maybe.<NERTag>Nothing(), new HashSet<String>(), n, false);
  }


  public List<CoreMap> querySentences( String entityName, NERTag entityType, String slotValue, int n  ) {
    return querySentences(entityName, Maybe.Just(entityType),
        Maybe.<String>Nothing(), Maybe.Just(slotValue),
        Maybe.<NERTag>Nothing(), new HashSet<String>(), n, false);
  }


  public List<CoreMap> querySentences( String entityName, NERTag entityType, String slotValue, String reln, int n  ) {
    return querySentences(entityName, Maybe.Just(entityType),
        Maybe.Just(reln), Maybe.Just(slotValue),
        Maybe.<NERTag>Nothing(), new HashSet<String>(), n, false);
  }

  public List<CoreMap> querySentences( String entityName, String slotValue, String reln, int n  ) {
    return querySentences(entityName, Maybe.<NERTag>Nothing(),
        Maybe.Just(reln), Maybe.Just(slotValue),
        Maybe.<NERTag>Nothing(), new HashSet<String>(), n, false);
  }

  public List<String> queryDocIDs( String entityName, String slotValue, String reln, int n  ) {
    return queryDocIDs(entityName, Maybe.<NERTag>Nothing(),
        Maybe.Just(reln), Maybe.Just(slotValue),
        Maybe.<NERTag>Nothing(), n, false);
  }

  public List<String> queryDocIDs( String entityName, String slotValue, int n  ) {
    return queryDocIDs(entityName, Maybe.<NERTag>Nothing(),
        Maybe.<String>Nothing(), Maybe.Just(slotValue),
        Maybe.<NERTag>Nothing(), n, false);
  }

  public List<String> queryDocIDs(String entityName, RelationType relation, int maxDocuments) {
    return queryDocIDs(entityName, Maybe.<NERTag>Nothing(),
        Maybe.Just(relation.canonicalName), Maybe.<String>Nothing(),
        Maybe.<NERTag>Nothing(), maxDocuments, false);
  }

  public List<String> queryDocIDs(String entityName, NERTag entityType, int maxDocuments) {
    return queryDocIDs(entityName, Maybe.Just(entityType),
            Maybe.<String>Nothing(), Maybe.<String>Nothing(),
            Maybe.<NERTag>Nothing(), maxDocuments, false);
  }

  public List<String> queryDocIDs(String entityName, NERTag entityType, Set<String> docidsToForce, int maxDocuments) {
    // Add the forced docids
    List<String> rtn = new ArrayList<String>(docidsToForce);
    // Add the queried docids
    List<String> docs = queryDocIDs(entityName, Maybe.Just(entityType),
        Maybe.<String>Nothing(), Maybe.<String>Nothing(),
        Maybe.<NERTag>Nothing(), maxDocuments, false);
    for (String doc : docs) {
      if (!docidsToForce.contains(doc)) {
        rtn.add(doc);
      }
    }
    return rtn;
  }

  public List<Annotation> queryDocuments( String entityName, String slotValue, String reln, int n  ) {
    return queryDocuments(entityName, Maybe.<NERTag>Nothing(),
        Maybe.Just(reln), Maybe.Just(slotValue),
        Maybe.<NERTag>Nothing(), new HashSet<String>(), n, false);
  }

  public List<Annotation> queryDocuments(String entityName, RelationType relation, int maxDocuments) {
    return queryDocuments(entityName, Maybe.<NERTag>Nothing(),
        Maybe.Just(relation.canonicalName), Maybe.<String>Nothing(),
        Maybe.<NERTag>Nothing(), new HashSet<String>(), maxDocuments, false);
  }

  public List<Annotation> queryDocuments(String entityName, NERTag entityType, int maxDocuments) {
    return queryDocuments(entityName, Maybe.Just(entityType),
        Maybe.<String>Nothing(), Maybe.<String>Nothing(),
        Maybe.<NERTag>Nothing(), new HashSet<String>(), maxDocuments, false);
  }

  public List<Annotation> queryDocuments(String entityName, NERTag entityType, Set<String> docidsToForce, int maxDocuments) {
    return queryDocuments(entityName, Maybe.Just(entityType),
        Maybe.<String>Nothing(), Maybe.<String>Nothing(),
        Maybe.<NERTag>Nothing(), docidsToForce, maxDocuments, false);
  }

  public Annotation fetchDocument(String docId) {
    return fetchDocument( docId, false );
  }

  protected static KBTriple toKBTriple( Pair<String, String> pair, String relation ) {
    NERTag entityType = NERTag.fromRelation( relation ).orCrash();
    return KBPNew.entName(pair.first).entType(entityType).entId(Maybe.<String>Nothing()).slotValue(pair.second).rel(relation).KBTriple();
  }
}
