package edu.stanford.nlp.kbp.slotfilling.ir.query;

import edu.stanford.nlp.kbp.slotfilling.common.NERTag;
import edu.stanford.nlp.kbp.slotfilling.common.Maybe;
import edu.stanford.nlp.kbp.slotfilling.common.NERTag;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IterableIterator;
import edu.stanford.nlp.util.Pair;

import java.util.Set;

/**
 * An abstract specification of a querier, as a function from a query to sentences.
 *
 * Note that not all queriers will return coherent documents (e.g., web snippets)
 *
 * @author Gabor Angeli
 */
public interface Querier {
  /**
   * Query a collection of sentences based on a specification of things to query for.
   * Note that many terms could be undefined -- these are denoted with {@link edu.stanford.nlp.kbp.slotfilling.common.Maybe}s.
   * @param entityName The query entity name
   * @param entityType The type of the query entity
   * @param relation The relation to query for, as per RelationType.name (not to be confused with RelationType.name())
   * @param slotValue The slot value to query for
   * @param slotValueType The slot type to query for.
   * @param maxDocuments The maximum number of documents to query.
   * @return An iterable of sentences, along with their scores (e.g., Lucene scores). Often, these documents are lazily loaded.
   */
  public IterableIterator<Pair<CoreMap, Double>> querySentences(String entityName,
                                                                Maybe<NERTag> entityType,
                                                                Maybe<String> relation,
                                                                Maybe<String> slotValue,
                                                                Maybe<NERTag> slotValueType,
                                                                Set<String> docidsToForce,
                                                                Maybe<Integer> maxDocuments);

  /** Perform any close actions which may be relevant. This includes closing Lucene, file handles, etc. */
  public void close();
}
