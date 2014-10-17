package edu.stanford.nlp.kbp.slotfilling.ir;

import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.util.logging.Redwood;

import java.util.*;

/**
 * Quick interface to query documents (annotated and featurized).
 */
public class StandardIR extends KBPIR {

  protected static final Redwood.RedwoodChannels logger = Redwood.channels("IR");

  @Override
  protected <E extends CoreMap> List<E> queryCoreMaps(String tableName, Class<E> expectedOutput, String entityName, Maybe<NERTag> entityType, Maybe<String> relation, Maybe<String> slotValue, Maybe<NERTag> slotValueType, Set<String> docidsToForce, int maxDocuments, boolean officialIndexOnly) {
    return new ArrayList<E>();
  }

  @Override
  public Annotation fetchDocument(String docId, boolean officialIndexOnly) {
    return new Annotation("foo");
  }

  @Override
  public int queryNumHits(Collection<String> terms) {
    return -1;
  }

  @Override
  protected List<String> queryDocIDs(String entityName, Maybe<NERTag> entityType, Maybe<String> relation, Maybe<String> slotValue, Maybe<NERTag> slotValueType, int maxDocuments, boolean officialIndexOnly) {
    return new ArrayList<String>();
  }

  @Override
  public Set<String> getKnownRelationsForPair(KBPair pair) {
    return new HashSet<String>();
  }

  @Override
  public List<KBPSlotFill> getKnownSlotFillsForEntity(KBPEntity entity) {
    return new ArrayList<KBPSlotFill>();
  }
}
