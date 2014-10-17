package edu.stanford.nlp.kbp.slotfilling.process;

import edu.stanford.nlp.kbp.slotfilling.common.KBPEntity;
import edu.stanford.nlp.util.CoreMap;

import java.util.List;

/**
 * Annotate various facets.
 * A nice interface; most of the functionality is in KBPReader currently
 */
public interface DocumentAnnotator {

  public List<CoreMap> annotateSentenceFeatures( KBPEntity entity, List<CoreMap> sentences );

}
