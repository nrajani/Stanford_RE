package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.kbp.slotfilling.common.KBPOfficialEntity;
import edu.stanford.nlp.kbp.slotfilling.common.KBPSlotFill;
import edu.stanford.nlp.kbp.slotfilling.common.Maybe;

import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * @author Gabor Angeli
 */
public interface Evaluator {

  public List<KBPOfficialEntity> testEntities();

  public Maybe<KBPScore> evaluate(Map<KBPOfficialEntity, Collection<KBPSlotFill>> relations);
}
