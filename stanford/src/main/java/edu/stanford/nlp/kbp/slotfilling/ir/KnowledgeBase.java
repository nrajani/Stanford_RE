package edu.stanford.nlp.kbp.slotfilling.ir;

import edu.stanford.nlp.kbp.slotfilling.common.*;

import java.util.*;
import java.io.Serializable;

/**
 */
public class KnowledgeBase implements Serializable {

  public final Map<KBPEntity,Set<KBPSlotFill>> data;
  public final Map<String, Set<KBPSlotFill>> dataByName;

  public KnowledgeBase() {
    data = new HashMap<KBPEntity,Set<KBPSlotFill>>();
    dataByName = new HashMap<String, Set<KBPSlotFill>>();
  }

  public void put(List<KBPSlotFill> facts) {
    for( KBPSlotFill fact : facts ) {
      put(fact);
    }
  }
  public void put(KBPSlotFill fact) {
    KBPEntity entity = fact.key.getEntity();
    if(!data.containsKey(entity)) {
      data.put(entity, new HashSet<KBPSlotFill>());
      dataByName.put(entity.name, new HashSet<KBPSlotFill>());
    }
    data.get(entity).add(fact);
    dataByName.get(entity.name).add(fact);
  }

  public Maybe<Set<KBPSlotFill>> get(KBPEntity entity) {
    if( data.containsKey(entity) ) {
      return Maybe.Just(data.get(entity));
    } else if (dataByName.containsKey(entity.name)) {
      return Maybe.Just(dataByName.get(entity.name));
    } else {
      return Maybe.Nothing();
    }
  }

  public boolean isEmpty() {
    return data.isEmpty();
  }

  public List<KBTriple> triples() {
    List<KBTriple> triples = new ArrayList<KBTriple>();
    for (Set<KBPSlotFill> fills : data.values()) {
      for (KBPSlotFill fill : fills) {
        triples.add(fill.key);
      }
    }
    Collections.sort(triples);
    return triples;
  }
}
