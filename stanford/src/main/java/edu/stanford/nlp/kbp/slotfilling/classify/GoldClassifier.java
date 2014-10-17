package edu.stanford.nlp.kbp.slotfilling.classify;

import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.evaluate.CustomSFScore;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * If a sentence is passed in with the correct entity and slot value,
 * from the correct document, return the known label for that entity.
 *
 * @author Gabor Angeli
 */
public class GoldClassifier extends HeuristicRelationExtractor {

  public final Collection<Triple<KBPOfficialEntity, RelationType, String>> goldResponses;
  public final Map<KBPair, Collection<Pair<String, Integer>>> memorizedResponses;

  @SuppressWarnings("UnusedDeclaration") // Used by ModelType via reflection
  public GoldClassifier(Properties props) {
    this(getGoldResponses(DataUtils.testEntities(Props.TEST_QUERIES.getPath(), Maybe.<KBPIR>Nothing())));
  }

  public GoldClassifier(Collection<Triple<KBPOfficialEntity, RelationType, String>> goldResponses) {
    this.goldResponses = goldResponses;
    this.memorizedResponses = new HashMap<KBPair, Collection<Pair<String, Integer>>>();
    for (Triple<KBPOfficialEntity, RelationType, String> triple : goldResponses) {
      KBPair key = KBPNew.entName(triple.first.name).entType(triple.first.type).slotValue(triple.third).KBPair();
      if (!memorizedResponses.containsKey(key)) { memorizedResponses.put(key, new HashSet<Pair<String,Integer>>()); }
      memorizedResponses.get(key).add(Pair.makePair(triple.second.canonicalName, 0));
    }

  }

  @Override
  public Collection<Pair<String, Integer>> extractRelations(KBPair key, CoreMap[] input) {
    key = KBPNew.entName(key.entityName).entType(key.entityType).slotValue(key.slotValue).KBPair();
    if (memorizedResponses.containsKey(key)) {
      return memorizedResponses.get(key);
    }
    for (CoreMap sentence : input) {
      for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
        if (token.containsKey(CoreAnnotations.AntecedentAnnotation.class)) {
          KBPair key1 = KBPNew.entName(token.get(CoreAnnotations.AntecedentAnnotation.class)).entType(key.entityType).slotValue(key.slotValue).KBPair();
          @SuppressWarnings("UnusedDeclaration") KBPair key2 = KBPNew.from(key.getEntity()).slotValue(token.get(CoreAnnotations.AntecedentAnnotation.class)).KBPair();
          if (memorizedResponses.containsKey(key1)) {
            return memorizedResponses.get(key1);
          }
//          if (memorizedResponses.containsKey(key2)) {
//            return memorizedResponses.get(key2);
//          }  // don't want coreferent slot fills
        }
      }
    }
    return new ArrayList<Pair<String, Integer>>();
  }

  private static Collection<Triple<KBPOfficialEntity, RelationType, String>> getGoldResponses(Collection<KBPOfficialEntity> queries) {
    Collection<Triple<KBPOfficialEntity, RelationType, String>> responses = new LinkedList<Triple<KBPOfficialEntity, RelationType, String>>();

    File keyFile = Props.TEST_RESPONSES;//("Cannot use gold classifier without gold responses specified");
    Map<String, KBPOfficialEntity> idToEntity = new HashMap<String, KBPOfficialEntity>();
    Map<String, String> goldIdToEntityId = new HashMap<String, String>();
    for (KBPOfficialEntity entity : queries) {
      goldIdToEntityId.put(entity.queryId.orCrash(), entity.id.orCrash());
      idToEntity.put(entity.id.orCrash(), entity);
    }

    try {
      BufferedReader keyReader = new BufferedReader(new FileReader(keyFile));
      String line;
      while ((line = keyReader.readLine()) != null) {
        // Read line
        String[] fields = line.split("\t", 11);
        assert fields.length == 11;
        // Get fields
        String entityId = goldIdToEntityId.get(fields[1]);
        String relation = fields[3];
        String slotValue = fields[8];
        int judgement = Integer.parseInt(fields[10]);
        // Add response
        if (judgement == CustomSFScore.CORRECT) {

          KBPOfficialEntity entity = entityId == null ? null : idToEntity.get(entityId);
          if (entity != null && entity.name != null) // Ignore entities without mapping to test entity set
          {
            // angelx - slot values shouldn't have _ ??
            slotValue = slotValue.replaceAll("_", " ");

            Triple<KBPOfficialEntity, RelationType, String> key = Triple.makeTriple(entity, RelationType.fromString(relation).orCrash(), slotValue);

            responses.add(key);
          }
        }
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    return responses;
  }


}
