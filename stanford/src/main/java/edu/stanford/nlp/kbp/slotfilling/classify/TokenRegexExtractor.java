package edu.stanford.nlp.kbp.slotfilling.classify;

import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.ling.CoreAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.tokensregex.CoreMapExpressionExtractor;
import edu.stanford.nlp.ling.tokensregex.TokenSequencePattern;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;

import java.io.File;
import java.util.*;

/**
 * A relation extractor making use of simple token regex patterns
 *
 * @author Gabor Angeli
 */
public class TokenRegexExtractor extends HeuristicRelationExtractor {

  private final Map<RelationType, CoreMapExpressionExtractor> rules = new HashMap<RelationType, CoreMapExpressionExtractor>();

  public TokenRegexExtractor() {
    // Create extractors
    for (RelationType rel : RelationType.values()) {
      if (new File(Props.TRAIN_TOKENREGEX_DIR + File.separator + rel.canonicalName + ".rules").exists()) {
        rules.put(rel, CoreMapExpressionExtractor.createExtractorFromFiles(TokenSequencePattern.getNewEnv(),
            Props.TRAIN_TOKENREGEX_DIR + File.separator + "defs.rules",
            Props.TRAIN_TOKENREGEX_DIR + File.separator + rel.canonicalName + ".rules"));
      }
    }
  }

  public TokenRegexExtractor(@SuppressWarnings("UnusedParameters") Properties props) {
    this();
  }

  @Override
  public Collection<Pair<String,Integer>> extractRelations(KBPair key, CoreMap[] input) {
    /// Sanity Check
    if (Utils.assertionsEnabled()) {
      for (CoreMap sentence : input) {
        for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
          assert !token.containsKey(KBPEntity.class);
          assert !token.containsKey(KBPSlotFill.class);
        }
      }
    }
    // Annotate Sentence
    for (CoreMap sentence : input) {
      // Annotate where the entity is
      for (EntityMention entityMention : sentence.get(MachineReadingAnnotations.EntityMentionsAnnotation.class)) {
        if ((entityMention.getValue() != null && entityMention.getValue().equalsIgnoreCase(key.entityName)) ||
            (entityMention.getNormalizedName() != null && entityMention.getNormalizedName().equalsIgnoreCase(key.entityName))) {
          for (int i = entityMention.getExtentTokenStart(); i < entityMention.getExtentTokenEnd(); ++i) {
            sentence.get(CoreAnnotations.TokensAnnotation.class).get(i).set(KBPEntity.class, "true");
          }
        }
      }
      // Annotate where the slot fill is
      for (EntityMention slotMention : sentence.get(KBPAnnotations.SlotMentionsAnnotation.class)) {
        if ((slotMention.getValue() != null && slotMention.getValue().replaceAll("\\\\", "").equals(key.slotValue)) ||
            (slotMention.getNormalizedName() != null && slotMention.getNormalizedName().equalsIgnoreCase(key.slotValue))) {
          for (int i = slotMention.getExtentTokenStart(); i < slotMention.getExtentTokenEnd(); ++i) {
            sentence.get(CoreAnnotations.TokensAnnotation.class).get(i).set(KBPSlotFill.class, "true");
          }
        }
      }
    }
    // Run Rules
    Set<Pair<String,Integer>> output = new HashSet<Pair<String, Integer>>();
    relationLoop: for (RelationType rel : RelationType.values()) {
      if (rules.containsKey(rel)) {
        CoreMapExpressionExtractor extractor = rules.get(rel);
        for (int sentI = 0; sentI < input.length; ++sentI) {
          CoreMap sentence = input[sentI];
          List extractions = extractor.extractExpressions(sentence);
          if (extractions != null && extractions.size() > 0) {
            output.add(Pair.makePair(rel.canonicalName, sentI));
            continue relationLoop;
          }
        }
      }
    }
    // Un-Annotate Sentence
    for (CoreMap sentence : input) {
      for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
        token.remove(KBPEntity.class);
        token.remove(KBPSlotFill.class);
      }
    }
    return output;
  }

  public static class KBPEntity implements CoreAnnotation<String> {
    public Class<String> getType() { return String.class; }
  }

  public static class KBPSlotFill implements CoreAnnotation<String> {
    public Class<String> getType() { return String.class; }
  }
}
