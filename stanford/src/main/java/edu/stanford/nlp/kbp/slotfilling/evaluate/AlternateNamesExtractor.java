package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.ie.machinereading.structure.Span;

import java.util.*;

/**
 * Extract alternate names based on coreference cues.
 *
 * Return all coreferent entities that have more than a certain number of occurences in
 * *distinct* sentences.
 *
 * @author Gabor Angeli
 */
public class AlternateNamesExtractor {

  private static KBPSlotFill mkSlotFill(KBPEntity entity, final CoreMap sentence, final int spanBegin, final int spanEnd,
                                        boolean isSubsidiary) {
    // Create the slot value
    String slotValue = "";
    for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class).subList(spanBegin, spanEnd)) {
      if (token.containsKey(CoreAnnotations.OriginalTextAnnotation.class)) {
        slotValue += token.originalText() + " ";
      } else {
        slotValue += token.word() + " ";
      }
    }
    slotValue = slotValue.trim();
    // Create the key
    KBTriple key = KBPNew.from(entity).slotValue(slotValue).slotType(entity.type)  // copy entity's type, since it's an alternate name
        .rel(isSubsidiary ? RelationType.ORG_SUBSIDIARIES.canonicalName : (entity.type == NERTag.PERSON ? RelationType.PER_ALTERNATE_NAMES.canonicalName : RelationType.ORG_ALTERNATE_NAMES.canonicalName)).KBTriple();
    // Create the provenance
    KBPRelationProvenance provenance = null;
    if (sentence.containsKey(KBPAnnotations.SourceIndexAnnotation.class) &&
        sentence.containsKey(CoreAnnotations.DocIDAnnotation.class) &&
        sentence.containsKey(CoreAnnotations.SentenceIndexAnnotation.class)) {
      provenance = new KBPRelationProvenance(
          sentence.get(CoreAnnotations.DocIDAnnotation.class),
          sentence.get(KBPAnnotations.SourceIndexAnnotation.class),
          sentence.get(CoreAnnotations.SentenceIndexAnnotation.class),
          new Span(spanBegin, spanEnd),
          new Span(spanBegin, spanEnd),
          sentence);

    }
    // Create slot fill
    return KBPNew.from(key).provenance(Maybe.fromNull(provenance)).score(1.0).KBPSlotFill();
  }

  private static boolean containsDuplicateTokens(List<CoreLabel> tokens, Span alternateNameSpan) {
    Set<String> seenTokens = new HashSet<String>();
    for (int k : alternateNameSpan) {
      if (tokens.get(k).tag().startsWith("N")) {
        if (seenTokens.contains(tokens.get(k).word())) { return true; }
        seenTokens.add(tokens.get(k).word());
      }
    }
    return false;
  }

  private static boolean isSameNEType(List<CoreLabel> tokens, Span alternateNameSpan, NERTag fillType) {
    for (int k : alternateNameSpan) {
      if (!tokens.get(k).ner().equals(fillType.name)) { return false; }
    }
    return true;
  }

  @SuppressWarnings("UnusedDeclaration")
  public static Collection<KBPSlotFill> extractSlotsViaCoref(KBPEntity entity, List<CoreMap> sentences) {
    Counter<KBPSlotFill> alternateNamesByValue = new ClassicCounter<KBPSlotFill>();
    int literalMentionsCount = 0;
    for (CoreMap sentence : sentences) {
      List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
      int spanBegin = -1;
      Set<KBTriple> addedInSentenceAlready = new HashSet<KBTriple>();
      for (int i = 0; i < tokens.size(); i++) {
        CoreLabel token = tokens.get(i);
        if (token.containsKey(CoreAnnotations.AntecedentAnnotation.class) && token.get(CoreAnnotations.AntecedentAnnotation.class).equals(entity.name) &&
            ((entity.type == NERTag.PERSON && token.ner().equals(NERTag.PERSON.name)) ||
             (entity.type == NERTag.ORGANIZATION && token.ner().equals(NERTag.ORGANIZATION.name)))) {
          // Case: this is a token in an alternate name
          if (spanBegin == -1) { spanBegin = i; }
        } else {
          if (spanBegin != -1 && (spanBegin < i - 1 || tokens.get(spanBegin).tag().startsWith("N"))) {  // was in a span, and it's more than a single word (or is a noun)
            // Case: end of an alternate name
            KBPSlotFill candidate = mkSlotFill(entity, sentence, spanBegin, i, false);
            if (candidate.key.slotValue.equalsIgnoreCase(entity.name) ||
                entity.name.startsWith(candidate.key.slotValue) || entity.name.endsWith(candidate.key.slotValue))  {
              // case: exact or approximate match with the literal mention
              literalMentionsCount += 1;
            } else {
              // case: a sufficiently different mention
              alternateNamesByValue.incrementCount(mkSlotFill(entity, sentence, spanBegin, i, false), addedInSentenceAlready.contains(candidate.key) ? 0.0 : 1.0);
              addedInSentenceAlready.add(candidate.key);
            }
          }
          // Case: clear current token
          spanBegin = -1;
        }
      }
    }

    double threshold = ((double) literalMentionsCount + alternateNamesByValue.totalCount()) * Props.TEST_RULES_ALTERNATENAMES_FRACTION;
    return Counters.keysAbove(alternateNamesByValue, threshold);
  }


  public static Collection<KBPSlotFill> extractSlotsViaReadingAlternateNamesAnnotation(KBPEntity entity, List<CoreMap> sentences) {
    Collection<KBPSlotFill> alternateNames = new ArrayList<KBPSlotFill>();
    Set<String> foundNames = new HashSet<String>();
    for (CoreMap sentence : sentences) {
      Map<String, Set<Span>> alternateNamesInSentence = sentence.get(KBPAnnotations.AlternateNamesAnnotation.class);
      if (alternateNamesInSentence != null && alternateNamesInSentence.containsKey(entity.name)) {
        for (Span alternateNameSpan : alternateNamesInSentence.get(entity.name)) {
          String gloss = CoreMapUtils.sentenceSpanString(sentence.get(CoreAnnotations.TokensAnnotation.class), alternateNameSpan);
          String normalizedGloss = gloss.toLowerCase().replaceAll("\\.", "").replaceAll(",", "").replaceAll("\\s+", " ").trim();
          if (alternateNameSpan.size() <= 10 && !foundNames.contains(normalizedGloss) &&
              !containsDuplicateTokens(sentence.get(CoreAnnotations.TokensAnnotation.class), alternateNameSpan) &&
              isSameNEType(sentence.get(CoreAnnotations.TokensAnnotation.class), alternateNameSpan, entity.type)) {
            foundNames.add(normalizedGloss);
            // Found an alternate name
            // Determine if it's really a subsidiary
            boolean isSubsidiary = false;
            if (entity.type == NERTag.ORGANIZATION) {
              for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class).subList(alternateNameSpan.start(), alternateNameSpan.end())) {
                String word = token.containsKey(CoreAnnotations.OriginalTextAnnotation.class) ? token.originalText().toLowerCase() : token.word().toLowerCase();
                if (entity.name.toLowerCase().contains(word)) { continue; }
                // below here, we are only considering words which do not occur in the
                // target entity. That is, the extra words.
                if (word.equals("of") || word.equals("in") ||
                    Utils.geography().isValidRegion(word) || Utils.geography().isValidCountry(word)) { isSubsidiary = true; }
              }
            }
            // Return it
            alternateNames.add(mkSlotFill(entity, sentence, alternateNameSpan.start(), alternateNameSpan.end(), isSubsidiary));
          }
        }
      }
    }

    return alternateNames;
  }

  public static Collection<KBPSlotFill> extractSlots(KBPEntity entity, List<CoreMap> sentences) {
    Collection<KBPSlotFill> alternateNames = extractSlotsViaReadingAlternateNamesAnnotation(entity, sentences);
    if (Props.TEST_RULES_ALTERNATENAMES_COREF) {
      alternateNames.addAll(extractSlotsViaCoref(entity, sentences));
    }
    return alternateNames;
  }
}
