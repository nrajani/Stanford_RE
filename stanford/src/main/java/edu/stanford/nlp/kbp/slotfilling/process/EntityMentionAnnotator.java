package edu.stanford.nlp.kbp.slotfilling.process;


import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations.*;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.logging.Redwood;

import java.util.*;

/**
 * Annotates EntityMentions in a set of sentences. This should be run post IR
 */
public class EntityMentionAnnotator implements Annotator {
  private static Redwood.RedwoodChannels logger = Redwood.channels("EntityAnn");

  public final KBPEntity entity;

  public EntityMentionAnnotator(KBPEntity queryEntity) {
    this.entity = queryEntity;
  }

  @Override
  public void annotate(Annotation annotation) {
    for (CoreMap sentence : annotation.get(SentencesAnnotation.class)) {
      // Set Entity Mentions Annotation
      List<EntityMention> entityMentions = extractEntityMentions(entity, sentence);
      sentence.set(EntityMentionsAnnotation.class, entityMentions);
    }
  }

  @SuppressWarnings("unchecked")
  @Override
  public Set<Requirement> requirementsSatisfied() {
    return new HashSet<Requirement>(Arrays.asList(new Requirement[]{new Requirement("postir.entitymentions")}));
  }

  @Override
  public Set<Requirement> requires() {
    return new HashSet<Requirement>();
  }

  public static List<EntityMention> extractEntityMentions(KBPEntity entity, CoreMap sentence) {
    List<EntityMention> entities = new ArrayList<EntityMention>();
    List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
    String[] entityTokens = entity.name.split("\\s+");
    assert tokens != null;

    // (1) Look for exact matches of the query entity
    Set<Span> literalMatches = new HashSet<Span>();
    int entityIndex = 0;
    // Loop over tokens
    for (int i = 0; i < tokens.size(); ++i) {
      assert i >= 0;
      assert entityIndex <= i;
      // Logic for string match
      CoreLabel token = tokens.get(i);
      if (entityTokens[entityIndex].equalsIgnoreCase(token.originalText()) ||
          entityTokens[entityIndex].equalsIgnoreCase(token.word())) {
        entityIndex += 1;
      } else {
        entityIndex = 0;
      }
      if (entityIndex >= entityTokens.length) {
        // Case: found the entity
        // Create a mention
        Span entitySpan = new Span(i + 1 - entityIndex, i + 1);
        EntityMention em = new EntityMention(
            Utils.makeEntityMentionId(entity instanceof KBPOfficialEntity ? ((KBPOfficialEntity) entity).id : Maybe.<String>Nothing()),
            sentence, entitySpan, entitySpan, Utils.makeNERTag(entity.type), null, null);
        assert em.getType() != null && !em.getType().equals("") && !em.getType().equals(Props.NER_BLANK_STRING);
        em.setNormalizedName(entity.name);
        logger.debug("found entity mention [direct]: " + em);
        entities.add(em);
        // Mark span as filled
        literalMatches.add(entitySpan);
        // Reset entity index
        entityIndex = 0;
      }
    }

    // (2) Look for coreferent entities, which are not already [partially or completely] absorbed
    // Step i: find possible entity starts
    Set<Integer> candidateCorefStarts = new HashSet<Integer>();
    boolean lastTokenWasEntity = false;
    for (int i = 0; i < tokens.size(); ++i) {
      CoreLabel token = tokens.get(i);
      if (token.containsKey(AntecedentAnnotation.class) && token.get(AntecedentAnnotation.class).contains(entity.name)) {
        if (!lastTokenWasEntity) { candidateCorefStarts.add(i); }
        lastTokenWasEntity = true;
      } else {
        lastTokenWasEntity = false;
      }
    }
    // Step ii: expand the starts into spans
    Set<Span> corefSpans = new HashSet<Span>();
    for (int candidateStart : candidateCorefStarts) {
      // Create span
      int end = candidateStart + 1;
      while (end < tokens.size() &&
          tokens.get(end).containsKey(AntecedentAnnotation.class) &&
          tokens.get(end).get(AntecedentAnnotation.class).contains(entity.name)) {
        end += 1;
      }
      Span corefEntitySpan = new Span(candidateStart, end);
      // Check to make sure span is valid
      if (!Span.overlaps(corefEntitySpan, literalMatches)) {
        corefSpans.add(corefEntitySpan);
      }
    }
    // Step iii: construct and add mentions
    for (Span corefSpan : corefSpans) {
      EntityMention em = new EntityMention(
          Utils.makeEntityMentionId(entity instanceof KBPOfficialEntity ? ((KBPOfficialEntity) entity).id : Maybe.<String>Nothing()),
          sentence, corefSpan, corefSpan, Utils.makeNERTag(entity.type), null, null);
      assert em.getType() != null && !em.getType().equals("") && !em.getType().equals(Props.NER_BLANK_STRING);
      em.setNormalizedName(entity.name);
      logger.debug("found entity mention [coref]: " + em);
      entities.add(em);
    }

    // (3) Return
    return entities;
  }
}


