package edu.stanford.nlp.kbp.slotfilling.process;

import edu.stanford.nlp.kbp.entitylinking.EntityLinkingAnnotators;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.tokensregex.PhraseTable;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.util.CoreMap;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Annotate documents with entity mentions in the document
 * combining information from NER, entity lists
 *
 * @author Angel Chang
 */
public class CombinedMentionAnnotator implements Annotator {
  private final LabeledChunkIdentifier chunkIdentifier;
  private final PhraseTable phraseTable;

  private final Class tokenTextKey =  CoreAnnotations.TextAnnotation.class;

  /** Mentions will be identified in a List<CoreMap> */
  //private final Class mentionsKey = CoreAnnotations.MentionsAnnotation.class;
  private final Class mentionsKey = EntityLinkingAnnotators.EntitiesAnnotation.class;

  /** Annotation keys that get added for each mention */
  private final Class entityIdKey   = EntityLinkingAnnotators.EntityIdAnnotation.class;    // Value will be a string
  private final Class entityNameKey = EntityLinkingAnnotators.EntityNameAnnotation.class;  // Value will be a string
  private final Class entityTypeKey = EntityLinkingAnnotators.EntityTypeAnnotation.class;  // Value will be a string
  private final Class entityIdScoresKey   = EntityLinkingAnnotators.EntityIdScoresAnnotation.class;  // Value will be a Counter<String>
  private final Class entityTypeScoresKey = EntityLinkingAnnotators.EntityIdScoresAnnotation.class;  // Value will be a Counter<String>

  boolean nonoverlappingOnly = false;

  //private final static String NELL_DIR = "C:\\code\\NLP\\kbp\\javanlp\\models";
  private final static String NELL_DIR = "/scr/nlp/data/tackbp2013/data/nell";

  public CombinedMentionAnnotator(String name, Properties props) {
    this.chunkIdentifier = new LabeledChunkIdentifier();

    // TODO: make option
    String[] nellFiles = new String[] {
      NELL_DIR + File.separator + "NELL.KBP2012.max3.v1.nps.csv.gz",
      NELL_DIR + File.separator + "NELL.KBP2012.min4.v1.nps.csv.gz"
    };
    this.phraseTable = new PhraseTable();
    try {
      for (String filename:nellFiles) {
        this.phraseTable.readPhrasesWithTagScores(filename);
      }
    } catch (IOException ex) {
      throw new RuntimeException(ex);
    }
  }

  @Override
  public void annotate(Annotation annotation) {
    List<CoreLabel> tokens = annotation.get(CoreAnnotations.TokensAnnotation.class);
    Integer annoTokenBegin = annotation.get(CoreAnnotations.TokenBeginAnnotation.class);
    if (annoTokenBegin == null) { annoTokenBegin = 0; }

    List<PhraseTable.PhraseMatch> matched =
            phraseTable.findAllMatches(new PhraseTable.TokenList(tokens, tokenTextKey));
    if (nonoverlappingOnly) {
      matched = phraseTable.findNonOverlappingPhrases(matched);
    }

    List<CoreMap> mentions = new ArrayList<CoreMap>(matched.size());
    for (PhraseTable.PhraseMatch match:matched) {
      Annotation chunk = ChunkAnnotationUtils.getAnnotatedChunk(annotation,
              match.getTokenBegin(), match.getTokenEnd());

      // Entity name
      String name = (String) CoreMapAttributeAggregator.MOST_FREQ.aggregate(
              CoreAnnotations.NormalizedNamedEntityTagAnnotation.class, chunk.get(CoreAnnotations.TokensAnnotation.class));
      if (name == null) {
        name = chunk.get(CoreAnnotations.TextAnnotation.class);
      } else {
        chunk.set(CoreAnnotations.NormalizedNamedEntityTagAnnotation.class, name);
      }
      chunk.set(entityNameKey, name);
      // Entity type
      String type = (String) CoreMapAttributeAggregator.MOST_FREQ.aggregate(
              CoreAnnotations.NamedEntityTagAnnotation.class, chunk.get(CoreAnnotations.TokensAnnotation.class));
      chunk.set(entityTypeKey, type);
      // Entity semantic types from nell
      chunk.set(entityTypeScoresKey, match.getPhrase().getData());

      // TODO: Entity Id, Entity Id scores (get from entitylinking)
      mentions.add(chunk);

    }

    // TODO: merge with chunks identified by NER
//    List<CoreMap> chunks = chunkIdentifier.getAnnotatedChunks(tokens, annoTokenBegin,
//            textKey, labelKey, tokenChunkKey, tokenLabelKey);
    annotation.set(mentionsKey, mentions);
  }

  @Override
  public Set<Requirement> requires() {
    return Collections.singleton(TOKENIZE_REQUIREMENT);
  }

  @Override
  public Set<Requirement> requirementsSatisfied() {
    // TODO: figure out what this produces
    return Collections.emptySet();
  }
}
