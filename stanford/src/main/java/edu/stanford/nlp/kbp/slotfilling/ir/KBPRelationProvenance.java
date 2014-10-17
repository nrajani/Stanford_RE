package edu.stanford.nlp.kbp.slotfilling.ir;

import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations;
import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.time.TimeAnnotations;
import edu.stanford.nlp.util.ArrayCoreMap;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.logging.PrettyLoggable;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import static edu.stanford.nlp.util.logging.Redwood.Util.debug;

/**
 * This class tracks the provenance of a particular entity of interest -- usually a Slot fill.
 *
 * @author Gabor Angeli
 */
public class KBPRelationProvenance implements Serializable {
  private static final long serialVersionUID = 1L;

  /** The document id we have found */
  public final String docId;
  /** The index name we searched in */
  public final String indexName;
  /** The index of the sentence in the document relevant to the relation */
  public final Maybe<Integer> sentenceIndex;
  /** The span of the entity in the sentence */
  public final Maybe<Span> entityMentionInSentence;
  /** The span of the slot fill in the sentence */
  public final Maybe<Span> slotValueMentionInSentence;
  /** The span of the slot fill in the sentence */
  public final Maybe<Span> justificationMention;
  /** Optionally, the sentence in which this slot fill occurs */
  public  Maybe<CoreMap> containingSentenceLossy;

  /** Confidence score assigned by the classifier */
  public final Maybe<Double> score;


  public Maybe<EntityContext> toSlotContext(KBPEntity entity, Maybe<KBPIR> ir) {
    if (!ir.isDefined()) { return Maybe.Nothing(); }
    if (!sentenceIndex.isDefined()) { return Maybe.Nothing(); }
    if (!slotValueMentionInSentence.isDefined()) { return Maybe.Nothing(); }
    Annotation doc = ir.get().fetchDocument(this.docId);
    List<CoreMap> sentences = doc.get(CoreAnnotations.SentencesAnnotation.class);
    int sentenceOffset = 0;
    for (int i = 0; i < sentenceIndex.get(); ++i) {
      sentenceOffset += sentences.get(i).get(CoreAnnotations.TokensAnnotation.class).size();
    }
    return Maybe.Just(new EntityContext(entity, doc,
        new Span(slotValueMentionInSentence.get().start() + sentenceOffset, slotValueMentionInSentence.get().end() + sentenceOffset)));
  }

  public KBPRelationProvenance(String docId, String indexName) {
    this.docId = docId != null ? docId.trim() : docId;
    this.indexName = indexName != null ? indexName.trim() : indexName;
    this.sentenceIndex = Maybe.Nothing();
    this.entityMentionInSentence = Maybe.Nothing();
    this.slotValueMentionInSentence = Maybe.Nothing();
    this.containingSentenceLossy = Maybe.Nothing();
    this.justificationMention=null;
    if (isOfficial()) {
      debug("Provenance", "creating official provenance without sentence level information");
    }
    this.score = Maybe.Nothing();
  }

  public KBPRelationProvenance(String docId, String indexName, int sentenceIndex, Span entityMention, Span slotValue,
                               CoreMap containingSentence, Maybe<Double> score) {
    this.docId = docId != null ? docId.trim() : docId;
    this.indexName = indexName != null ? indexName.trim() : indexName;
    this.sentenceIndex = Maybe.Just(sentenceIndex);
    this.entityMentionInSentence = Maybe.Just(entityMention);
    this.slotValueMentionInSentence = Maybe.Just(slotValue);
    this.justificationMention=null;
    assert slotValue != null;
    // Save some useful information from the containing sentence.
    // Be careful: a bunch of these get serialized as datums.
    this.containingSentenceLossy = Maybe.Just(containingSentence);
    this.score = score;
  }
  
  public KBPRelationProvenance(String docId, String indexName, int sentenceIndex, Span entityMention, Span slotValue,Span justification,
          CoreMap containingSentence, Maybe<Double> score) {
	  this.docId = docId != null ? docId.trim() : docId;
	  this.indexName = indexName != null ? indexName.trim() : indexName;
	  this.sentenceIndex = Maybe.Just(sentenceIndex);
	  this.entityMentionInSentence = Maybe.Just(entityMention);
	  this.slotValueMentionInSentence = Maybe.Just(slotValue);
	  this.justificationMention=Maybe.Just(justification);
	  assert slotValue != null;
	  // Save some useful information from the containing sentence.
	  // Be careful: a bunch of these get serialized as datums.
	  this.containingSentenceLossy = Maybe.Just(containingSentence);
	  this.score = score;
}


  public KBPRelationProvenance(String docId, String indexName, int sentenceIndex, Span entityMention, Span slotValue,
                               CoreMap containingSentence) {
    this(docId, indexName, sentenceIndex, entityMention, slotValue, containingSentence, Maybe.<Double>Nothing());
  }


  public PrettyLoggable loggable(final KBPIR querier) {
    if (true) { throw new IllegalStateException("This requires IR"); }
    return new PrettyLoggable() {
      @Override
      public void prettyLog(Redwood.RedwoodChannels channels, String description) {
        if (sentenceIndex.isDefined() && entityMentionInSentence.isDefined() && slotValueMentionInSentence.isDefined()) {
          try {
            // Get offsets
            if (!(querier instanceof StandardIR)) {
              throw new IllegalArgumentException("Not sure how to fetch documents by docid through anything but a KBPIR");
            }
            Annotation doc = null; // was: ((StandardIR) querier).officialIndex.fetchDocument(docId);
            String text = doc.get(CoreAnnotations.TextAnnotation.class);
            CoreMap sentence = doc.get(CoreAnnotations.SentencesAnnotation.class).get(sentenceIndex.get());
            channels.log("         " + CoreMapUtils.sentenceToProvenanceString(text, sentence, entityMentionInSentence.get(), slotValueMentionInSentence.get()).replaceAll("\n", " ").replaceAll("\\s+", " "));
          } catch (RuntimeException e) {
            channels.log(e);
          }
        } else {
          channels.log("  <no sentence provenance>");
        }
      }
    };
  }

  /**
   * Compute a relation provenance from a sentence and KBTriple.
   * If possible, this will also populate the in-sentence provenance.
   * @param sentence The sentence we are using for provenance
   * @param key The KBTriple representing the entities we would like to obtain provenance on
   * @return A provenance object with as much information filled in as possible
   */
  public static Maybe<KBPRelationProvenance> compute(CoreMap sentence, KBTriple key) {
    String docid = sentence.get(CoreAnnotations.DocIDAnnotation.class);
    assert docid != null;
    String indexName = sentence.get(KBPAnnotations.SourceIndexAnnotation.class);
    assert indexName != null;

    KBPRelationProvenance rtn = null;
    List<RelationMention> relationMentions = sentence.get(MachineReadingAnnotations.RelationMentionsAnnotation.class);
    if (relationMentions == null) { relationMentions = new LinkedList<RelationMention>(); }
    for (RelationMention rel : relationMentions) {
      assert (rel instanceof NormalizedRelationMention);
      NormalizedRelationMention normRel = (NormalizedRelationMention) rel;
      KBPEntity entity = normRel.getNormalizedEntity();
      String slotValue = normRel.getNormalizedSlotValue();
      List<EntityMention> args = normRel.getEntityMentionArgs();
      EntityMention leftArg = args.get(0);
      EntityMention rightArg = args.get(1);
      if ((entity == null || key.getEntity().equals(entity)) && slotValue.trim().equalsIgnoreCase(key.slotValue.trim())) {
        Integer sentenceIndex = leftArg.getSentence().get(CoreAnnotations.SentenceIndexAnnotation.class);
        Span entitySpan = leftArg.getExtent();
        Span slotValueSpan = rightArg.getExtent();
        if (sentenceIndex != null) {
          rtn = new KBPRelationProvenance(docid, indexName, sentenceIndex, entitySpan, slotValueSpan, sentence);
        }
      }
    }

    return Maybe.fromNull(rtn);
  }

  public boolean isOfficial() {
    return isOfficialIndex(this.indexName);
  }

  public static boolean isOfficialIndex(String indexName) {
    //noinspection SimplifiableIfStatement
    if (Props.INDEX_OFFICIAL == null) { return false; } // for JUnit tests; this should never be null
    return indexName != null && indexName.toLowerCase().endsWith(Props.INDEX_OFFICIAL.getName().toLowerCase());
  }

  public KBPRelationProvenance rewrite(double score) {
    return new KBPRelationProvenance(docId, indexName, sentenceIndex.get(), entityMentionInSentence.get(), slotValueMentionInSentence.get(), containingSentenceLossy.get(), Maybe.Just(score));
  }

}
