package edu.stanford.nlp.kbp.slotfilling.common;

import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ling.CoreAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.DocIDAnnotation;
import edu.stanford.nlp.util.ErasureUtils;
import edu.stanford.nlp.util.IntPair;
import edu.stanford.nlp.util.Pair;

import java.util.List;
import java.util.Map;
import java.util.Set;

public class KBPAnnotations {
  
  private KBPAnnotations() {} // only static members

  /**
   * The CoreMap key for getting the slot mentions corresponding to a sentence.
   * 
   * This key is typically set on sentence annotations.
   */
  public static class SlotMentionsAnnotation implements CoreAnnotation<List<EntityMention>> {
    public Class<List<EntityMention>> getType() {
      return ErasureUtils.uncheckedCast(List.class);
    }
  }

  /**
   * This class indicates which index a particular sentence came from.
   * Should be set on a sentence level.
   */
  public static class SourceIndexAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {
      return String.class;
    }
  }

  /**
   * Stores the Lucene (integer) document ID. You can use this to obtain the
   * full Lucene Document for a sentence by retrieving this document from an
   * IndexSearcher. This should not be confused with {@link DocIDAnnotation}
   * which is generally what you want instead of this annotation.
   * Typically set on the sentence level.
   */
  public static class SourceIndexDocIDAnnotation implements CoreAnnotation<Integer> {
    public Class<Integer> getType() {
      return Integer.class;
    }
  }

  public static class DatetimeAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() {
      return String.class;
    }
  }

  /**
   * This annotation indicates the positions where the text has been marked
   * (either entity or slot filler)
   */
  public static class MarkedPositionsAnnotation implements CoreAnnotation<List<IntPair>> {
    public Class<List<IntPair>> getType() {
      return ErasureUtils.uncheckedCast(List.class);
    }
  }

  // Attaches to sentences
  public static class EntitySpanAnnotation implements CoreAnnotation<Span> {
    public Class<Span> getType() {
      return ErasureUtils.uncheckedCast(Span.class);
    }
  }

  // Attaches to sentences
  public static class SlotValueSpanAnnotation implements CoreAnnotation<Span> {
    public Class<Span> getType() {
      return ErasureUtils.uncheckedCast(Span.class);
    }
  }

  // Attaches to sentences
  public static class AllAntecedentsAnnotation implements CoreAnnotation<Set<String>> {
    public Class<Set<String>> getType() { return ErasureUtils.uncheckedCast(Set.class); }
  }

  // Attaches to sentences or tokens
  public static class IsCoreferentAnnotation implements CoreAnnotation<Boolean> {
    public Class<Boolean> getType() { return Boolean.class; }
  }

  /**
   * A mapping from terms in AllAntecedentsAnnotation to alternate names for those terms.
   * The key should match an entity in {@link AllAntecedentsAnnotation}; the span should be a
   * token (not character) offset span in the sentence it is attached to.
   *
   * @see edu.stanford.nlp.kbp.slotfilling.ir.PostIRAnnotator
   */
  // Attaches to sentences
  public static class AlternateNamesAnnotation implements CoreAnnotation<Map<String, Set<Span>>> {
    public Class<Map<String, Set<Span>>> getType() { return ErasureUtils.uncheckedCast(Map.class); }
  }

  /** An annotation for the span of the canonical entity mention in the document. The integer is the sentence index, and the span is the token span */
  // Attaches to documents
  public static class CanonicalEntitySpanAnnotation implements CoreAnnotation<Pair<Integer,Span>> {
    public Class<Pair<Integer,Span>> getType() { return ErasureUtils.uncheckedCast(Pair.class); }
  }

  /** An annotation for the span of the canonical slot value mention in the document. The integer is the sentence index, and the span is the token span */
  // Attaches to documents
  public static class CanonicalSlotValueSpanAnnotation implements CoreAnnotation<Pair<Integer,Span>> {
    public Class<Pair<Integer, Span>> getType() { return ErasureUtils.uncheckedCast(Pair.class); }
  }
}
