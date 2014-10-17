package edu.stanford.nlp.kbp.slotfilling.common;

import edu.stanford.nlp.ie.machinereading.structure.ExtractionObject;
import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.util.CoreMap;

import java.util.List;

/**
 * Stores a normalized version of the slot value, which may be different from the actual extent of the argument
 * @author Mihai
 *
 */
public class NormalizedRelationMention extends RelationMention {
  private static final long serialVersionUID = 1L;

  /**
   * Normalized value of the slot (getArg(1))
   * This may be different from getArg(1).getExtentString() because we allow approximate matches in KBPReader
   */
  private String normalizedSlotValue;
  /**
   * The normalized entity name, from the knowledge base
   */
  private final KBPEntity normEntity;

  public NormalizedRelationMention(
      String normSlot,
      KBPEntity normEntity,
      String objectId, 
      CoreMap sentence,
      Span span,
      String type,
      String subtype,
      List<ExtractionObject> args,
      List<String> argNames) {
    super(objectId, sentence, span, type, subtype, args, argNames);
    this.normalizedSlotValue = normSlot;
    this.normEntity = normEntity;
  }
  
  public String getNormalizedSlotValue() { return normalizedSlotValue; }
  
  public KBPEntity getNormalizedEntity() { return normEntity; }
}
