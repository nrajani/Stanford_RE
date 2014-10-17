package edu.stanford.nlp.kbp.slotfilling.common;

import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations;
import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.Label;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.process.AbstractTokenizer;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.ArrayCoreMap;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Function;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.logging.Color;

import java.io.PrintStream;
import java.io.StringReader;
import java.io.UnsupportedEncodingException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;

/**
 * A utility class for manipulating CoreMaps
 *
 * @author Gabor Angeli
 */
@SuppressWarnings("UnusedDeclaration")
public class CoreMapUtils {

  /**
   * This is a custom copy function and does some funky things.
   *  * Make a shallow Copy everything in the document EXCEPT the
   *  sentences
   *  * Make a new copy of the sentence, with a shallow copy of
   *  everything EXCEPT the tokens.
   *  * Make a new copy of the tokens.
   */
  public static Annotation copyDocument( Annotation document ) {
    Annotation newDocument = new Annotation( document );
    // Change the sentence annotation
    List<CoreMap> sentences = document.get( CoreAnnotations.SentencesAnnotation.class );
    List<CoreMap> newSentences = new ArrayList<CoreMap>( sentences.size() );
    for( CoreMap sentence : sentences ) {
      newSentences.add( copySentence( sentence ) );
    }
    newDocument.set( CoreAnnotations.SentencesAnnotation.class, newSentences );
    newDocument.set(CoreAnnotations.DocIDAnnotation.class, document.get(CoreAnnotations.DocIDAnnotation.class));
    return newDocument;
  }

  public static CoreMap copySentence( CoreMap sentence ) {
    CoreMap newSentence = new ArrayCoreMap( sentence );
    List<CoreLabel> tokens = sentence.get( CoreAnnotations.TokensAnnotation.class );
    List<CoreLabel> newTokens = new ArrayList<CoreLabel>( tokens.size() );
    for( CoreLabel token : tokens ) {
      newTokens.add( new CoreLabel(token) );
    }
    newSentence.set( CoreAnnotations.TokensAnnotation.class, newTokens );
    return newSentence;
  }

  /**
   * I'd say that words cannot express how much I hate Java's verbosity,
   * but the number of words in this class kind of does.
   * Don't get confused -- this is just storing 6 integers.
   */
  private static class SentenceTextOffsets {
    public final int sentenceBegin;
    public final int sentenceEnd;
    public final int entityBegin;
    public final int entityEnd;
    public final int slotFillBegin;
    public final int slotFillEnd;

    public SentenceTextOffsets(int sentenceBegin, int sentenceEnd, int entityBegin, int entityEnd, int slotFillBegin, int slotFillEnd) {
      this.sentenceBegin = sentenceBegin;
      this.sentenceEnd = sentenceEnd;
      this.entityBegin = entityBegin;
      this.entityEnd = entityEnd;
      this.slotFillBegin = slotFillBegin;
      this.slotFillEnd = slotFillEnd;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;
      SentenceTextOffsets that = (SentenceTextOffsets) o;
      return entityBegin == that.entityBegin && entityEnd == that.entityEnd && sentenceBegin == that.sentenceBegin && sentenceEnd == that.sentenceEnd && slotFillBegin == that.slotFillBegin && slotFillEnd == that.slotFillEnd;

    }

    @Override
    public int hashCode() {
      int result = sentenceBegin;
      result = 31 * result + sentenceEnd;
      result = 31 * result + entityBegin;
      result = 31 * result + entityEnd;
      result = 31 * result + slotFillBegin;
      result = 31 * result + slotFillEnd;
      return result;
    }
  }

  public static String sentenceToProvenanceString(CoreMap sentence) {
    if (sentence.containsKey(KBPAnnotations.EntitySpanAnnotation.class) &&
        sentence.containsKey(KBPAnnotations.SlotValueSpanAnnotation.class)) {
      return sentenceToProvenanceString(sentence, sentence.get(KBPAnnotations.EntitySpanAnnotation.class), sentence.get(KBPAnnotations.SlotValueSpanAnnotation.class));
    } else {
      return sentenceToMinimalString(sentence);
    }
  }

  public static String sentenceToProvenanceString(CoreMap sentence, Span entitySpan, Span slotValueSpan) {
    StringBuilder b = new StringBuilder();
    if (sentence.containsKey(CoreAnnotations.OriginalTextAnnotation.class)) {
      // Case: we can reliably recreate the highlighting with character offsets
      for (int i = 0; i < sentence.get(CoreAnnotations.TokensAnnotation.class).get(0).get(CoreAnnotations.CharacterOffsetBeginAnnotation.class); ++i) {
        b.append("~");
      }
      b.append(sentence.get(CoreAnnotations.OriginalTextAnnotation.class));
      return sentenceToProvenanceString(b.toString(), sentence, entitySpan, slotValueSpan);
    } else {
      // Case: revert back to token offsets
      for (int i = 0; i < sentence.get(CoreAnnotations.TokensAnnotation.class).size(); ++i) {
        CoreLabel token = sentence.get(CoreAnnotations.TokensAnnotation.class).get(i);
        String word = token.containsKey(CoreAnnotations.OriginalTextAnnotation.class) ? token.originalText() : token.word();
        if (entitySpan.contains(i) || slotValueSpan.contains(i)) {
          b.append(Color.CYAN.apply(word));
        } else {
          b.append(word);
        }
        b.append(" ");
      }
      return b.toString();
    }
  }

  public static String sentenceToProvenanceString(String text, CoreMap sentence, Span entitySpan, Span slotValueSpan) {
    // Get offsets
    int sentenceStart = sentence.containsKey(CoreAnnotations.CharacterOffsetBeginAnnotation.class)
        ? sentence.get(CoreAnnotations.CharacterOffsetBeginAnnotation.class)
        : sentence.get(CoreAnnotations.TokensAnnotation.class).get(0).get(CoreAnnotations.CharacterOffsetBeginAnnotation.class);
    int sentenceEnd = sentence.containsKey(CoreAnnotations.CharacterOffsetEndAnnotation.class)
        ? sentence.get(CoreAnnotations.CharacterOffsetEndAnnotation.class)
        : sentence.get(CoreAnnotations.TokensAnnotation.class).get(sentence.get(CoreAnnotations.TokensAnnotation.class).size() - 1).get(CoreAnnotations.CharacterOffsetEndAnnotation.class);
    int entityStart = sentence.get(CoreAnnotations.TokensAnnotation.class).get(entitySpan.start()).get(CoreAnnotations.CharacterOffsetBeginAnnotation.class);
    int entityEnd = sentence.get(CoreAnnotations.TokensAnnotation.class).get(entitySpan.end() - 1).get(CoreAnnotations.CharacterOffsetEndAnnotation.class);
    int slotFillStart = sentence.get(CoreAnnotations.TokensAnnotation.class).get(slotValueSpan.start()).get(CoreAnnotations.CharacterOffsetBeginAnnotation.class);
    int slotFillEnd = sentence.get(CoreAnnotations.TokensAnnotation.class).get(slotValueSpan.end() - 1).get(CoreAnnotations.CharacterOffsetEndAnnotation.class);
    SentenceTextOffsets offsets = new SentenceTextOffsets(sentenceStart, sentenceEnd, entityStart, entityEnd, slotFillStart, slotFillEnd);

    // Make things pretty
    StringBuilder b = new StringBuilder();
    int firstBreak = StrictMath.min(offsets.entityBegin, offsets.slotFillBegin);
    int secondBreak = StrictMath.max(offsets.entityBegin, offsets.slotFillBegin);
    // (begin)
    if (firstBreak - 25 > offsets.sentenceBegin) {
      b.append("...");
      b.append(text.substring(Math.max(0, firstBreak - 25), firstBreak));
    } else {
      b.append(text.substring(offsets.sentenceBegin, firstBreak));
    }
    if (entitySpan.contains(slotValueSpan) || slotValueSpan.contains(entitySpan)) {
      b.append(Color.CYAN.apply(text.substring(
          Math.min(offsets.entityBegin, offsets.slotFillBegin),
          Math.max(offsets.entityEnd, offsets.slotFillEnd))));
    } else {
      // (first entity of interest)
      if (firstBreak == offsets.entityBegin) {
        b.append(Color.CYAN.apply(text.substring(offsets.entityBegin, offsets.entityEnd)));
      } else {
        b.append(Color.CYAN.apply(text.substring(offsets.slotFillBegin, offsets.slotFillEnd)));
      }
      // (middle)
      b.append(text.substring(Math.min(offsets.entityEnd, offsets.slotFillEnd),
          Math.max(offsets.entityBegin, offsets.slotFillBegin)));
      // (second entity of interest)
      if (secondBreak == offsets.entityBegin) {
        b.append(Color.CYAN.apply(text.substring(offsets.entityBegin, offsets.entityEnd)));
      } else {
        b.append(Color.CYAN.apply(text.substring(offsets.slotFillBegin, offsets.slotFillEnd)));
      }
    }
    // (end)
    int endOfEntities = Math.max(offsets.entityEnd, offsets.slotFillEnd);
    if (endOfEntities + 25 < offsets.sentenceEnd) {
      b.append(text.substring(endOfEntities, Math.min(endOfEntities + 25, text.length())));
      b.append("...");
    } else {
      b.append(text.substring(endOfEntities, Math.min(offsets.sentenceEnd, text.length())));
    }

    // Print
    return b.toString();
  }

  public static String sentenceToString(CoreMap sentence) {
    return sentenceToString(sentence, true, true, true, true,
        true, true, true, false);
  }

  public static String sentenceToMinimalString(CoreMap sentence) {
    return sentenceToString(sentence, true,
        false, false, false,
        false, false, false, false);
  }

  public static String sentenceToNERString(CoreMap sentence) {
    return sentenceToString(sentence, true,
        false, true, false,
        false, true, false, false);
  }

  public static String sentenceToString(CoreMap sentence,
      boolean showText,
      boolean showTokens,
      boolean showEntities,
      boolean showSlots,
      boolean showRelations,
      boolean showDocId,
      boolean showDatetime) {
    return sentenceToString(sentence, showText, showTokens, showEntities, showSlots, showRelations, showDocId, showDatetime, false);
  }

  public static String sentenceToString(CoreMap sentence,
      boolean showText,
      boolean showTokens,
      boolean showEntities,
      boolean showSlots,
      boolean showRelations,
      boolean showDocId,
      boolean showDatetime,
      boolean showParseTree) {
    StringBuilder os = new StringBuilder();
    boolean justText = showText && ! showTokens && ! showEntities && ! showSlots && ! showRelations && ! showDocId && ! showDatetime;

    //
    // Print text and tokens
    //
    if (sentence == null) { return "(null)"; }
    List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
    if(tokens != null){
      if(showText){
        if(! justText) os.append("TEXT: ");
        boolean first = true;
        for(CoreLabel token: tokens) {
          if(! first) os.append(" ");
          os.append(token.word());
          first = false;
        }
        if(! justText) os.append("\n");
      }
      if (showTokens) {
        os.append("TOKENS:");
        String[] tokenAnnotations = {
            "Text", "PartOfSpeech", "NamedEntityTag" , "Antecedent"
        };
        for (CoreLabel token: tokens) {
          os.append(' ');
          os.append(token.toShortString(tokenAnnotations));
        }
        os.append('\n');
      }
    }

    Tree tree = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
    if (tree != null && showParseTree) {
      os.append(tree.toString());
      os.append('\n');
    }

    //
    // Print EntityMentions
    //
    List<EntityMention> ents = sentence.get(MachineReadingAnnotations.EntityMentionsAnnotation.class);
    if (ents != null && showEntities) {
      os.append("ENTITY MENTIONS:\n");
      for (EntityMention e: ents){
        os.append('\t');
        os.append(e);
        os.append('\n');
      }
    }

    //
    // Print SlotMentions
    //
    List<EntityMention> slots = sentence.get(KBPAnnotations.SlotMentionsAnnotation.class);
    if (slots != null && showSlots) {
      os.append("SLOT CANDIDATES:\n");
      for(EntityMention e: slots){
        os.append("\t").append(e.toString()).append("\n");
      }
    }

    //
    // Print RelationMentions
    //
    List<RelationMention> relations = sentence.get(MachineReadingAnnotations.RelationMentionsAnnotation.class);
    if(relations != null && showRelations){
      os.append("RELATION MENTIONS:\n");
      for(RelationMention r: relations){
        os.append(r.toString()).append("\n");
      }
    }

    String docId = sentence.get(CoreAnnotations.DocIDAnnotation.class);
    if (docId != null && showDocId) {
      os.append("DOCID: ").append(docId).append("\n");
    }

    String datetime = sentence.get(KBPAnnotations.DatetimeAnnotation.class);
    if (datetime != null && showDatetime) {
      os.append("DATETIME: ").append(datetime).append("\n");
    }

    os.append("\n");
    return os.toString();
  }

  public static String sentenceToAMTString(CoreMap sentence, boolean amtStringOnly)
  {
      StringBuffer os = new StringBuffer();

      List<CoreLabel> tokens = sentence.get(CoreAnnotations.TokensAnnotation.class);
      if (tokens != null) {
          os.append("TEXT: ");
          for (CoreLabel token : tokens) {
              os.append(token.word());
          }
          os.append('\n');

          os.append("TOKENS:");
          String[] tokenAnnotations = { "Text", "PartOfSpeech", "NameEntityTag", "Antecedent" };

          for (CoreLabel token : tokens) {
              os.append(" ").append(token.toShortString(tokenAnnotations));
          }
          os.append('\n');
      }

      List<EntityMention> ents = sentence.get(MachineReadingAnnotations.EntityMentionsAnnotation.class);
      if (ents != null) {
          os.append("ENTITY MENTIONS:\n");
          for (EntityMention e : ents)
          {
              os.append('\t');
              os.append(e);
              os.append('\n');
          }
      }

      List<EntityMention> slots = sentence.get(KBPAnnotations.SlotMentionsAnnotation.class);
      if (slots != null) {
          os.append("SLOT CANDIDATES:\n");
          for (EntityMention e : slots) {
              os.append("\t").append(e.toString()).append("\n");
          }
      }

      List<RelationMention> relations = sentence.get(MachineReadingAnnotations.RelationMentionsAnnotation.class);
      if (relations != null) {
          os.append("RELATION MENTIONS:\n");
          for (RelationMention r : relations) {
              os.append(r.toString()).append("\n");
          }
      }

      if (amtStringOnly) {
          os = new StringBuffer();
          if (relations != null) {
              for (RelationMention r : relations) {
                  os.append(r.toString()).append("\n");

              }
          }
          os.append("\n");
          return os.toString();
      }

      os.append("\n");
      return os.toString();
  }

  /**
   * Prints a pipelined sentence in a format easy on human eyes
   */
  public static void printSentence(PrintStream os, CoreMap sentence) {
    os.print(sentenceToString(sentence));
  }

  public static String documentToMinimalString(Annotation annotation) {
    return StringUtils.join(
        CollectionUtils.map(annotation.get(CoreAnnotations.SentencesAnnotation.class),
            new Function<CoreMap, String>() {
              @Override
              public String apply(CoreMap sentence) {
                return sentenceToMinimalString(sentence);
              }
            })
    );
  }

  /**
   * Tokenizes a string using our default tokenizer
   */
  public static List<CoreLabel> tokenize(String text) {
    AbstractTokenizer<CoreLabel> tokenizer;
    CoreLabelTokenFactory tokenFactory = new CoreLabelTokenFactory();
    // note: ptb3Escaping must be true because this is the default behavior in the pipeline
    // String options = "ptb3Escaping=false";
    String options="";
    StringReader sr = new StringReader(text);
    tokenizer = new PTBTokenizer<CoreLabel>(sr, tokenFactory, options);
    return tokenizer.tokenize();
  }

  public static String [] tokenizeToStrings(String text) {
    List<CoreLabel> tokens = tokenize(text);
    String [] stringToks = new String[tokens.size()];
    for(int i = 0; i < tokens.size(); i ++) stringToks[i] = tokens.get(i).word();
    return stringToks;
  }

  /**
   * Verifies if the tokens in needle are contained in the tokens in haystack.
   * @param caseInsensitive Perform case insensitive matching of token texts if true
   */
  public static boolean contained(List<CoreLabel> needle, List<CoreLabel> haystack, boolean caseInsensitive) {
    int index = findStartIndex(needle, haystack, caseInsensitive);
    return (index >= 0);
  }

  public static int findStartIndex(List<CoreLabel> needle, List<CoreLabel> haystack, boolean caseInsensitive) {
    return findStartIndex(needle, haystack, caseInsensitive, 0);
  }

  public static int findStartIndex(List<CoreLabel> needle, List<CoreLabel> haystack, boolean caseInsensitive, int searchStart) {
    for(int start = searchStart; start <= haystack.size() - needle.size(); start ++){
      boolean failed = false;
      for(int i = 0; i < needle.size(); i ++){
        String n = needle.get(i).word();
        String h = haystack.get(start + i).word();
        if(caseInsensitive && ! n.equalsIgnoreCase(h)){
          failed = true;
          break;
        }
        if(! caseInsensitive && ! n.equals(h)){
          failed = true;
          break;
        }
      }
      if(! failed) return start;
    }
    return -1;
  }

  public static String sentenceSpanString(CoreMap tokens, Span span) {
    return sentenceSpanString(tokens.get(CoreAnnotations.TokensAnnotation.class), span);
  }

  public static String sentenceSpanString(List<CoreLabel> tokens, Span span) {
    StringBuilder os = new StringBuilder();
    for(int i = span.start(); i < span.end(); i ++){
    	System.out.println("span vals: "+i);
      if(i > span.start()) os.append(' ');
      os.append(tokens.get(i).word());
    }
    return os.toString();
  }

  /* Beware of changing me! You may invalidate the sentencegloss cache */
  public static String getHexKeyString(String toHash) {
    return getHexKeyString(toHash, Integer.MAX_VALUE);
  }

  /* Beware of changing me! You may invalidate the sentencegloss cache */
  public static String getHexKeyString( String toHash, int size ) {
    try {
      MessageDigest md = MessageDigest.getInstance("SHA-256");
      byte[] digested;
      try {
        digested = md.digest(toHash.getBytes("UTF8"));
      } catch (UnsupportedEncodingException e) {
        throw new RuntimeException(e);
      }

      StringBuilder sb = new StringBuilder();

      for (byte hexKeyByte : digested) {
        sb.append(Integer.toString((hexKeyByte & 0xff) + 0x100, 16).substring(1));
      }

      return sb.toString().substring(0, Math.min(size, sb.toString().length()));
    } catch (NoSuchAlgorithmException e) {
      throw new RuntimeException(e);
    }
  }

  /* Beware of changing me! You may invalidate the sentencegloss cache */
  public static String getSentenceGlossKey(String sentencegloss, String entityGloss, String valueGloss) {
    return getHexKeyString(sentencegloss.replaceAll("\\s+", " ").trim()) + ":" + getHexKeyString(entityGloss.trim(), 7) + ":" + getHexKeyString(valueGloss.trim(), 7);
  }

  /* Beware of changing me! You may invalidate the sentencegloss cache */
  public static String getSentenceGlossKey(List<CoreLabel> sentencegloss, String entityGloss, String valueGloss) {
    StringBuilder sentenceString = new StringBuilder();
    for (CoreLabel token : sentencegloss) {
      if (token.containsKey(CoreAnnotations.OriginalTextAnnotation.class)) {
        sentenceString.append(token.originalText().trim()).append(" ");
      } else {
        sentenceString.append(token.word().trim()).append(" ");
      }
    }
    return getSentenceGlossKey(sentenceString.toString().trim(), entityGloss, valueGloss);
  }


  public static String phraseToOriginalString(List<CoreLabel> coreLabels) {
    StringBuilder b = new StringBuilder();
    for (CoreLabel token : coreLabels) {
      String word;
      if (token.containsKey(CoreAnnotations.OriginalTextAnnotation.class)) {
        word = token.originalText();
      } else {
        word = token.word().replace("-LRB-", "(").replace("-RRB-", ")")
            .replace("-LCB-", "{").replace("-RCB-", "}")
            .replace("-LSB-", "[").replace("-RSB-", "]");
      }
      b.append(word);
      b.append(" ");
    }
   return b.toString().trim().replaceAll(" 's ", "'s ");
  }

  /**
   * Converts the tree labels to CoreLabels.
   * We need this because we store additional info in the CoreLabel, like token span.
   * @param tree The tree to convert to augmented core labels
   */
  public static void convertToCoreLabels(Tree tree) {
    Label l = tree.label();
    if(! (l instanceof CoreLabel)){
      CoreLabel cl = new CoreLabel();
      cl.setValue(l.value());
      tree.setLabel(cl);
    }

    for (Tree kid : tree.children()) {
      convertToCoreLabels(kid);
    }
  }
}
