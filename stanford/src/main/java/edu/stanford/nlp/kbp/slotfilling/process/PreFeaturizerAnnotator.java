package edu.stanford.nlp.kbp.slotfilling.process;

import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations.*;
import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.ie.machinereading.structure.RelationMention;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.common.KBPAnnotations.*;
import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.parser.lexparser.ParserAnnotations;
import edu.stanford.nlp.parser.lexparser.ParserConstraint;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.pipeline.ParserAnnotator;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.HeadFinder;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.regex.Pattern;

import static edu.stanford.nlp.util.logging.Redwood.Util.endTrack;
import static edu.stanford.nlp.util.logging.Redwood.Util.startTrack;

/**
 * This used to be GenericDocumentPreProcessor.
 * The class runs pre-processing tasks which need to be done before
 * featurization.
 *
 * As best I can tell, the two functions this class performs is:
 *   (1) fixing up the parse trees for relevant sentences, e.g. assigning
 *       reliable head tokens
 *   (2) Selecting relevant sentences
 *
 * @author Gabor Angeli
 */
public class PreFeaturizerAnnotator implements Annotator {
  private static Redwood.RedwoodChannels logger = Redwood.channels("KBPParse");
  private static final HeadFinder headFinder = new NoPunctuationHeadFinder();
  private static Annotator parserProcessorOrNull = null;

  public final Properties props;
  public final boolean enforceAtLeastOneEntityInSentence;

  public PreFeaturizerAnnotator(Properties props) {
    this(props, true);
  }

  public PreFeaturizerAnnotator(Properties props, boolean enforceAtLeastOneEntityInSentence) {
    this.props = props;
    this.enforceAtLeastOneEntityInSentence = enforceAtLeastOneEntityInSentence;
  }

  private static final List<TriggerSeq> triggers = new ArrayList<TriggerSeq>() {{
    try {
      BufferedReader is = IOUtils.getBufferedReaderFromClasspathOrFileSystem(Props.INDEX_RELATIONTRIGGERS.getPath());
      String line;
      while ((line = is.readLine()) != null) {
        line = line.trim();
        int firstTab = line.indexOf('\t');
        if (firstTab < 0) {
          firstTab = line.indexOf(' ');
        }
        assert (firstTab > 0 && firstTab < line.length());
        String label = line.substring(0, firstTab).trim();
        List<CoreLabel> tokens = CoreMapUtils.tokenize(line.substring(firstTab).trim());
        String[] words = new String[tokens.size()];
        for (int i = 0; i < tokens.size(); i++)
          words[i] = tokens.get(i).word();
        add(new TriggerSeq(label, words));
      }
      is.close();

      // make sure trigger sequences are sorted in descending order of length so
      // we always match the longest sequence first
    } catch (IOException e) {
      logger.err(e);
    }
  }};
  static { Collections.sort(triggers); }

  /**
   * Take a dataset Annotation, generate their parse trees and identify syntactic heads (and head spans, if necessary).
   * Also, filters out sentences without any relation mentions.
   */
  @Override
  public void annotate(Annotation annotation) {
    List<CoreMap> sentencesWithExamples = new ArrayList<CoreMap>();

    for (CoreMap sentence : annotation.get(SentencesAnnotation.class)) {
      // add to corpus if any RelationMentions created
      if (sentence.get(TokensAnnotation.class).size() <= 150 &&
          sentence.get(RelationMentionsAnnotation.class) != null &&
          sentence.get(EntityMentionsAnnotation.class) != null &&
          !sentence.get(RelationMentionsAnnotation.class).isEmpty()) {
        assert sentence.get(SlotMentionsAnnotation.class) != null && !sentence.get(SlotMentionsAnnotation.class).isEmpty();
        markTriggerWords(sentence.get(TokensAnnotation.class));  // mark trigger words
        sentencesWithExamples.add(sentence);
      } else {
        logger.debug("nothing in sentence (entities=" + sentence.get(EntityMentionsAnnotation.class).size() + ", values=" + sentence.get(SlotMentionsAnnotation.class).size() + "): " + CoreMapUtils.sentenceToMinimalString(sentence));
      }
    }

    startTrack("Pre-processing the corpus");
    if (Props.KBP_VERBOSE) logger.debug("processing " + sentencesWithExamples.size() + " sentences.");
    for (CoreMap sentence : sentencesWithExamples) {
      List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
      if (Props.KBP_VERBOSE) logger.debug("processing sentence " + tokens);
      Tree tree = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
      if(tree == null) throw new RuntimeException("ERROR: MR requires full syntactic analysis!");

      // convert tree labels to CoreLabel if necessary
      // we need this because we store additional info in the CoreLabel, such as the spans of each tree
      CoreMapUtils.convertToCoreLabels(tree);

      // store the tree spans, if not present already
      tree.indexSpans(0);
      if (Props.KBP_VERBOSE) logger.debug("index spans were generated.");
      if (Props.KBP_VERBOSE) logger.debug("parse tree using CoreLabel:\n" + tree.pennString());

      //
      // now match all entity mentions against the syntactic tree
      //
      List<EntityMention> allMentions = new ArrayList<EntityMention>(
          sentence.containsKey(EntityMentionsAnnotation.class) ? sentence.get(EntityMentionsAnnotation.class) : new ArrayList<EntityMention>());
      allMentions.addAll(
      sentence.containsKey(SlotMentionsAnnotation.class) ? sentence.get(SlotMentionsAnnotation.class) : new ArrayList<EntityMention>());
      for (EntityMention ent : allMentions) {
        if (Props.KBP_VERBOSE) logger.debug("finding head for entity: " + ent);
        int headPos = assignSyntacticHead(ent, tree, tokens);
        if (Props.KBP_VERBOSE) logger.debug("syntactic head of mention \"" + ent + "\" is: " + tokens.get(headPos).word());

        assert ent.getExtent() != null;
        assert ent.getHead() != null;
        assert ent.getSyntacticHeadTokenPosition() >= 0;
      }
      // Sanity check (we've annotated every argument for every relation)
      for (RelationMention rm : sentence.get(RelationMentionsAnnotation.class)) {
        assert allMentions.contains(rm.getEntityMentionArgs().get(0));
        assert allMentions.contains(rm.getEntityMentionArgs().get(1));
      }
    }
    endTrack("Pre-processing the corpus");

    logger.debug("found " + sentencesWithExamples.size() + " relevant sentences");
    annotation.set(SentencesAnnotation.class, sentencesWithExamples);  // set new sentences
  }

  @Override
  public Set<Requirement> requirementsSatisfied() {
    return new HashSet<Requirement>(Arrays.asList(new Requirement[]{new Requirement("postir.parse")}));
  }

  @Override
  public Set<Requirement> requires() {
    return new HashSet<Requirement>(Arrays.asList(new Requirement[]{new Requirement("postir.relationmentions")}));
  }


  /**
   * Find the index of the head of an entity.
   *
   * @param ent The entity mention
   * @param tree The Tree for the entire sentence in which it occurs.
   * @param tokens The Sentence in which it occurs
   * @return The index of the entity head
   */
  private int assignSyntacticHead(EntityMention ent, Tree tree, List<CoreLabel> tokens) {
    if (ent.getSyntacticHeadTokenPosition() != -1) {
      return ent.getSyntacticHeadTokenPosition();
    }

//    logger.debug("finding syntactic head for entity: " + ent + " in tree: " + tree.toString());
//    logger.debug("flat sentence is: " + tokens);
    Tree sh = null;
    try {
      sh = findSyntacticHead(ent, tree, tokens);
    } catch(Exception e) {
      logger.err(e);
    } catch(AssertionError e) {
      logger.err(e);
    }

    int headPos = ent.getExtentTokenEnd() - 1;
    if(sh != null){
      CoreLabel label = (CoreLabel) sh.label();
      headPos = label.get(BeginIndexAnnotation.class);
    } else {
      logger.debug("WARNING: failed to find syntactic head for entity: " + ent + " in tree: " + tree);
      logger.debug("fallback strategy: will set head to last token in mention: " + tokens.get(headPos));
    }
    ent.setHeadTokenPosition(headPos);

    return headPos;
  }

  /**
   * Finds the syntactic head of the given entity mention.
   *
   * @param ent The entity mention
   * @param root The Tree for the entire sentence in which it occurs.
   * @param tokens The Sentence in which it occurs
   * @return The tree object corresponding to the head. This MUST be a child of root.
   *     It will be a leaf in the parse tree.
   */
  private Tree findSyntacticHead(EntityMention ent, Tree root, List<CoreLabel> tokens) {
//    logger.debug("searching for tree matching " + ent);
    Tree exactMatch = findTreeWithSpan(root, ent.getExtentTokenStart(), ent.getExtentTokenEnd());

    //
    // found an exact match
    //
    if (exactMatch != null) {
      return safeHead(exactMatch);
    }

    // no exact match found
    // in this case, we parse the actual extent of the mention, embedded in a sentence
    // context, so as to make the parser work better :-)

    int approximateness = 0;
    List<CoreLabel> extentTokens = new ArrayList<CoreLabel>();
    extentTokens.add(initCoreLabel("It"));
    extentTokens.add(initCoreLabel("was"));
    assert extentTokens.get(0).containsKey(ValueAnnotation.class);
    assert extentTokens.get(1).containsKey(ValueAnnotation.class);
    final int ADDED_WORDS = 2;
    for (int i = ent.getExtentTokenStart(); i < ent.getExtentTokenEnd(); i++) {
      // Add everything except separated dashes! The separated dashes mess with the parser too badly.
      CoreLabel label = tokens.get(i);
      if ( ! "-".equals(label.word())) {
        extentTokens.add(tokens.get(i));
      } else {
        approximateness++;
      }
    }
    extentTokens.add(initCoreLabel("."));

    // constrain the parse to the part we're interested in.
    // Starting from ADDED_WORDS comes from skipping "It was".
    // -1 to exclude the period.
    // We now let it be any kind of nominal constituent, since there
    // are VP and S ones
    ParserConstraint constraint = new ParserConstraint();
    constraint.start = ADDED_WORDS;
    constraint.end = extentTokens.size() - 1;
    constraint.state = Pattern.compile(".*");
    List<ParserConstraint> constraints = Collections.singletonList(constraint);
    Tree tree = parse(extentTokens, constraints);
    if (Props.KBP_VERBOSE) logger.debug("no exact match found. Local parse:\n" + tree.pennString());
    CoreMapUtils.convertToCoreLabels(tree);
    tree.indexSpans(ent.getExtentTokenStart() - ADDED_WORDS);  // remember it has ADDED_WORDS extra words at the beginning
    Tree subtree = findPartialSpan(tree, ent.getExtentTokenStart());
    Tree extentHead = safeHead(subtree);
    if (Props.KBP_VERBOSE) logger.debug("head is: " + extentHead);
    assert(extentHead != null);
    // extentHead is a child in the local extent parse tree. we need to find the corresponding node in the main tree
    // Because we deleted dashes, it's index will be >= the index in the extent parse tree
    CoreLabel l = (CoreLabel) extentHead.label();
    // Tree realHead = findTreeWithSpan(root, l.get(BeginIndexAnnotation.class), l.get(EndIndexAnnotation.class));
    Tree realHead = funkyFindLeafWithApproximateSpan(root, l.value(), l.get(BeginIndexAnnotation.class), approximateness);
    if(Props.KBP_VERBOSE && realHead != null) logger.debug("chosen head: " + realHead);
    return realHead;
  }

  /**
   * Finds the tree with the given token span.
   * The tree must have CoreLabel labels and Tree.indexSpans must be called before this method.
   *
   * @param tree The tree to search in
   * @param start The beginning index
   * @param end The end index
   * @return A child of tree if match; otherwise null
   */
  private static Tree findTreeWithSpan(Tree tree, int start, int end) {
    CoreLabel l = (CoreLabel) tree.label();
    if (l != null && l.has(BeginIndexAnnotation.class) && l.has(EndIndexAnnotation.class)) {
      int myStart = l.get(BeginIndexAnnotation.class);
      int myEnd = l.get(EndIndexAnnotation.class);
      if (start == myStart && end == myEnd){
        // found perfect match
        return tree;
      } else if (end < myStart) {
        return null;
      } else if (start >= myEnd) {
        return null;
      }
    }

    // otherwise, check inside children - a match is possible
    for (Tree kid : tree.children()) {
      if (kid == null) continue;
      Tree ret = findTreeWithSpan(kid, start, end);
      // found matching child
      if (ret != null) return ret;
    }

    // no match
    return null;
  }

  private static CoreLabel initCoreLabel(String token) {
    CoreLabel label = new CoreLabel();
    label.setWord(token);
    label.setValue(token);
    label.set(TextAnnotation.class, token);
    return label;
  }

  private static Tree safeHead(Tree top) {
    Tree head;
    synchronized (headFinder) {
      head = top.headTerminal(headFinder);
    }
    if (head != null) return head;
    // if no head found return the right-most leaf
    List<Tree> leaves = top.getLeaves();
    if(leaves.size() > 0) return leaves.get(leaves.size() - 1);
    // fallback: return top
    return top;
  }

  private Annotator getParser() {
    if(parserProcessorOrNull == null){
      if (StanfordCoreNLP.getExistingAnnotator("parse") == null) {
        logger.log("no StanfordCoreNLP has ever been created; manually creating parse annotator");
        parserProcessorOrNull = new ParserAnnotator("parse", props);
      } else {
        parserProcessorOrNull = StanfordCoreNLP.getExistingAnnotator("parse");
      }
    }
    assert(parserProcessorOrNull != null);
    return parserProcessorOrNull;
  }

  protected Tree parse(List<CoreLabel> tokens,
                              List<ParserConstraint> constraints) {
    CoreMap sent = new Annotation("");
    sent.set(TokensAnnotation.class, tokens);
    sent.set(ParserAnnotations.ConstraintAnnotation.class, constraints);
    Annotation doc = new Annotation("");
    List<CoreMap> sents = new ArrayList<CoreMap>();
    sents.add(sent);
    doc.set(SentencesAnnotation.class, sents);
    getParser().annotate(doc);
    sents = doc.get(SentencesAnnotation.class);
    return sents.get(0).get(TreeCoreAnnotations.TreeAnnotation.class);
  }

  private static Tree findPartialSpan(Tree current, int start) {
    CoreLabel label = (CoreLabel) current.label();
    int startIndex = label.get(BeginIndexAnnotation.class);
    if (startIndex == start) {
      if (Props.KBP_VERBOSE) logger.debug("findPartialSpan: Returning " + current);
      return current;
    }
    for (Tree kid : current.children()) {
      CoreLabel kidLabel = (CoreLabel) kid.label();
      int kidStart = kidLabel.get(BeginIndexAnnotation.class);
      int kidEnd = kidLabel.get(EndIndexAnnotation.class);
      // System.err.println("findPartialSpan: Examining " + kidLabel.value() + " from " + kidStart + " to " + kidEnd);
      if (kidStart <= start && kidEnd > start) {
        return findPartialSpan(kid, start);
      }
    }
    throw new RuntimeException("Shouldn't happen: " + start + " " + current);
  }

  private static Tree funkyFindLeafWithApproximateSpan(Tree root, String token, int index, int approximateness) {
    if (Props.KBP_VERBOSE) logger.debug("looking for " + token + " at pos " + index + " plus upto " + approximateness + " in tree: " + root.pennString());
    List<Tree> leaves = root.getLeaves();
    for (Tree leaf : leaves) {
      CoreLabel label = CoreLabel.class.cast(leaf.label());
      int ind = label.get(BeginIndexAnnotation.class);
      // System.err.println("Token #" + ind + ": " + leaf.value());
      if (token.equals(leaf.value()) && ind >= index && ind <= index + approximateness) {
        return leaf;
      }
    }
    // this shouldn't happen
    // but it does happen (VERY RARELY) on some weird web text that includes SGML tags with spaces
    // TODO: does this mean that somehow tokenization is different for the parser? check this by throwing an Exception in KBP
    logger.err("failed to find head token when looking for " + token + " at pos " + index + " plus upto " + approximateness + " in tree: " + root.pennString());
    return null;
  }

  private static void markTriggerWords(List<CoreLabel> tokens) {
    for (TriggerSeq seq : triggers) {
      for (int start = 0; start < tokens.size() - seq.tokens.length;) {
        boolean matches = true;
        for (int i = 0; i < seq.tokens.length; i++) {
          if (!tokens.get(start + i).word().equalsIgnoreCase(seq.tokens[i])) matches = false;
        }
        if (matches) {
          tokens.get(start).set(TriggerAnnotation.class, "B-" + seq.label);
          for (int i = 1; i < seq.tokens.length; i++)
            tokens.get(start + i).set(TriggerAnnotation.class, "I-" + seq.label);

          start += seq.tokens.length;
        } else {
          start++;
        }
      }
    }
  }

  private static class TriggerSeq implements Comparable<TriggerSeq> {
    String label;
    String[] tokens;

    public String toString() {
      StringBuilder os = new StringBuilder();
      os.append(label).append(":");
      for (String t : tokens)
        os.append(" ").append(t);
      return os.toString();
    }

    public TriggerSeq(String l, String[] ts) {
      label = l;
      tokens = ts;
    }

    /** Descending order of lengths */
    public int compareTo(TriggerSeq other) {
      return other.tokens.length - this.tokens.length;
    }
  }
}
