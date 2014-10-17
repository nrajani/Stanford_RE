package edu.stanford.nlp.kbp.slotfilling.ir;


import edu.stanford.nlp.dcoref.CorefChain;
import edu.stanford.nlp.dcoref.CorefCoreAnnotations;
import edu.stanford.nlp.dcoref.Dictionaries;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.time.TimeAnnotations;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.*;
import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * Annotates a document with information that depends on the entity.
 *
 * <h3>Catalogue of hacks (add to me as more are added!):</h3>
 * <ul>
 *     <li>Plural pronouns are disabled (e.g., "they")</li>
 *     <li>First person pronouns are [usually] disabled (e.g., "I", "me")</li>
 *     <li>Nested mentions are disabled (only the most specific mention is taken).</li>
 *       This is globally enforced over all chains, so things like "University of California" and "California" are not allowed
 *       to co-occur in the same span.
 *     <li>(enableable with option) Approximate names are matched to the query entity iff no other person in the article could be assigned that partial name</li>
 *     <li>Representative mentions are found by:
 *       <ol>
 *         <li>checking if the given mention matches the candidate NER tag exactly</li>
 *         <li>checking if any of the other mentions match the candidate NER tag exactly (longest to shortest)</li>
 *         <li>finding a mention with the longest sub-part matching the NER tag (ties broken by appearance in document)</li>
 *         <li>finding a mention with the longest sub-part matching a noun (ties broken by appearance in document)</li>
 *       </ol>
 *     <li>We look for verbatim matches with the entity name in the document. This is actually somewhat relaxed, to include
 *       approximate matches and acronyms.</li>
 * </ul>
 *
 * <h3>Annotations Provided:</h3>
 * <ul>
 *   <li>{@link KBPAnnotations.AllAntecedentsAnnotation} attached to the sentence. These are all the antecedents in the sentence,
 *       and are guaranteed to contain the exact entity name if the entity occurs in the sentence. </li>
 *   <li>{@link edu.stanford.nlp.kbp.slotfilling.common.KBPAnnotations.AlternateNamesAnnotation}   attached to a sentence, identifying alternate names for
 *        entities in AllAntecedentsAnnotation</li>
 *   <li>{@link edu.stanford.nlp.kbp.slotfilling.common.KBPAnnotations.CanonicalEntitySpanAnnotation}   attached to a document,
 *        identifying the sentence and token span of the canonical mention for the entity</li>
 *   <li>{@link edu.stanford.nlp.kbp.slotfilling.common.KBPAnnotations.CanonicalSlotValueSpanAnnotation}   attached to a document,
 *        identifying the sentence and token span of the canonical mention for the slot value</li>
 *   <li>{@link KBPAnnotations.IsCoreferentAnnotation}   attached to the sentence</li>
 *   <li>{@link KBPAnnotations.IsCoreferentAnnotation}   attached to entity and slot value tokens</li>
 *   <li>{@link edu.stanford.nlp.ling.CoreAnnotations.AntecedentAnnotation}   attached to any token with an antecedent (especially the entity and slot value)</li>
 * </ul>
 *
 * @author Gabor Angeli
 */
public class PostIRAnnotator implements Annotator {
  private static final Redwood.RedwoodChannels logger = Redwood.channels("PostIR");

  public final String entityName;
  public final Maybe<String> entityType;
  public final String[] entityTokens;
  public final Maybe<String> slotValue;
  public final Maybe<String> slotValueType;
  public final Maybe<String[]> slotValueTokens;
  public final boolean doCoref;

  private static final Dictionaries dictionaries = new Dictionaries();
  private static final Set<String> commonNames = new HashSet<String>() {
    {
      BufferedReader input = null;
      try {
        input = IOUtils.readerFromString(Props.INDEX_POSTIRANNOTATOR_COMMONNAMES.getPath());
        for (String line;  (line = input.readLine()) != null; ) {
          this.add(line);  // add to HashSet
        }
      } catch (IOException e) {
        logger.warn("Could not load common names (exception below)");
        logger.warn(e);  // soft exception
      } finally {
        IOUtils.closeIgnoringExceptions(input);
      }
    }
    private static final long serialVersionUID = 871045641050423018L;
  };
  @SuppressWarnings("UnusedDeclaration")
  public static final HashMap<String, String> knownAliases = new HashMap<String, String>();

  private static class CorpusStats {
    public final Map<String, Set<String>> peopleByFirstName = new HashMap<String, Set<String>>();
    public final Map<String, Set<String>> peopleByLastName = new HashMap<String, Set<String>>();
    public final Map<String, Collection<String>> entitiesByType = new HashMap<String, Collection<String>>();

    @SuppressWarnings("StatementWithEmptyBody")
    public CorpusStats(Annotation corpus, String entityName) {
      startTrack("Creating stats for " + entityName);
      // -- Loop Over Sentences
      for (CoreMap sentence : corpus.get(SentencesAnnotation.class)) {
        // -- Sentence State
        // State for NER
        String lastNER = "O";
        String lastEntity = null;
        // -- Loop Over Tokens
        for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
          // Logic for NER
          assert lastNER.equals("O") || lastEntity != null;
          if ((lastNER.equals("O") && !token.ner().equals("O")) || (!lastNER.equals("O") && lastNER.equals(token.ner()))) {
            // We've gone O->NER or NER1->NER1
            if (lastEntity == null) {
              lastEntity = token.originalText();
            } else {
              lastEntity = lastEntity + " " + token.originalText();
            }
          } else if ( (!lastNER.equals("O") && token.ner().equals("O")) || (!lastNER.equals("O") && !lastNER.equals(token.ner()))) {
            // We've gone NER->O or NER1->NER2
            assert lastEntity != null;
            if (!entitiesByType.containsKey(lastNER)) { entitiesByType.put(lastNER, new HashSet<String>()); }
            entitiesByType.get(lastNER).add(lastEntity);
            if (token.ner().equals("O")) {
              lastEntity = null;
            } else {
              lastEntity = token.originalText();
            }
          } else if (lastNER.equals("O") && token.ner().equals("O")) {
            // do nothing
          } else {
            throw new IllegalStateException("Unknown transition: " + lastNER + " -> " + token.ner());
          }
          lastNER = token.ner();
        }
      }

      // -- Post Process Info
      // Register names
      if (entitiesByType.containsKey(NERTag.PERSON.name)) {
        for (String person : entitiesByType.get(NERTag.PERSON.name)) {
          String[] nameParts = person.split("\\s+");
          if (nameParts.length > 1) {
            if (!peopleByFirstName.containsKey(nameParts[0])) { peopleByFirstName.put(nameParts[0], new HashSet<String>()); }
            peopleByFirstName.get(nameParts[0]).add(person);
            if (!peopleByLastName.containsKey(nameParts[nameParts.length - 1])) { peopleByLastName.put(nameParts[nameParts.length - 1], new HashSet<String>()); }
            peopleByLastName.get(nameParts[nameParts.length - 1]).add(person);
          }
        }
      }
      endTrack("Creating stats for " + entityName);
    }
  }

  public PostIRAnnotator(String entityName, Maybe<String> entityType, Maybe<String> slotValue, Maybe<String> slotValueType,
                         boolean doCoref) {
    this.entityName = entityName;
    this.entityTokens = entityName.split("\\s+");
    this.slotValue = slotValue;
    this.slotValueTokens = slotValue.isDefined() ? Maybe.Just(slotValue.get().split("\\s+")) : Maybe.<String[]>Nothing();
    this.entityType = entityType;
    this.slotValueType = slotValueType;
    this.doCoref = doCoref;
  }

  /** Create a PostIRAnnotator from an entity */
  public PostIRAnnotator(String entityName) {
    this(entityName, Maybe.<String>Nothing(), Maybe.<String>Nothing(), Maybe.<String>Nothing(), true);
  }

  @Override
  public void annotate(final Annotation corpus) {
    // Compute stats
    Lazy<CorpusStats> entityStats  = new Lazy<CorpusStats>(){
      @Override
      protected CorpusStats compute() {
        return new CorpusStats(corpus, entityName);
      }
    };
    Lazy<Maybe<CorpusStats>> slotValueStats = new Lazy<Maybe<CorpusStats>>() {
      @Override
      protected Maybe<CorpusStats> compute() {
        return  slotValue.isDefined() ? Maybe.Just(new CorpusStats(corpus, slotValue.get())) : Maybe.<CorpusStats>Nothing();
      }
    };
    // Annotate coref chains
    try {
      if (doCoref) { annotateCoref(corpus, entityStats, slotValueStats); }
    } catch (Exception e) {
      logger.warn("could not annotate Coref; continuing anyways");
    }
    // Annotate literal coref
    try {
      annotateLiteralCoref(corpus, entityName, entityType, entityTokens);
      if (slotValue.isDefined() && slotValueTokens.isDefined()) {
        annotateLiteralCoref(corpus, slotValue.get(), slotValueType, slotValueTokens.get());
      }
    } catch (RuntimeException e) {
      logger.err(e);
    }
    // Annotate "other" coref (e.g., times)
    annotateTimex(corpus);
  }

  @SuppressWarnings("unchecked")
  @Override
  public Set<Requirement> requirementsSatisfied() {
    return new HashSet<Requirement>(Arrays.asList(new Requirement[]{new Requirement("postir.coref")}));
  }

  @Override
  public Set<Requirement> requires() {
    return new HashSet<Requirement>();
  }

  /**
   * Find a representative mention for every coref chain, and set
   * the AntecedentAnnotation for every coreferent token to its representative mention.
   *
   * In addition, set the utility annotations:
   *   - AllAntecedentsAnnotation: set for each sentence, listing off all the antecedents in the sentence
   *   - IsCoreferentAnnotation: set for each sentence with an entity mention, and set to
   *                             false if the entity appears verbatim in the sentence, and true
   *                             if the entity appears as a coreferent term.
   *                             The same is done for tokens.
   *
   * Important Notes:
   *   - Only tokens in mentions with a coherent NER tag are annotated, to cut down on spurious chains.
   *
   * @param corpus The document to annotate, as it comes fresh out of lucene.
   * @param entityStats Various statistics on the document which have been precomputed, based on the entity.
   * @param slotValueStats Various statistics on the document which have been precomputed, based on the slot value (if provided).
   */
  private void annotateCoref(Annotation corpus, Lazy<CorpusStats> entityStats, Lazy<Maybe<CorpusStats>> slotValueStats) {
    // Some preparation
    // NOTE: AllAntecedentsAnnotation is used as the check in annotateSentenceFeatures for whether PostIR has been run.
    //       thus, be careful before deleting or changing this block of code.
    for (CoreMap sentence : corpus.get(SentencesAnnotation.class)) {
      sentence.set(KBPAnnotations.AllAntecedentsAnnotation.class, new HashSet<String>());
    }
    if (!corpus.containsKey(CorefCoreAnnotations.CorefChainAnnotation.class)) {
      err(RED, "Annotation doesn't have coref chain annotation!");
      return;
    }

    // Get cluster positions from chains ...

    Collection<Pair<List<CorefChain.CorefMention>, CorefChain.CorefMention>> cleanedChains = cleanCorefChains(corpus.get(CorefCoreAnnotations.CorefChainAnnotation.class));
    Counter<String> nerVotes = new ClassicCounter<String>();
    for (Pair<List<CorefChain.CorefMention>,CorefChain.CorefMention> mentionsAndRepresentantiveMention : cleanedChains) {
      // Setup
      nerVotes.clear();  // the likely NER of the cluster, chosen by mentions voting on their NER type
      List<CorefChain.CorefMention> mentionsByLength = new LinkedList<CorefChain.CorefMention>(mentionsAndRepresentantiveMention.first);
      Collections.sort(mentionsByLength, new Comparator<CorefChain.CorefMention>() {
        @Override
        public int compare(CorefChain.CorefMention o1, CorefChain.CorefMention o2) {
          int lengthDifference = (o2.endIndex - o2.startIndex) - (o1.endIndex - o1.startIndex);
          return lengthDifference == 0 ? (o1.sentNum * 1000 + o1.startIndex) - (o2.sentNum * 1000 + o2.startIndex) : lengthDifference;
        }
      });
      CorefChain.CorefMention representativeMention = null;

      // Try to assign the representative mention
      // ...by looking for an exact match with the entity
      for (CorefChain.CorefMention mention : mentionsAndRepresentantiveMention.first) {
        CoreLabel token = corpus.get(SentencesAnnotation.class).get(mention.sentNum - 1).get(TokensAnnotation.class).get(mention.headIndex - 1);
        nerVotes.incrementCount(token.ner());
        // Check if this is an exact match with the entity
        if (representativeMention == null && (mention.mentionSpan.equalsIgnoreCase(entityName) ||
            (slotValue.isDefined() && mention.mentionSpan.equalsIgnoreCase(slotValue.get())))) {
          representativeMention = mention;
        }
      }
      nerVotes.remove("O");
      String ner = Counters.argmax(nerVotes);

      //...by looking for a mention which contains the entity name
      String entityToLowerCase = entityName.toLowerCase();
      String slotValueToLowerCase = slotValue.getOrElse("").toLowerCase();
      if (representativeMention == null) {
        for (CorefChain.CorefMention candidate : mentionsAndRepresentantiveMention.first) {
          String mentionToLowerCase = candidate.mentionSpan.toLowerCase();
          if (representativeMention == null && mentionToLowerCase.contains(entityToLowerCase) &&
              (candidate.endIndex - candidate.startIndex) <= entityTokens.length + 2) {
            representativeMention = candidate;
            break;
          } if (representativeMention == null && slotValue.isDefined() &&
                mentionToLowerCase.contains(slotValueToLowerCase) &&
                (candidate.endIndex - candidate.startIndex) <= slotValueTokens.get().length + 2) {
            representativeMention = candidate;
            break;
          }
        }
      }
      //...by looking for an NER compatible mention (preference to representativeMention)
      if (representativeMention == null && ner != null) {
        CorefChain.CorefMention representativeCandidate = mentionsAndRepresentantiveMention.second;
        if (representativeMention == null && representativeCandidate != null && mentionMatchesNER(representativeCandidate, ner, corpus)) {
          // Case: the representative mention is NER compatible
          representativeMention = representativeCandidate;
        } else {
          // Case: search for a mention (longest to shortest) that's NER compatible
          for (CorefChain.CorefMention candidate : mentionsByLength) {
            if (representativeMention == null && mentionMatchesNER(candidate, ner, corpus)) {
              representativeMention = candidate;
              break;
            }
          }
        }
      }
      // ...by hacks
      String antecedent = representativeMention == null ? null : representativeMention.mentionSpan;
      Pair<Integer, Span> antecedentSpanInfo = representativeMention == null ? null : Pair.makePair(representativeMention.sentNum - 1, new Span(representativeMention.startIndex - 1, representativeMention.endIndex - 1));
      if (antecedent == null) {
        // Trim out a matching NER phrase, from the head word, preferring (a) longer phrases and (b) earlier mentions
        for (CorefChain.CorefMention candidate : mentionsAndRepresentantiveMention.first) {
          List<CoreLabel> tokens = corpus.get(SentencesAnnotation.class).get(candidate.sentNum - 1).get(TokensAnnotation.class);
          if (tokens.get(candidate.headIndex - 1).ner().equals(ner)) {
            int start = candidate.headIndex - 1; int end = candidate.headIndex;
            while (start > 0 && tokens.get(start - 1).ner().equals(ner)) { start -= 1; }
            while (end < tokens.size() - 1 && tokens.get(end).ner().equals(ner)) { end += 1; }
            String candidateAntecedent = CoreMapUtils.phraseToOriginalString(tokens.subList(start, end));
            if (antecedent == null || candidateAntecedent.split("\\s+").length > antecedent.split("\\s+").length) {
              antecedent = candidateAntecedent;
              antecedentSpanInfo = Pair.makePair(candidate.sentNum - 1, new Span(start, end));
            }
          }
        }
      }
      if (antecedent == null) {
        // Find a head word which is a noun, preferring (a) proper nouns, and (b) earlier mentions
        for (String pos : new String[]{"nnp.*", "nns.*", "n.*", ".*"}) {
          for (CorefChain.CorefMention candidate : mentionsAndRepresentantiveMention.first) {
            List<CoreLabel> tokens = corpus.get(SentencesAnnotation.class).get(candidate.sentNum - 1).get(TokensAnnotation.class);
            CoreLabel head = tokens.get(candidate.headIndex - 1);
            if (head.tag().toLowerCase().matches(pos)) {
              antecedent = head.containsKey(OriginalTextAnnotation.class) ? head.originalText() : head.word();
              antecedentSpanInfo = Pair.makePair(candidate.sentNum - 1, new Span(candidate.headIndex - 1, candidate.headIndex));
            }
          }
        }
      }
      if (antecedent == null) { throw new IllegalStateException("Could not find antecedent for chain with: " + mentionsAndRepresentantiveMention.second); }
      // Clean up antecedent a bit
      if (antecedent.toLowerCase().contains(entityName.toLowerCase())) {
        antecedent = entityName;
      }
      if (slotValue.isDefined() && antecedent.toLowerCase().contains(slotValue.get().toLowerCase())) {
        antecedent = slotValue.get();
      }
      // Rewrite the antecedent to the query entity, if it forms an approximate name match
      if (Props.INDEX_POSTIRANNOTATOR_APPROXNAME && ner != null && ner.equals(NERTag.PERSON.name) &&
          !commonNames.contains(antecedent)) {
        if (partialNameMatchesEntity(entityStats.get(), antecedent)) {
          antecedent = entityName;
        }
        if (slotValue.isDefined() && slotValueStats.get().isDefined() && partialNameMatchesEntity(slotValueStats.get().get(), antecedent)) {
          antecedent = slotValue.get();

        }
      }

      // Set the AntecedentAnnotation and AllAntecedentsAnnotation
      if (ner != null) {  // Only annotate entities with a valid NER tag
        for (CorefChain.CorefMention mention : mentionsAndRepresentantiveMention.first) {
          // AllAntecedents
          CoreMap sentence = corpus.get(SentencesAnnotation.class).get(mention.sentNum - 1);
          sentence.get(KBPAnnotations.AllAntecedentsAnnotation.class).add(antecedent);
          // Antecedent
          List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
          for (int i = mention.startIndex; i < mention.endIndex; ++i) {
            CoreLabel token = tokens.get(i - 1);
            if ((antecedent.equals(entityName) || slotValue.equalsOrElse(antecedent, false)) &&
                !(token.tag().startsWith("NN") || token.tag().startsWith("PRP"))) { continue; }
            if (!token.containsKey(AntecedentAnnotation.class) || antecedent.equals(entityName) || slotValue.equalsOrElse(antecedent, false)) {
              // set per-token antecedent
              token.set(AntecedentAnnotation.class, antecedent);
              // set per-document canonical antecedent
              if (!corpus.containsKey(KBPAnnotations.CanonicalEntitySpanAnnotation.class) && antecedent.equals(entityName)) {
                corpus.set(KBPAnnotations.CanonicalEntitySpanAnnotation.class, antecedentSpanInfo);
              }
              if (!corpus.containsKey(KBPAnnotations.CanonicalSlotValueSpanAnnotation.class) && slotValue.equalsOrElse(antecedent, false)) {
                corpus.set(KBPAnnotations.CanonicalSlotValueSpanAnnotation.class, antecedentSpanInfo);
              }
            }
            // set token is-coreferent flag
            token.set(KBPAnnotations.IsCoreferentAnnotation.class, !mention.mentionSpan.equals(antecedent));
          }
        }
      }
    }

    // Set IsCoreferentAnnotation
    for (CoreMap sentence : corpus.get(SentencesAnnotation.class)) {
      if (sentence.get(KBPAnnotations.AllAntecedentsAnnotation.class).contains(entityName)) {
        boolean isCoref = !sentence.get(TextAnnotation.class).contains(entityName) &&
            !(sentence.containsKey(OriginalTextAnnotation.class) && sentence.get(OriginalTextAnnotation.class).contains(entityName));
        logger.debug("marking sentence [" + (isCoref ? "coref" : "direct") + "]: " + CoreMapUtils.sentenceToMinimalString(sentence));
        sentence.set(KBPAnnotations.IsCoreferentAnnotation.class, isCoref);
      }
    }
  }

  /**
   * Annotate all mentions of the entity name as coreferent with the entity.
   * This is to get around cases where the entity is mentioned, but is not recognized as a mention by
   * the coref system.
   * @param ann The document to annotate.
   * @param toMatch The String gloss of the item to match
   * @param toMatchNER The NER tag of the item to match, if known
   * @param toMatchTokens The tokenized gloss of the item to match,
   */
  public void annotateLiteralCoref(Annotation ann, String toMatch, Maybe<String> toMatchNER, String[] toMatchTokens) {
    for (int sentI = 0; sentI < ann.get(SentencesAnnotation.class).size(); ++sentI) {
      CoreMap sentence = ann.get(SentencesAnnotation.class).get(sentI);
      // Get sentence-level variables
      if (!sentence.containsKey(KBPAnnotations.AlternateNamesAnnotation.class)) { sentence.set(KBPAnnotations.AlternateNamesAnnotation.class, new HashMap<String, Set<Span>>()); }
      Map<String, Set<Span>> alternateNames = sentence.get(KBPAnnotations.AlternateNamesAnnotation.class);
      if (!sentence.containsKey(KBPAnnotations.AllAntecedentsAnnotation.class)) { sentence.set(KBPAnnotations.AllAntecedentsAnnotation.class, new HashSet<String>()); }
      alternateNames.clear();
      Set<String> antecedents = sentence.get(KBPAnnotations.AllAntecedentsAnnotation.class);
      List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
      int entityIndex = 0;

      // Iterate over tokens
      for (int i = 0; i < tokens.size(); ++i) {
        assert i >= 0;
        assert entityIndex <= i;
        // Logic for string match
        CoreLabel token = tokens.get(i);
        if (approximateMatch(toMatchTokens[entityIndex], token.originalText()) ||
            approximateMatch(toMatchTokens[entityIndex], token.word())) {
          entityIndex += 1;
        } else {
          entityIndex = 0;
        }
        if (entityIndex >= toMatchTokens.length) {
          // Case: found the entity
          Span literalEntitySpan = new Span(i + 1 - entityIndex, i + 1);
          // Set the antecedent annotation
          for (int k : literalEntitySpan) {
            tokens.get(k).set(AntecedentAnnotation.class, toMatch);
            tokens.get(k).set(KBPAnnotations.IsCoreferentAnnotation.class, false);
          }
          // Add the entity to all antecedents
          antecedents.add(toMatch);
          sentence.set(KBPAnnotations.IsCoreferentAnnotation.class, false);
          // Set canonical mention, if appropriate
          if (toMatch.equals(entityName) && !ann.containsKey(KBPAnnotations.CanonicalEntitySpanAnnotation.class)) {
            ann.set(KBPAnnotations.CanonicalEntitySpanAnnotation.class, Pair.makePair(sentI, literalEntitySpan));
          }
          if (slotValue.equalsOrElse(toMatch, false) && !ann.containsKey(KBPAnnotations.CanonicalSlotValueSpanAnnotation.class)) {
            ann.set(KBPAnnotations.CanonicalSlotValueSpanAnnotation.class, Pair.makePair(sentI, literalEntitySpan));
          }
          // Check for alternate names
          if (!toMatchNER.isDefined() || toMatchNER.get().equals(NERTag.ORGANIZATION.name)) { // Only for ORGs
            for (Span alternateNameSpan : tryExpandNamedEntitySpanFrom(tokens, literalEntitySpan)) {
              if (!alternateNameSpan.equals(literalEntitySpan)) {
                if (!alternateNames.containsKey(toMatch)) { alternateNames.put(toMatch, new HashSet<Span>()); }
                alternateNames.get(toMatch).add(alternateNameSpan);
              }
            }
          }
          // Reset entity Index
          entityIndex = 0;
        }

        // Logic for acronym match
        // The entity should be sufficiently long, it should match as an acronym, and it should not be clustered with anything yet
        String text = token.containsKey(OriginalTextAnnotation.class) ? token.originalText() : token.word();
        if (toMatchTokens.length >= 3 && text.length() >= 3) {
          boolean isAcronym = true;
          int indexInText = 0;
          for (String toMatchToken : toMatchTokens) {
            if (toMatchToken.length() == 0) { isAcronym = false; break; }    // Token is empty (for some strange reason)
            if (indexInText >= text.length()) { isAcronym = false; break; }  // candidate acronym is too short
            if (Character.toUpperCase(toMatchToken.charAt(0)) == text.charAt(indexInText)) { indexInText += 1; continue; }  // we match the word
            if (dictionaries.stopWords.contains(toMatchToken.toLowerCase())) {
              if (toMatchToken.charAt(0) == text.charAt(indexInText)) { indexInText += 1; continue; }  // we match lowercase stop word (e.g., BoA)
              else { continue; } // Ignore stop words
            }
            isAcronym = false; break;  // default: no match
          }
          if (isAcronym) {
            logger.debug("expanded acronym: " + text + " to " + StringUtils.join(toMatchTokens));
            // Set antecedent
            token.set(AntecedentAnnotation.class, toMatch);
            token.set(KBPAnnotations.IsCoreferentAnnotation.class, false);
            antecedents.add(toMatch);
            sentence.set(KBPAnnotations.IsCoreferentAnnotation.class, false);
            // Force some other properties
            token.setTag("NNP");
            for (String ner : toMatchNER) { if (ner != null && !ner.trim().equals("")) { token.setNER(ner); } }
            // Handle alternate names
            if (!alternateNames.containsKey(toMatch)) { alternateNames.put(toMatch, new HashSet<Span>()); }
            alternateNames.get(toMatch).add(new Span(i, i + 1));
            // Set canonical mention, if appropriate
            if (toMatch.equals(entityName) && !ann.containsKey(KBPAnnotations.CanonicalEntitySpanAnnotation.class)) {
              ann.set(KBPAnnotations.CanonicalEntitySpanAnnotation.class, Pair.makePair(sentI, new Span(i, i + 1)));
            }
            if (slotValue.equalsOrElse(toMatch, false) && !ann.containsKey(KBPAnnotations.CanonicalSlotValueSpanAnnotation.class)) {
              ann.set(KBPAnnotations.CanonicalSlotValueSpanAnnotation.class, Pair.makePair(sentI, new Span(i, i + 1)));
            }
          }
        }
      }
    }
  }

  /**
   * Try to find the largest enclosing consistent named entity span from the initial starting point.
   * Behavior is undefined (returns Maybe.Nothing) if the initial span has no, or has inconsistent Named
   * Entity labels.
   * Also, limit the named entities to PERSON or ORGANIZATION.
   *
   * @param tokens The sentence to expand within.
   * @param initialSpan The initial span to expand from.
   * @return The largest span (not necessarily larger than initial) which has the same named entity type as the
   *         initial span; or, Nothing if the initial span is inconsistent.
   */
  private static Maybe<Span> tryExpandNamedEntitySpanFrom(List<CoreLabel> tokens, Span initialSpan) {
    // Get NE Label we are expanding
    String neLabel = null;
    for (int i : initialSpan) {
      String check = tokens.get(i).ner();
      if (neLabel == null) { neLabel = check; }
      else if (!neLabel.equals(check)) { return Maybe.Nothing(); }
    }
    if (neLabel == null || (!neLabel.equals(NERTag.PERSON.name) && !neLabel.equals(NERTag.ORGANIZATION.name))) {
      return Maybe.Nothing();
    }
    // Expand
    int start = initialSpan.start();
    int end = initialSpan.end();
    while (start > 0 && tokens.get(start - 1).ner().equals(neLabel) &&
        tokens.get(start - 1).tag().startsWith("N") &&
        !"and".equals(tokens.get(start - 1).originalText()) && !"&".equals(tokens.get(start - 1).originalText()) && !"&amp;".equals(tokens.get(start - 1).originalText()))
      { start -= 1; }  // note: lowercase 'and'
    while (end < tokens.size() && tokens.get(end).ner().equals(neLabel) &&
        tokens.get(end).tag().startsWith("N") &&
        !"and".equals(tokens.get(end).originalText()) && !"&".equals(tokens.get(end).originalText()) && !"&amp;".equals(tokens.get(end).originalText()))
      { end += 1; }    // note: lowercase 'and'
    return Maybe.Just(new Span(start, end));
  }

  /**
   * An approximate name matching scheme, for if we can safely match a name with
   * the entity name.
   * @param stats The corpus statistics gathered
   * @param name The name we are matching
   * @return Return true if the name can be safely clustered with the query entity
   */
  private boolean partialNameMatchesEntity(CorpusStats stats, String name) {
    Collection<String> byFirstName = stats.peopleByFirstName.get(name);
    if ((byFirstName == null && entityName.startsWith(name)) ||
        (byFirstName != null && byFirstName.size() == 1 && byFirstName.contains(name))) {
      // case: same first name as entity, and no one else has that first name
      return true;
    }
    Collection<String> byLastName = stats.peopleByLastName.get(name);
    //noinspection RedundantIfStatement
    if ((byLastName == null && entityName.endsWith(name)) ||
        (byLastName != null && byLastName.size() == 1 && byLastName.contains(name))) {
      // case: same last name as entity, and no one else has that first name
      return true;
    }
    return false;
  }

  /**
   * Removes coref mentions (and, potentially entire chains) if the mention is something
   * that is very unlikely to be relevant for KBP.
   * For example:
   *   - First person pronouns (I, me, etc.)
   *
   * In addition, nested mentions are disallowed, taking only the most narrow mention(s) in the span.
   *
   * @param inputs The coref chains in the document
   * @return The mention chain in order of appearance, and the representative mention.
   */
  private static Collection<Pair<List<CorefChain.CorefMention>, CorefChain.CorefMention>> cleanCorefChains(Map<Integer, CorefChain> inputs) {
    // Variables that will come in useful
    int numSentences = -1;
    int numTokensInSentence = -1;

    // Prune invalid mentions
    Set<CorefChain.CorefMention> singletonsValid = new IdentityHashSet<CorefChain.CorefMention>();
    for (CorefChain chain : inputs.values()) {
      int numPersonalPronouns = 0;
      for (CorefChain.CorefMention mention : chain.getMentionsInTextualOrder()) {
        if (dictionaries.firstPersonPronouns.contains(mention.mentionSpan.toLowerCase())) { numPersonalPronouns += 1; }
      }
      for (CorefChain.CorefMention mention : chain.getMentionsInTextualOrder()) {
        numSentences = Math.max(numSentences, mention.sentNum + 1);
        numTokensInSentence = Math.max(numTokensInSentence, mention.endIndex + 1);
        // (add valid mentions)
        if (numPersonalPronouns >= 3 || !dictionaries.firstPersonPronouns.contains(mention.mentionSpan.toLowerCase())) {
          // Conditions:
          //   1. if this is no a personal pronoun -> add
          //   2. if this chain composes of enough personal pronouns -> add (likely an interview)
          singletonsValid.add(mention);
        }

      }
    }
    // Just in case?
    if (numSentences < 0) { numSentences = 1; }
    if (numTokensInSentence < 0) { numTokensInSentence = 1000; }
    // (add representative mentions)
    for (CorefChain chain : inputs.values()) { singletonsValid.add(chain.getRepresentativeMention()); }

    // Prune nested mentions, keeping shorter mentions when there's overlap
    boolean[][] mask = new boolean[numSentences][numTokensInSentence];
    // (sort mentions by size)
    List<CorefChain.CorefMention> mentionsByLength = new ArrayList<CorefChain.CorefMention>(singletonsValid);
    Collections.sort(mentionsByLength, new Comparator<CorefChain.CorefMention>() {
      @Override public int compare(CorefChain.CorefMention o1, CorefChain.CorefMention o2) {
        return (o1.endIndex - o1.startIndex) - (o2.endIndex - o2.startIndex);
      }
    });
    // (collect valid mentions)
    Set<CorefChain.CorefMention> nonOverlappingMentions = new IdentityHashSet<CorefChain.CorefMention>();
    for (CorefChain.CorefMention mention : mentionsByLength) {
      boolean overlaps = false;
      for (int i = mention.startIndex; i < mention.endIndex; ++i) {
        if (mask[mention.sentNum][i]) { overlaps = true; break; }
      }
      if (!overlaps) {
        for (int i = mention.startIndex; i < mention.endIndex; ++i) { mask[mention.sentNum][i] = true; }
        nonOverlappingMentions.add(mention);
      }
    }

    // Assemble return structure
    ArrayList<Pair<List<CorefChain.CorefMention>,CorefChain.CorefMention>> rtn = new ArrayList<Pair<List<CorefChain.CorefMention>,CorefChain.CorefMention>>();
    for (CorefChain chain : inputs.values()) {
      // Add the valid mentions
      List<CorefChain.CorefMention> mentions = new ArrayList<CorefChain.CorefMention>();
      for (CorefChain.CorefMention mention : chain.getMentionsInTextualOrder()) {
        if (nonOverlappingMentions.contains(mention)) { mentions.add(mention); }
      }
      if (mentions.size() > 0) {
        rtn.add(Pair.makePair(mentions, nonOverlappingMentions.contains(chain.getRepresentativeMention()) ? chain.getRepresentativeMention() : null));
      }
    }
    return rtn;
  }

  /**
   * Ensure that the given candidate mention matches the target NER in every token.
   * This is to prevent overly-greedy mentions being claimed as the representative mention.
   * @param candidate The candidate mention
   * @param ner The NER we are constraining it to match
   * @param corpus The document
   * @return True if every token in the candidate mention matches the NER type specified
   */
  private static boolean mentionMatchesNER(CorefChain.CorefMention candidate, String ner, Annotation corpus) {
    List<CoreLabel> tokens = corpus.get(SentencesAnnotation.class).get(candidate.sentNum - 1).get(TokensAnnotation.class);
    for (int i = candidate.startIndex; i < candidate.endIndex; ++i) {
      if (!tokens.get(i - 1).ner().equals(ner)) { return false; }
    }
    return true;
  }

  /**
   * Some awful hacks for matching approximate names.
   * For example: "Nuclear Supplier Group" with "Nuclear Suppliers Group" or "ABC Co." with "ABC Corporation"
   * @param token The token we are checking for a match.
   * @param reference The reference we are checking against.
   * @return True if the token matches the reference, according to the approximate criteria.
   */
  @SuppressWarnings("RedundantIfStatement")
  private static boolean approximateMatch(String token, String reference) {
    if (token == null || reference == null) { return false; }
    if (token.equalsIgnoreCase(reference)) { return true; }
    if (reference.toUpperCase().startsWith(token.toUpperCase()) && token.endsWith(".")) { return true; }
    if ((token + "s").equals(reference)) { return true; }
    if ((token + "es").equals(reference)) { return true; }
    if ( (token + ".").equals(reference) || (reference + ".").equals(token)) { return true; }
    return false;
  }


  /**
   * Annotate normalized Timex values as the canonical Antecedent of the temporal phrase
   * @param ann The document to annotate
   */
  public static void annotateTimex(Annotation ann) {
    for (CoreMap sentence : ann.get(SentencesAnnotation.class)) {
      for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
        if (token.containsKey(TimeAnnotations.TimexAnnotation.class)) {
          String timexValue = token.get(TimeAnnotations.TimexAnnotation.class).value();
          if (timexValue != null) { token.set(AntecedentAnnotation.class, timexValue); }
        }
      }
    }
  }

  /**
   * Uses PostIR annotation to identify relevant sentences.
   * @param document - document you want to annotate.
   * @return - returns sentence indices in the document that are good.
   */
  public Set<Integer> findRelevantSentences(Annotation document) {
    final List<CoreMap> sentences = document.get(SentencesAnnotation.class);
    Set<Integer> goodSentences = new HashSet<Integer>();

    // Annotate document
    annotate(document);
    // Select relevant sentences
    for (int i = 0; i < sentences.size(); i++) {
      CoreMap sentence = sentences.get(i);
      if (!sentence.containsKey(KBPAnnotations.AllAntecedentsAnnotation.class)) { continue; }  // No antecedents
      if (sentence.get(TokensAnnotation.class).size() > Props.TEST_SENTENCES_MAX_TOKENS) { continue; }  // Way too long
      Set<String> antecedents = sentence.get(KBPAnnotations.AllAntecedentsAnnotation.class);
      if (antecedents != null && antecedents.contains(entityName) &&  // must contain entity name
          (!slotValue.isDefined() || antecedents.contains(slotValue.get()) || sentence.get(TextAnnotation.class).contains(slotValue.get())) &&  // must contain slot value (if defined)
          (Props.INDEX_COREF_DO || (sentence.containsKey(KBPAnnotations.IsCoreferentAnnotation.class) &&!sentence.get(KBPAnnotations.IsCoreferentAnnotation.class)))) {  // maybe filter coreferent sentences
        goodSentences.add(i);
      }
    }
    return goodSentences;
  }
}
