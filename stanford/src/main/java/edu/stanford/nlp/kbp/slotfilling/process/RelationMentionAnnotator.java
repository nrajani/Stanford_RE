package edu.stanford.nlp.kbp.slotfilling.process;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.ie.machinereading.structure.*;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations.AllRelationMentionsAnnotation;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations.EntityMentionsAnnotation;
import edu.stanford.nlp.ie.machinereading.structure.MachineReadingAnnotations.RelationMentionsAnnotation;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.common.KBPAnnotations.SlotMentionsAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.AntecedentAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;
import edu.stanford.nlp.util.logging.Redwood;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * Annotates sentences with Relation Mention annotations
 *
 * @author Gabor Angeli
 */
public class RelationMentionAnnotator implements Annotator {
  private static Redwood.RedwoodChannels logger = Redwood.channels("RelAnn");
  private static AtomicInteger relationMentionCount = new AtomicInteger(0);

  public final KBPEntity entity;
  public final List<KBPSlotFill> fillsForEntity;
  public final KBPProcess.AnnotateMode annotateMode;

  public RelationMentionAnnotator(KBPEntity entity, List<KBPSlotFill> fillsForEntity, KBPProcess.AnnotateMode annotateMode) {
    this.entity = entity;
    this.fillsForEntity = fillsForEntity;
    this.annotateMode = annotateMode;
  }


  @Override
  public void annotate(Annotation annotation) {
    for (CoreMap sentence : annotation.get(SentencesAnnotation.class)) {
      // Create Relation Mention Annotations
      Triple<List<RelationMention>, List<RelationMention>, List<EntityMention>> relationsAndNewSlotMentions = 
          getRelationAndNewSlotMentions(
              sentence.get(EntityMentionsAnnotation.class),
              sentence.get(SlotMentionsAnnotation.class),
              sentence, this.fillsForEntity);
      List<RelationMention> relationMentions = relationsAndNewSlotMentions.first;
      List<RelationMention> allRelationMentions = relationsAndNewSlotMentions.second;
      List<EntityMention> slotMentions = relationsAndNewSlotMentions.third;
      
      sentence.set(RelationMentionsAnnotation.class, new ArrayList<RelationMention>(relationMentions));
      if(annotateMode == KBPProcess.AnnotateMode.ALL_PAIRS) {
        sentence.set(AllRelationMentionsAnnotation.class, allRelationMentions);
      }
      sentence.set(SlotMentionsAnnotation.class, new ArrayList<EntityMention>(slotMentions));  // should re-set slot mentions
    }
  }

  @Override
  public Set<Requirement> requirementsSatisfied() {
    return new HashSet<Requirement>(Arrays.asList(new Requirement[]{new Requirement("postir.relationmentions")}));
  }

  @Override
  public Set<Requirement> requires() {
    return new HashSet<Requirement>(Arrays.asList(new Requirement[]{new Requirement("postir.entitymentions"), new Requirement("postir.slotmentions")}));
  }

  /**
   * Returns the relation mentions in the sentence, as well as the re-written slot mentions.
   * @param entityMentions The entity mentions in the sentence currently
   * @param candidateSlotMentions The slot mentions in the sentence currently. A subset of these will be
   *                              returned in the third argument of the return value
   * @param sentence The sentence we are extracting relation mentions on
   * @param knownSlots The known slots for this entity
   * @return A triple: (1) the relation mentions between the entity mentions and slot mentions;
   *                   (2) the relation mentions between all pairs of (relevant) slot mentions and entity mentions.
   *                       Thus, Julie was born in Canada and attends Stanford would extract (Julie, Canada),
   *                       (Julie, Stanford) <b>and</b> (Stanford, Canada).
   *                       Note that (Canada, Stanford) is not extracted, as Canada is not a person or entity;
   *                   (3) the new slot mentions.
   */
  private Triple<List<RelationMention>, List<RelationMention>, List<EntityMention>> getRelationAndNewSlotMentions(
      List<EntityMention> entityMentions,
      List<EntityMention> candidateSlotMentions,
      CoreMap sentence,
      List<KBPSlotFill> knownSlots) {

    List<RelationMention> relations = new ArrayList<RelationMention>();        // The relation mentions (for the pivot entity) we will return
    List<RelationMention> allRelations = new ArrayList<RelationMention>();    // if annotateMode == ALL_PAIRS, this will hold all pairwise relations
    List<EntityMention> validSlotMentions = new ArrayList<EntityMention>();  // The new set of valid slot mentions, in case we find something from knownSlots

    // (0) Compute relevant information
    // Variables we'll need
    boolean[] absorbedSlots = new boolean[sentence.get(TokensAnnotation.class).size()];
    Arrays.fill(absorbedSlots, false);
    Set<KBPair> extractedPairs = new HashSet<KBPair>();  // If an entity or fill appears twice in a sentence, only extract one of the pairs
    List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
    // Tokens which are known entities
    boolean[] entityMask = new boolean[tokens.size()];
    List<Span> entitySpans = new ArrayList<Span>();
    for (EntityMention entityMention : entityMentions) {
      for (int k : entityMention.getExtent()) { entityMask[k] = true; }
      entitySpans.add(entityMention.getExtent());
    }

    // (1) Add all directly known relations (known slots)
    // This is the usual case for training, as we know the relations we're
    // interested in and what their slot values are.
    // Step 1: Get known slots
    Map<Span, Collection<Pair<Boolean, KBPSlotFill>>> directSlotMatches = new HashMap<Span, Collection<Pair<Boolean, KBPSlotFill>>>();
    List<Span> directSlotMatchesKeys = new ArrayList<Span>();
    for (KBPSlotFill slot : knownSlots) {
      logger.debug("attempting to match slot " + slot.key.relationName + ":" + slot.key.slotValue);
      for (Pair<Span, Boolean> pair : matchSlotInSentence(slot, sentence, entityMask)) {
        if (!directSlotMatches.containsKey(pair.first)) {
          directSlotMatchesKeys.add(pair.first);
          directSlotMatches.put(pair.first, new ArrayList<Pair<Boolean, KBPSlotFill>>());
        }
        directSlotMatches.get(pair.first).add(Pair.makePair(pair.second, slot));
      }
    }
    Collections.sort(directSlotMatchesKeys, new Comparator<Span>() {
      @Override public int compare(Span o1, Span o2) { return o2.size() - o1.size(); }
    });
    assert directSlotMatchesKeys.size() < 2 || directSlotMatchesKeys.get(0).size() >= directSlotMatchesKeys.get(1).size();
    // Step 2: Add relation mentions
    OUTER: for (Span slotSpan : directSlotMatchesKeys) {
      // Ignore already absorbed slots
      for (int k : slotSpan) { if (absorbedSlots[k]) { continue OUTER; } }
      // Ignore slots which are too far from the entity
      if (!Utils.closeEnough(slotSpan, entitySpans)) { continue; }
      // Mark this slot as taken
      for (int k : slotSpan) {absorbedSlots[k] = true; }
      // Get all known relations for span
      Set<String> relationLabels = new HashSet<String>();
      Maybe<NERTag> fillType = Maybe.Nothing();
      String normalizedValue = null;
      for (Pair<Boolean, KBPSlotFill> slotFill : directSlotMatches.get(slotSpan)) {
        relationLabels.add(slotFill.second.key.relationName);
        fillType = fillType.orElse(slotFill.second.key.slotType);
        if (slotFill.first) { normalizedValue = slotFill.second.key.slotValue; } // set the normalized value to be the exactly matched fill
      }
      if (!fillType.isDefined()) { fillType = voteOnSpanNEType(tokens, slotSpan); }
      if (normalizedValue == null) { normalizedValue = directSlotMatches.get(slotSpan).iterator().next().second.key.slotValue; }
      String concatenatedLabel = concatenateLabels(relationLabels);
      // Create slot mention
      final EntityMention slotValue = new EntityMention(
          Utils.makeEntityMentionId(Maybe.<String>Nothing()), sentence, slotSpan, slotSpan,
          fillType.getOrElse(NERTag.MISC).name, null, null);
      assert slotValue.getType() != null && !slotValue.getType().equals("") && !slotValue.getType().equals(Props.NER_BLANK_STRING);
      validSlotMentions.add(slotValue);
      // Add relation for each entity
      for (final EntityMention entityMention : entityMentions) {
        // Create arg
        List<ExtractionObject> args = new ArrayList<ExtractionObject>();
        args.add(entityMention);
        args.add(slotValue);
        // Create relation
        NormalizedRelationMention rm = new NormalizedRelationMention(normalizedValue, this.entity,
            "RM" + relationMentionCount.incrementAndGet(), sentence, ExtractionObject.getSpan(
            entityMention, slotValue), concatenatedLabel, null, args, null);
        // Create KBPair
        KBPair extractedPair = KBPNew.from(entity).slotValue(slotValue.getExtentString()).KBPair();
        // Add relation
        if (!extractedPairs.contains(extractedPair)) {
          relations.add(rm);
          extractedPairs.add(extractedPair);
        } else {
          logger.debug("ignoring relation mention (another already exists between these objects): " + rm.getNormalizedEntity().name + " [" + CoreMapUtils.sentenceSpanString(tokens, rm.getArg(0).getExtent()) + "] :: " + rm.getNormalizedSlotValue());
        }
        logger.debug("found [known] relation mention: " + rm.getNormalizedEntity().name + " [" + CoreMapUtils.sentenceSpanString(tokens, rm.getArg(0).getExtent()) + "] :: " + rm.getNormalizedSlotValue());
      }
    }

    // (2) Compute slot candidates which have not been absorbed by (1)
    // This is necessary, as otherwise we may have created a slot value in (1)
    // that contradicts a slot value we will be considering in (3), and we do not want
    // both to co-exist in validSlotMentions.
    // Yet, we would like to know validSlotMentions ahead of time, in case we are annotating
    // relations between all pairs of objects in the sentence (thus, also between valid slot values).
    List<EntityMention> unabsorbedCandidateSlotMentions = new ArrayList<EntityMention>();
    OUTER: for (final EntityMention slotValue : candidateSlotMentions) {
      // Make sure we haven't seen this slot before
      for (int k : slotValue.getExtent()) { if (absorbedSlots[k]) { continue OUTER; } }
      for (int k : slotValue.getExtent()) {absorbedSlots[k] = true; }
      // Register this slot as valid
      validSlotMentions.add(slotValue);
      unabsorbedCandidateSlotMentions.add(slotValue);
    }

    // (3) Add other possible relations
    // This is the usual case for testing, when we just want to propose
    // all sorts of interesting stuff.
    for (final EntityMention slotValue : unabsorbedCandidateSlotMentions) {
      // Compute what we should count as an "entity"
      // In any event, keep track of which ones are the actual entity, as these will be flagged with the
      // KBPOfficialEntity passed into the annotator.
      List<Pair<EntityMention, Boolean>> mentionsToTreatAsEntities = new ArrayList<Pair<EntityMention, Boolean>>();
      for (EntityMention entityMention : entityMentions) {
        mentionsToTreatAsEntities.add(Pair.makePair(entityMention, true));
      }
      if (annotateMode == KBPProcess.AnnotateMode.ALL_PAIRS) {  // Also extract relation mentions between two slot fills
        for (EntityMention slotMention : validSlotMentions) {   // note: only valid slot mentions are added here!
          for (NERTag nerType : NERTag.fromString(slotMention.getMentionType())) {  // if it has an NER
            if (nerType == NERTag.PERSON || nerType == NERTag.ORGANIZATION) {  // and that NER is a person or organization
              mentionsToTreatAsEntities.add(Pair.makePair(slotMention, false));  // .. then, add it as an entity
            }
          }
        }
      }
      // Add relations for each entity
      for (final Pair<EntityMention, Boolean> entityMentionAndIsEntity : mentionsToTreatAsEntities) {
        if (entityMentionAndIsEntity.first == slotValue) { continue; } // no reflexive relations
        // Create arg
        List<ExtractionObject> args = new ArrayList<ExtractionObject>();
        args.add(entityMentionAndIsEntity.first);
        args.add(slotValue);
        // Create normalized slot
        String normSlot = slotValue.getExtentString();  // default, if there's no better antecedent
        if (Props.PROCESS_RELATION_NORMALIZECOREFSLOT) {
          Counter<String> normSlotVotes = new ClassicCounter<String>();  // try to find a better slot from coref
          for (int k : slotValue.getExtent()) {
            if (tokens.get(k).containsKey(AntecedentAnnotation.class)) { normSlotVotes.incrementCount(tokens.get(k).get(AntecedentAnnotation.class)); }
          }
          if (normSlotVotes.size() > 0) {
            String mostLikelyAntecedent = Counters.argmax(normSlotVotes);
            if (mostLikelyAntecedent.length() > normSlot.length()) { normSlot = mostLikelyAntecedent; }
          }
        }
        // Get (or mock) entity
        final KBPEntity entity;
        if (entityMentionAndIsEntity.second) {
          // Case: the entity is the official entity
          entity = this.entity;
        } else {
          // Case: the "entity" is really another slot value, and we need to mock it as an entity
          //       This happens when AnnotateMOde.ALL_PAIRS is active
          Counter<String> antecedentVotes = new ClassicCounter<String>();
          Counter<NERTag> nerVotes = new ClassicCounter<NERTag>();
          for (int k : entityMentionAndIsEntity.first.getExtent()) {
            CoreLabel token = tokens.get(k);
            if ((token.ner().equals(NERTag.PERSON.name) || token.ner().equals(NERTag.ORGANIZATION.name))) {
              if (token.containsKey(AntecedentAnnotation.class)) {
                antecedentVotes.incrementCount(token.get(AntecedentAnnotation.class), 1.0);
              }
              if (token.ner().equals(NERTag.PERSON.name)) { nerVotes.incrementCount(NERTag.PERSON, 1.0); }
              if (token.ner().equals(NERTag.ORGANIZATION.name)) { nerVotes.incrementCount(NERTag.ORGANIZATION, 1.0); }
            }
          }
          if (antecedentVotes.size() > 0 && nerVotes.size() > 0) {
            entity = KBPNew.entName(Counters.argmax(antecedentVotes)).entType(Counters.argmax(nerVotes)).KBPOfficialEntity();
          } else if (nerVotes.size() > 0) {
            entity = KBPNew.entName(entityMentionAndIsEntity.first.getExtentString()).entType(Counters.argmax(nerVotes)).KBPOfficialEntity();
          } else {
            warn(RED, "unknown NER type for entity: " + entityMentionAndIsEntity.first.getExtentString());
            entity = KBPNew.entName(entityMentionAndIsEntity.first.getExtentString()).entType(NERTag.PERSON).KBPOfficialEntity();
          }
        }
        // Create relation
        NormalizedRelationMention rm = new NormalizedRelationMention(normSlot, entity,
            "RM" + relationMentionCount.incrementAndGet(), sentence, ExtractionObject.getSpan(
            entityMentionAndIsEntity.first, slotValue), RelationMention.UNRELATED, null, args, null);
        // Create KBPair
        KBPair extractedPair = KBPNew.from(entity).slotValue(slotValue.getExtentString()).KBPair();
        // Add relation
        if (!extractedPairs.contains(extractedPair)) {
          if (entityMentionAndIsEntity.second) {
            relations.add(rm);
          }
          allRelations.add(rm);
          extractedPairs.add(extractedPair);
        } else {
          if(Props.KBP_VERBOSE) logger.debug("ignoring relation mention (another already exists between these objects): " +
              (rm.getNormalizedEntity() != null ? rm.getNormalizedEntity().name : rm.getArg(0).getExtentString()) +
              " [" + CoreMapUtils.sentenceSpanString(tokens, rm.getArg(0).getExtent()) + "] :: " + rm.getNormalizedSlotValue());
        }
        if (Props.KBP_VERBOSE) logger.debug("found [possible] relation mention: " +
            (rm.getNormalizedEntity() != null ? rm.getNormalizedEntity().name : rm.getArg(0).getExtentString()) +
            " [" + CoreMapUtils.sentenceSpanString(tokens, rm.getArg(0).getExtent()) + "] :: " + rm.getNormalizedSlotValue());

      }
    }
    return Triple.makeTriple(relations, allRelations, validSlotMentions);
  }


  /**
   * Finds all token spans where this slot matches over the sentence tokens This
   * includes matching the alternates values of the given slot!
   * This is used primarily to bypass the NER tag, and look directly for slot fills
   * when known.
   *
   * @param slot The slot fill we would like to match in this sentence
   * @param sentence The sentence we are considering
   * @return A list of matching spans, whether or not they are exact.
   */
  private static List<Pair<Span, Boolean>> matchSlotInSentence(KBPSlotFill slot, CoreMap sentence, boolean[] entityMask) {
    List<Pair<Span, Boolean>> matchingSpans = new ArrayList<Pair<Span, Boolean>>();
    // Get possible slot names
    List<String[]> names = new ArrayList<String[]>();
    String[] slotValueTokens = CoreMapUtils.tokenizeToStrings(slot.key.slotValue);
    names.add(slotValueTokens);
    names.addAll(alternateSlotValues(slot));
    Collections.sort(names, new Comparator<String[]>() {
      @Override public int compare(String[] o1, String[] o2) { return o2.length - o1.length; }
    });
    assert names.size() < 2 || names.get(0).length >= names.get(1).length;

    List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
    boolean[] used = new boolean[entityMask.length];
    System.arraycopy(entityMask, 0, used, 0, entityMask.length);
    for (String[] name : names) {
      int nameIndex = 0;
      for (int i = 0; i < tokens.size(); ++i) {
        if (used[i]) { nameIndex = 0; continue; } // already matched with something else. Try again...
        CoreLabel token = tokens.get(i);
        // Logic for string match
        if (name[nameIndex].equals(token.originalText()) ||
            name[nameIndex].equals(token.word())) {
          nameIndex += 1;
        } else {
          nameIndex = 0;
        }
        if (nameIndex >= name.length) {
          // Case: found the slot fill
          Span slotFillSpan = new Span(i + 1 - nameIndex, i + 1);
          for (int k : slotFillSpan) { used[k] = true; }
          matchingSpans.add(Pair.makePair(slotFillSpan, Arrays.equals(name, slotValueTokens)));
          nameIndex = 0;
        }
      }
    }
    return matchingSpans;
  }

  public static List<String[]> alternateSlotValues(KBPSlotFill fill) {
    List<String[]> alternateSlotValues = new ArrayList<String[]>();
    for( RelationType relation : fill.key.tryKbpRelation() ) {
      List<String> alternates = findAlternateSlotValues(fill.key.slotValue,
          relation.isDateRelation(), relation.isPersonNameRelation());
      for(String alt: alternates) alternateSlotValues.add(CoreMapUtils.tokenizeToStrings(alt));
      Collections.sort(alternateSlotValues, new Comparator<String []>() {

        public int compare(String[] o1, String[] o2) {
          if(o1.length > o2.length) return -1;
          if(o1.length == o2.length) return 0;
          return 1;
        }
      });
    }
    return alternateSlotValues;
  }

  /**
   * Determine a named entity type for a given span in a sentence, based on voting of all the tokens in
   * the sentence.
   * @param tokens The sentence
   * @param span The span of the entity / slot value in the sentence
   * @return A FillType (if one can be found) for the given span
   */
  private Maybe<NERTag> voteOnSpanNEType(List<CoreLabel> tokens, Span span) {
    Counter<NERTag> votes = new ClassicCounter<NERTag>();
    for (int k : span) {
      for (NERTag fill : NERTag.fromString(tokens.get(k).ner())) {
        votes.incrementCount(fill, 1.0);
      }
    }
    if (votes.size() > 0) { return Maybe.Just(Counters.argmax(votes)); } else { return Maybe.Nothing(); }
  }

  /**
   * A small utility function to concatenate a bunch of relations together
   * @param labels The relations to concatenate
   * @return A String representation of the concatenated labels (relations)
   */
  private static String concatenateLabels(Set<String> labels) {
    StringBuilder os = new StringBuilder();
    boolean first = true;
    List<String> sortedLabels = new ArrayList<String>(labels);
    Collections.sort(sortedLabels);
    for (String label : sortedLabels) {
      if (!first) os.append("|");
      os.append(label);
      first = false;
    }
    return os.toString();
  }



  //
  // Code for finding alternate slot values
  //
  public static final Pattern YEAR = Pattern.compile("[12]\\d\\d\\d");

  /**
   * Finds alternate values for the given slot
   * This includes: keeping just the year for dates; removing titles and middle names for person names
   */
  private static List<String> findAlternateSlotValues(String slotValue,
                                              boolean isDateSlot,
                                              boolean isPersonSlot) {
    List<String> alternates = new ArrayList<String>();

    if(isDateSlot){
      // matching just the year is fine
      Matcher m = YEAR.matcher(slotValue);
      if(m.find()){
        String year = m.group();
        if(year.length() < slotValue.length()){
          alternates.add(year);
        }
      }
    }

    if(isPersonSlot){
      List<String> alternatePersonNames = findPersonAlternateNames(slotValue);
      alternates.addAll(alternatePersonNames);
    }

    return alternates;
  }

  private static final Set<String> PERSON_PREFIXES = new HashSet<String>(Arrays.asList(new String[]{
      "mr", "mr.", "ms", "ms.", "mrs", "mrs.", "miss", "mister", "sir", "dr", "dr."
  }));
  private static final Set<String> PERSON_SUFFIXES = new HashSet<String>(Arrays.asList(new String[]{
      "jr", "jr.", "sr", "sr.", "i", "ii", "iii", "iv"
  }));

  private static List<String> findPersonAlternateNames(String fullName) {
    List<String> alternates = new ArrayList<String>();

    //
    // matching first name last name is fine => remove middle name
    //
    List<CoreLabel> tokens = CoreMapUtils.tokenize(fullName);
    int start = 0, end = tokens.size() - 1;
    while(start < end){
      if(PERSON_PREFIXES.contains(tokens.get(start).word().toLowerCase())) start ++;
      else break;
    }
    while(end > start){
      if(PERSON_SUFFIXES.contains(tokens.get(end).word().toLowerCase())) end --;
      else break;
    }
    if(start < end - 1){
      String firstlast = tokens.get(start).word() + " " + tokens.get(end).word();
      alternates.add(firstlast);
    }

    return alternates;
  }

}
