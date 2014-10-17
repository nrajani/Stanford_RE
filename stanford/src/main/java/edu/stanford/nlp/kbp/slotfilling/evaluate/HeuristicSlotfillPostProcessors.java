package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.time.TimeAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.logging.Redwood;

import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A collection of post processors
 *
 * @author Gabor Angeli
 */
public class HeuristicSlotfillPostProcessors {

  private static final Redwood.RedwoodChannels logger = Redwood.channels("Filter");

  protected static Maybe<KBPSlotFill> singletonFailure(KBPSlotFill toFail, Class<?> processor) {
    logger.debug("" + toFail + " failed " + processor.getSimpleName());
    return Maybe.Nothing();
  }

  protected static boolean nonlocalFailure(KBPSlotFill toFail, Class<?> processor) {
    logger.debug("" + toFail + " failed " + processor.getSimpleName());
    return false;
  }

  //
  // Single Slot Consistency Checks
  //

  /** Don't fill PER slots for Organizations and visa versa; and, that the slot value NER type is valid. */
  public static class RespectRelationTypes extends HeuristicSlotfillPostProcessor.Default {
    @Override
    public Maybe<KBPSlotFill> isValidSlotAndRewrite(KBPEntity pivot, KBPSlotFill candidate) {
      if (!candidate.key.hasKBPRelation()) { return Maybe.Just(candidate); }
      RelationType relation = candidate.key.kbpRelation();
      // Try to fix up the type
      if (pivot.type != relation.entityType && Props.TEST_CONSISTENCY_FLEXIBLETYPES) {
        if (relation == RelationType.PER_CITIES_OF_RESIDENCE) {
          candidate = KBPNew.from(candidate).rel(RelationType.ORG_CITY_OF_HEADQUARTERS).KBPSlotFill(); }
        if (relation == RelationType.PER_STATE_OR_PROVINCES_OF_RESIDENCE) {
          candidate = KBPNew.from(candidate).rel(RelationType.ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS).KBPSlotFill(); }
        if (relation == RelationType.PER_COUNTRIES_OF_RESIDENCE) {
          candidate = KBPNew.from(candidate).rel(RelationType.ORG_COUNTRY_OF_HEADQUARTERS).KBPSlotFill(); }
        if (relation == RelationType.PER_ALTERNATE_NAMES) {
          candidate = KBPNew.from(candidate).rel(RelationType.ORG_ALTERNATE_NAMES).KBPSlotFill(); }
        if (relation == RelationType.PER_MEMBER_OF) {
          candidate = KBPNew.from(candidate).rel(RelationType.ORG_MEMBER_OF).KBPSlotFill(); }
        if (relation == RelationType.ORG_CITY_OF_HEADQUARTERS) {
          candidate = KBPNew.from(candidate).rel(RelationType.PER_CITIES_OF_RESIDENCE).KBPSlotFill(); }
        if (relation == RelationType.ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS) {
          candidate = KBPNew.from(candidate).rel(RelationType.PER_STATE_OR_PROVINCES_OF_RESIDENCE).KBPSlotFill(); }
        if (relation == RelationType.ORG_COUNTRY_OF_HEADQUARTERS) {
          candidate = KBPNew.from(candidate).rel(RelationType.PER_COUNTRIES_OF_RESIDENCE).KBPSlotFill(); }
        if (relation == RelationType.ORG_ALTERNATE_NAMES) {
          candidate = KBPNew.from(candidate).rel(RelationType.PER_ALTERNATE_NAMES).KBPSlotFill(); }
        if (relation == RelationType.ORG_MEMBER_OF) {
          candidate = KBPNew.from(candidate).rel(RelationType.PER_MEMBER_OF).KBPSlotFill(); }
        relation = candidate.key.kbpRelation();
      }
      // Give up
      if (pivot.type != relation.entityType) {
        return singletonFailure(candidate, this.getClass());
      }
      for (NERTag slotType : candidate.key.slotType) {
        if (!relation.validNamedEntityLabels.contains(slotType)) {
          logger.debug("incompatible relation type: " + candidate + " of type " +  candidate.key.slotType);
          return singletonFailure(candidate, this.getClass());
        }
      }
      return Maybe.Just(candidate);
    }
  }

  /** Filter slots which are explicitly ignored in the entity */
  public static class FilterIgnoredSlots extends HeuristicSlotfillPostProcessor.Default {
    @Override
    public Maybe<KBPSlotFill> isValidSlotAndRewrite(KBPEntity pivot, KBPSlotFill candidate) {
      if (pivot instanceof KBPOfficialEntity && candidate.key.hasKBPRelation()) {
        return ((KBPOfficialEntity) pivot).ignoredSlots.getOrElse(new HashSet<RelationType>()).contains(candidate.key.kbpRelation()) ? singletonFailure(candidate, this.getClass()): Maybe.Just(candidate);
      } else {
        return Maybe.Just(candidate);
      }
    }
  }

  /** Filter slots which are too close to entries already in the KB */
  public static class FilterAlreadyKnownSlots extends HeuristicSlotfillPostProcessor.Default {
    private KBPIR irComponent;
    public FilterAlreadyKnownSlots(KBPIR irComponent) {
      this.irComponent = irComponent;
    }
    @Override
    public Maybe<KBPSlotFill> isValidSlotAndRewrite(KBPEntity pivot, KBPSlotFill candidate) {
      if (!candidate.key.hasKBPRelation()) { return Maybe.Just(candidate); }
      // for list-valued relations, discard the relation if the slot-value
      // is sufficiently similar to an existing relation from the KB
      for (KBPSlotFill kbRelation : irComponent.getKnownSlotFillsForEntity(pivot)) {
        // Filter exact match
        if (candidate.key.slotValue.equals(kbRelation.key.slotValue)) return singletonFailure(candidate, this.getClass());
        // Filter substring matches
        List<CoreLabel> tokens = CoreMapUtils.tokenize(candidate.key.slotValue.toLowerCase());
        List<CoreLabel> otherTokens = CoreMapUtils.tokenize(kbRelation.key.slotValue.toLowerCase());
        if (CoreMapUtils.contained(tokens, otherTokens, true) || CoreMapUtils.contained(otherTokens, tokens, true)) {
          return singletonFailure(candidate, this.getClass());
        }
      }
      // Filter same alternate name
      if (candidate.key.kbpRelation() == RelationType.PER_ALTERNATE_NAMES || candidate.key.kbpRelation() == RelationType.ORG_ALTERNATE_NAMES) {
        if (candidate.key.slotValue.equalsIgnoreCase(pivot.name)) {
          return singletonFailure(candidate, this.getClass());
        }
        if (pivot.type == NERTag.PERSON && !candidate.key.slotValue.contains(" ") &&
            pivot.name.toLowerCase().startsWith(candidate.key.slotValue.toLowerCase()) || pivot.name.toLowerCase().endsWith(candidate.key.slotValue.toLowerCase())) {
          // Single token name
          return singletonFailure(candidate, this.getClass());
        }
      }
      return Maybe.Just(candidate);
    }
  }

  /** Don't be silly and propose slot fills with probability less than 1% */
  public static class FilterVeryLowProbabilitySlots extends HeuristicSlotfillPostProcessor.Default {
    public static final double threshold = 0.01;
    @Override
    public Maybe<KBPSlotFill> isValidSlotAndRewrite(KBPEntity pivot, KBPSlotFill candidate) {
      return candidate.score.getOrElse(1.0) >= threshold ? Maybe.Just(candidate) : singletonFailure(candidate, this.getClass());
    }
  }

  /** Some basic sanity checks for absolute nonsense */
  public static class SanityCheckFilter extends HeuristicSlotfillPostProcessor.Default {
    @SuppressWarnings("EmptyCatchBlock")
    public Maybe<KBPSlotFill> isValidSlotAndRewrite(KBPEntity pivot, KBPSlotFill candidate) {
      if (!candidate.key.hasKBPRelation()) { return Maybe.Just(candidate); }
      // Filter way too long slots
      if (candidate.key.slotValue.length() > 80) { return singletonFailure(candidate, this.getClass()); }
      // Filter unreasonable ages
      try {
        if (candidate.key.kbpRelation() == RelationType.PER_AGE && (Integer.parseInt(candidate.key.slotValue) > 125 || Integer.parseInt(candidate.key.slotValue) <= 0)) { return singletonFailure(candidate, this.getClass()); }
      } catch (NumberFormatException e) { }
      // Filter symmetric entity / slot value
      if (candidate.key.entityName.equals(candidate.key.slotValue)) { return singletonFailure(candidate, this.getClass()); }
      // Everything's ok
      return Maybe.Just(candidate);
    }

  }

  /** Conform to potentially unintuitive specifications outlined in the guidelines */
  public static class ConformToGuidelinesFilter extends HeuristicSlotfillPostProcessor.Default {
    // A set of explicitly invalid top employees
    public static final Set<String> invalidTopEmployeeJustification = new HashSet<String>() {{
      add("spokesperson"); add("spokesman"); add("spokeswoman"); add("chief customer officer"); add("cco");
      add("information officer"); add("chief compliance officer"); add("frontman"); add("secretary of information");
      add("supreme court justice"); add("house minority leader"); add("press secretary"); add("representative");
      add("senior advisor"); add("senior editor"); add("member");
    }};

    // Generic terms like these are not valid alternate names
    public static final Set<String> invalidOrgAltNames = new HashSet<String>() {{
      add("association"); add("society"); add("group"); add("corporation"); add("corp"); add("corp.");
      add("llc"); add("llc");
    }};

    // A set of explicitly invalid titles
    public static final Set<String> invalidTitles = new HashSet<String>() {{
      add("senior leader"); add("leader"); add("member"); add("hero"); add("socialite");
    }};

    // "Partial" Timex regexps
    public static final Pattern YEAR = Pattern.compile("[12][0-9]{3}");
    public static final Pattern YEAR_ONLY = Pattern.compile("[12][0-9X]{3}");
    public static final Pattern YEAR_MONTH = Pattern.compile("[12X][0-9X]{3}-[0-9X]{2}");

    @Override
    public Maybe<KBPSlotFill> isValidSlotAndRewrite(KBPEntity pivot, KBPSlotFill candidate) {
      // Get span between the entity and the slot fill
      Maybe<String> spanBetweenArgs = Maybe.Nothing();
      if (candidate.provenance.isDefined() && candidate.provenance.get().containingSentenceLossy.isDefined() &&
          candidate.provenance.get().slotValueMentionInSentence.isDefined() &&
          candidate.provenance.get().entityMentionInSentence.isDefined()) {
        Span entitySpan = candidate.provenance.get().entityMentionInSentence.get();
        Span slotSpan = candidate.provenance.get().slotValueMentionInSentence.get();
        Span justSpan = candidate.provenance.get().justificationMention.get();
        spanBetweenArgs = Maybe.Just(CoreMapUtils.sentenceSpanString(candidate.provenance.get().containingSentenceLossy.get(),
            new Span(Math.min(entitySpan.end()-justSpan.start(), slotSpan.end()-justSpan.start()), Math.max(entitySpan.start()-justSpan.start(), slotSpan.start()-justSpan.start()))));
      }
      Maybe<String> originalSlotValue = Maybe.Nothing();
      if (candidate.provenance.isDefined() && candidate.provenance.get().containingSentenceLossy.isDefined() &&
          candidate.provenance.get().slotValueMentionInSentence.isDefined()) {
        originalSlotValue = Maybe.Just(CoreMapUtils.sentenceSpanString(candidate.provenance.get().containingSentenceLossy.get(), candidate.provenance.get().slotValueMentionInSentence.get()));
      }

      // Filters
      for (String span : spanBetweenArgs) {
        String spanToLower = span.toLowerCase();
        for (RelationType rel : candidate.key.tryKbpRelation()) {
          switch (rel) {
            case ORG_TOP_MEMBERS_SLASH_EMPLOYEES:
              if (invalidTopEmployeeJustification.contains(spanToLower)) { return singletonFailure(candidate, this.getClass()); }
              break;
            case ORG_ALTERNATE_NAMES:
              if (invalidOrgAltNames.contains(spanToLower)) { return singletonFailure(candidate, this.getClass()); }
              break;
            default:
              break;
          }
        }
      }

      for (RelationType rel : candidate.key.tryKbpRelation()) {
        if (rel.isDateRelation()) {
          // Fail relative times (save date of death)
          boolean hasYearInOriginalValue = true;
          for (String slotValue : originalSlotValue) { hasYearInOriginalValue = YEAR.matcher(slotValue).find(); }
          if (!YEAR.matcher(candidate.key.slotValue).find()) { hasYearInOriginalValue = false; }
          if (!hasYearInOriginalValue && rel != RelationType.PER_DATE_OF_DEATH &&
              rel != RelationType.ORG_DISSOLVED) { return singletonFailure(candidate, this.getClass()); }
          // Fail ranges (e.g., "1991 to 1992")
          for (String slotValue : originalSlotValue) {
            if (slotValue.contains("to") || slotValue.contains("until") || slotValue.contains(" -- ")) { return singletonFailure(candidate, this.getClass()); }
          }
          // Normalize the timex string
          if (YEAR_ONLY.matcher(candidate.key.slotValue).matches()) { return Maybe.Just(KBPNew.from(candidate).slotValue(candidate.key.slotValue + "-XX-XX").KBPSlotFill()); }
          if (YEAR_MONTH.matcher(candidate.key.slotValue).matches()) { return Maybe.Just(KBPNew.from(candidate).slotValue(candidate.key.slotValue + "-XX").KBPSlotFill()); }
        }
        if (rel == RelationType.PER_TITLE && invalidTitles.contains(candidate.key.slotValue)) { return singletonFailure(candidate, this.getClass()); }
      }

      // Return
      return Maybe.Just(candidate);
    }
  }

  //
  // Rewrites
  //

  /** Sanity check web pages, and convert to base URLs */
  public static class FilterUnrelatedURL extends HeuristicSlotfillPostProcessor.Default {
    public static final double minNGramOverlap = 4;
    public static final Pattern baseURL = Pattern.compile("(?:(?:.*)://(?:[wW]{3}\\.)?|[wW]{3}\\.)([^:/]*)/?");

    private boolean hasOverlap(String name, String url) {
      for (int length = Math.min(name.length(), url.length()); length >= Math.min(minNGramOverlap, name.length()); --length) {
        for (int entityStart = 0; entityStart <= name.length() - length; ++entityStart) {
          for (int slotStart = 0; slotStart <= url.length() - length; ++slotStart) {
            if (name.substring(entityStart, entityStart + length).equalsIgnoreCase(url.substring(slotStart, slotStart + length))) {
              // Case: we found at least an n-gram overlap (see minNGramOverlap)
              return true;
            }
          }
        }
      }
      // Case: there's no large enough overlap between the entity and the URL
      return false;
    }
    @Override
    public Maybe<KBPSlotFill> isValidSlotAndRewrite(KBPEntity pivot, KBPSlotFill candidate) {
      //noinspection LoopStatementThatDoesntLoop
      for (RelationType relation : candidate.key.tryKbpRelation()) {
        if (relation == RelationType.ORG_WEBSITE) {
          boolean matchesEntity =
                 hasOverlap(pivot.name.toLowerCase().replaceAll("\\s+", ""), candidate.key.slotValue.toLowerCase()) || // direct match
                 hasOverlap(pivot.name.toLowerCase().replaceAll("(^| )([a-z])[^ ]*","$2"), candidate.key.slotValue.toLowerCase()) ||  // accronym
                 hasOverlap(pivot.name.toLowerCase().replaceAll("and|or|of|the", "").replaceAll("(^| +)([a-z])[^ ]*","$2"), candidate.key.slotValue.toLowerCase());
          Matcher m;
          if (matchesEntity && (m = baseURL.matcher(candidate.key.slotValue)).find()) {
            KBPSlotFill rewritten = KBPNew.from(candidate).slotValue(m.group()).slotType(NERTag.URL).KBPSlotFill();
            return Maybe.Just(rewritten);
          } else {
            return singletonFailure(candidate, this.getClass());
          }
        } else {
          return Maybe.Just(candidate);
        }
      }
      return Maybe.Just(candidate);
    }
  }

  /** Replace a slot fill with its canonical mention */
  public static class CanonicalMentionRewrite extends HeuristicSlotfillPostProcessor.Default {
    private static final Pattern INTEGER = Pattern.compile("([0-9]+)");

    @Override
    public Maybe<KBPSlotFill> isValidSlotAndRewrite(KBPEntity pivot, KBPSlotFill candidate) {
      Maybe<List<CoreLabel>> provenanceSpan = Maybe.Nothing();
      // Case: Rewrite mentions to their antecedents (person and organization)
      // If we already have provenance...
      if (candidate.provenance.isDefined() && candidate.provenance.get().containingSentenceLossy.isDefined() &&
          candidate.provenance.get().slotValueMentionInSentence.isDefined() &&
          // ... and it's a person or organization
          (candidate.key.slotType.equalsOrElse(NERTag.PERSON, false) ||
//           candidate.key.slotType.equalsOrElse(NERTag.ORGANIZATION, false) ||  // TODO(gabor) why does this cause a 2 F1 reduction?
           (candidate.key.hasKBPRelation() && candidate.key.kbpRelation().isDateRelation()))
          ) {
        CoreMap lossySentence = candidate.provenance.get().containingSentenceLossy.get();
        Span slotSpan = candidate.provenance.get().slotValueMentionInSentence.get();
        Maybe<String> antecedent = Maybe.Nothing();
        List<CoreLabel> tokens = lossySentence.get(CoreAnnotations.TokensAnnotation.class);
        provenanceSpan = Maybe.Just(tokens.subList(slotSpan.start(), slotSpan.end()));
        // ...try to find an antecedent
        for (int i = slotSpan.start(); i < slotSpan.end(); i++) {
          antecedent = antecedent.orElse(Maybe.fromNull(tokens.get(i).get(CoreAnnotations.AntecedentAnnotation.class)));
        }
        if (antecedent.isDefined() &&
            (!candidate.key.slotType.equalsOrElse(NERTag.PERSON, false) || antecedent.get().length() >= candidate.key.slotValue.length()) &&
            !antecedent.get().equalsIgnoreCase(candidate.key.slotValue)) {
          // If we found an antecedent, take that as the slot value
          KBPSlotFill rewritten = KBPNew.from(candidate).slotValue(antecedent.get()).slotType(candidate.key.slotType).KBPSlotFill();
          return Maybe.Just(rewritten);

        }
      }

      // Case: always rewrite numbers to be numbers
      @SuppressWarnings("unchecked")
      Set<NERTag> validNETags = candidate.key.hasKBPRelation() ? candidate.key.kbpRelation().validNamedEntityLabels : Collections.EMPTY_SET;
      if (validNETags.contains(NERTag.NUMBER) &&
          !INTEGER.matcher(candidate.key.slotValue).matches()) {
        Matcher m = INTEGER.matcher(candidate.key.slotValue);
        // rewrite a term that contains a number to just the number
        if (m.find()) {
          return Maybe.Just(KBPNew.from(candidate).slotValue(m.group()).slotType(candidate.key.slotType).KBPSlotFill());
        }
        // Try to find a number in the provenance
        for (List<CoreLabel> valueSpan : provenanceSpan) {
          for (CoreLabel token : valueSpan) {
            if (token.get(CoreAnnotations.NumericValueAnnotation.class) != null) {
              return Maybe.Just(KBPNew.from(candidate).slotValue(token.get(CoreAnnotations.NumericValueAnnotation.class).toString()).slotType(candidate.key.slotType).KBPSlotFill());
            }
          }
        }
      }

      // Case: always rewrite dates to their canonical form
      if (validNETags.contains(NERTag.DATE) || validNETags.contains(NERTag.DURATION)) {
        for (List<CoreLabel> valueSpan : provenanceSpan) {
          for (CoreLabel token : valueSpan) {
            if (token.get(TimeAnnotations.TimexAnnotation.class) != null &&
                token.get(TimeAnnotations.TimexAnnotation.class).value() != null) {
              return Maybe.Just(KBPNew.from(candidate).slotValue(token.get(TimeAnnotations.TimexAnnotation.class).value()).slotType(candidate.key.slotType).KBPSlotFill());
            }
          }
        }
      }

      // Return the candidate as is
      return Maybe.Just(candidate);
    }
  }

  /** Replace a slot fill with its cannonical mention */
  public static class ExpandToMaximalPhraseRewrite extends HeuristicSlotfillPostProcessor.Default {
    @Override
    public Maybe<KBPSlotFill> isValidSlotAndRewrite(KBPEntity pivot, KBPSlotFill candidate) {
      // If we already have provenance...
      if ((candidate.key.hasKBPRelation() && candidate.key.kbpRelation() == RelationType.PER_TITLE) &&  // don't do for every relation
          candidate.provenance.isDefined() && candidate.provenance.get().containingSentenceLossy.isDefined() &&
          candidate.provenance.get().slotValueMentionInSentence.isDefined()) {
        KBPRelationProvenance provenance = candidate.provenance.get();
        CoreMap lossySentence = provenance.containingSentenceLossy.get();
        if(lossySentence==null){
        	System.out.println("Lossy sentence is null!!!!");
        }
        else{
        	System.out.println(lossySentence.get(CoreAnnotations.TokensAnnotation.class));
        }
        Span slotSpan = provenance.slotValueMentionInSentence.get();
        // ... Try to absorb as many modifiers as possible
        int newStart = slotSpan.start()-provenance.justificationMention.get().start();
        for (int candidateStart = newStart - 1; candidateStart >= 0; candidateStart -= 1) {
          CoreLabel token = lossySentence.get(CoreAnnotations.TokensAnnotation.class).get(candidateStart);
//          if (token.tag().toLowerCase().startsWith("n") && // while preceded by a noun
//              (token.ner().equals(RelationType.FillType.TITLE.name) || token.ner().equals("O")) &&  // which has no NE tag
//              !token.containsKey(CoreAnnotations.AntecedentAnnotation.class)) { // and has no antecedent
          if (token.tag().toLowerCase().startsWith("n") && !token.tag().toLowerCase().endsWith("p")) {// while preceded by a noun
            newStart = candidateStart; // ... suck up that word as well
          } else {
            break;
          }
        }

        if (newStart != (slotSpan.start()-provenance.justificationMention.get().start()-1)) {
        	System.out.println("newstart : "+newStart+ "slotSpan.start() : "+slotSpan.start());
        	System.out.println("newstart : "+newStart+ "slotSpan.start() : "+(slotSpan.start()-provenance.justificationMention.get().start()));
          while (newStart < slotSpan.end() && lossySentence.get(CoreAnnotations.TokensAnnotation.class).get(newStart).tag().toLowerCase().startsWith("cc")) { newStart += 1; } // don't start a fill with a conjunction
          // Compute new provenance
          KBPRelationProvenance newProvenance = new KBPRelationProvenance(provenance.docId, provenance.indexName,
              provenance.sentenceIndex.orCrash(), provenance.entityMentionInSentence.orCrash(),
              new Span(newStart, slotSpan.end()), provenance.containingSentenceLossy.get());
          // Compute new slot value
          StringBuilder newSlotValue = new StringBuilder();
          for (int i = newStart-1; i < slotSpan.start(); ++i) {
        	  System.out.println("newstart : "+i);
            newSlotValue.append(lossySentence.get(CoreAnnotations.TokensAnnotation.class).get(i).word()).append(" ");
          }
          newSlotValue.append(candidate.key.slotValue);
          // Rewrite slot
          KBPSlotFill rewritten = KBPNew.from(candidate).slotValue(newSlotValue.toString()).slotType(candidate.key.slotType).provenance(newProvenance).KBPSlotFill();
          return Maybe.Just(rewritten);
        }
      }
      return Maybe.Just(candidate);
    }
  }

  /** Rewrite top employees to "founded by" */
  public static class TopEmployeeRewrite extends HeuristicSlotfillPostProcessor.Default {
    @Override
    public Maybe<KBPSlotFill> isValidSlotAndRewrite(KBPEntity pivot, KBPSlotFill candidate) {
      if (candidate.key.hasKBPRelation() && candidate.key.kbpRelation() != RelationType.ORG_TOP_MEMBERS_SLASH_EMPLOYEES) { return Maybe.Just(candidate); }
      // Rewrite to FoundedBy
      // If we already have provenance...
      if (candidate.provenance.isDefined() && candidate.provenance.get().containingSentenceLossy.isDefined() &&
          candidate.provenance.get().entityMentionInSentence.isDefined() &&
          candidate.provenance.get().slotValueMentionInSentence.isDefined()) {
        KBPRelationProvenance provenance = candidate.provenance.get();
        CoreMap lossySentence = provenance.containingSentenceLossy.get();
        Span entitySpan = provenance.entityMentionInSentence.get();
        Span slotSpan = provenance.slotValueMentionInSentence.get();
        for (int i = Math.min(entitySpan.end(), slotSpan.end()); i < Math.max(entitySpan.start(), slotSpan.start()); ++i) {
          String word = lossySentence.get(CoreAnnotations.TokensAnnotation.class).get(i).word();
          if (word.equalsIgnoreCase("founded") || word.equalsIgnoreCase("founder") || word.equalsIgnoreCase("created") ||
              word.equalsIgnoreCase("creator")) {
            KBPSlotFill rewritten = KBPNew.from(candidate).rel(RelationType.ORG_FOUNDED_BY).KBPSlotFill();
            return Maybe.Just(rewritten);
          }
        }
      }
      return Maybe.Just(candidate);
    }
  }

  /** Rewrite "born in" to "resides in" */
  @SuppressWarnings("UnusedDeclaration")
  public static class BornInRewrite extends HeuristicSlotfillPostProcessor.Default {
    @Override
    public Maybe<KBPSlotFill> isValidSlotAndRewrite(KBPEntity pivot, KBPSlotFill candidate) {
      if (!candidate.key.hasKBPRelation()) { return Maybe.Just(candidate); }
      RelationType relation = candidate.key.kbpRelation();
      if (!relation.isBirthRelation()) { return Maybe.Just(candidate); }
      // Rewrite to XOfResidence
      // If we already have provenance...
      if (candidate.provenance.isDefined() && candidate.provenance.get().containingSentenceLossy.isDefined() &&
          candidate.provenance.get().entityMentionInSentence.isDefined() &&
          candidate.provenance.get().slotValueMentionInSentence.isDefined()) {
        KBPRelationProvenance provenance = candidate.provenance.get();
        CoreMap lossySentence = provenance.containingSentenceLossy.get();
        Span entitySpan = provenance.entityMentionInSentence.get();
        Span slotSpan = provenance.slotValueMentionInSentence.get();
        Span middleWords = new Span(Math.min(entitySpan.end(), slotSpan.end()), Math.max(entitySpan.start(), slotSpan.start()));
        // If keyphrases are contained between them, accept it
        for (int i : middleWords) {
          String word = lossySentence.get(CoreAnnotations.TokensAnnotation.class).get(i).word();
          if (word.equalsIgnoreCase("born") || word.toLowerCase().contains("birth") || word.toLowerCase().contains("origin") ||
              word.toLowerCase().equals("from") || word.toLowerCase().contains("native")) {
            return Maybe.Just(candidate);
          }
        }
        // If residence keyphrases are found between them, rewrite
        for (int i : middleWords) {
          String word = lossySentence.get(CoreAnnotations.TokensAnnotation.class).get(i).word();
          if (word.startsWith("reside") || word.toLowerCase().startsWith("live") || word.toLowerCase().contains("home") ||
              word.toLowerCase().startsWith("raise") || word.toLowerCase().contains("brought")) {
            final KBPSlotFill rewritten;
            if (relation == RelationType.PER_CITY_OF_BIRTH) {
              rewritten = KBPNew.from(candidate).rel(RelationType.PER_CITIES_OF_RESIDENCE).KBPSlotFill(); }
            else if (relation == RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH) {
              rewritten = KBPNew.from(candidate).rel(RelationType.PER_STATE_OR_PROVINCES_OF_RESIDENCE).KBPSlotFill(); }
            else if (relation == RelationType.PER_COUNTRY_OF_BIRTH) {
              rewritten = KBPNew.from(candidate).rel(RelationType.PER_COUNTRIES_OF_RESIDENCE).KBPSlotFill(); }
            else { throw new IllegalStateException("Unknown birth relation: " + relation); }
            return Maybe.Just(rewritten);
          }
        }
        // If the entity and fill are close enough, accept it
        if (middleWords.size() < 8) { return Maybe.Just(candidate); }
      }
      // Really very little support for birth place
      return singletonFailure(candidate, this.getClass());
    }
  }

  //
  // Pairwise Checks
  //

  /** Don't duplicate a slot fill */
  public static class NoDuplicates extends HeuristicSlotfillPostProcessor.Default {
    @Override
    public boolean pairwiseKeepLowerScoringFill(KBPEntity pivot, KBPSlotFill higherScoring, KBPSlotFill lowerScoring) {
      if (higherScoring.equals(lowerScoring)) { return nonlocalFailure(lowerScoring, this.getClass()); }
      //noinspection SimplifiableIfStatement
      if (lowerScoring.key.hasKBPRelation() && lowerScoring.key.kbpRelation() == RelationType.PER_EMPLOYEE_OF &&
          KBPNew.from(lowerScoring).rel(RelationType.PER_MEMBER_OF).KBPSlotFill().equals(higherScoring)) { return nonlocalFailure(lowerScoring, this.getClass()); }
      return true;
    }
  }

  /** Try to guess whether two slots are in the same equivalence class */
  public static class NoDuplicatesApproximate extends HeuristicSlotfillPostProcessor.Default {
    public final Maybe<KBPIR> irComponent;

    public NoDuplicatesApproximate() {
      this.irComponent = Maybe.Nothing();
    }
    public NoDuplicatesApproximate(KBPIR ir) {
      this.irComponent = Maybe.Just(ir);
    }


    @SuppressWarnings({"LoopStatementThatDoesntLoop", "SimplifiableIfStatement"})
    @Override
    public boolean pairwiseKeepLowerScoringFill(KBPEntity pivot, KBPSlotFill higherScoring, KBPSlotFill lowerScoring) {
      if (!higherScoring.key.hasKBPRelation() || !lowerScoring.key.hasKBPRelation()) { return true; }
      if (higherScoring.key.kbpRelation() == lowerScoring.key.kbpRelation()) {
        // Special case: alternate names
        // No KBP component would be complete without at least _one_ special case
        if (higherScoring.key.kbpRelation() == RelationType.PER_ALTERNATE_NAMES || higherScoring.key.kbpRelation() == RelationType.ORG_ALTERNATE_NAMES) {
          if (Utils.isValidAlternateName(higherScoring.key.slotValue, lowerScoring.key.slotValue)) {
            return true;
          } else {
            return nonlocalFailure(lowerScoring, this.getClass());  // too close according to guidelines
          }
        }

        // Classify based on provenance context, if available
        for (KBPEntity higherEntity : higherScoring.key.getSlotEntity()) {
          for (KBPEntity lowerEntity : lowerScoring.key.getSlotEntity()) {
            // Try to classify based on provenance
            for (KBPRelationProvenance higherProvenance : higherScoring.provenance) {
              for (EntityContext higherContext : higherProvenance.toSlotContext(higherEntity, irComponent)) {
                for (KBPRelationProvenance lowerProvenance : lowerScoring.provenance) {
                  for (EntityContext lowerContext : lowerProvenance.toSlotContext(lowerEntity, irComponent)) {
                    if (Props.KBP_ENTITYLINKER.sameEntity(higherContext, lowerContext)) {
                      return nonlocalFailure(lowerScoring, this.getClass());  // failure with context
                    } else {
                      return true;  // pass explicitly with context
                    }

                  }
                }
              }
            }
            // Classify based on surface string
            if (Props.KBP_ENTITYLINKER.sameEntity(new EntityContext(higherEntity), new EntityContext(lowerEntity))) {
              return nonlocalFailure(lowerScoring, this.getClass());  // failure without context
            } else {
              return true;  // pass explicitly without context
            }
          }
        }
      }
      return true;  // pass by default
    }
  }

  /** Don't propose multiple entries for single-valued slots */
  public static class DuplicateRelationOnlyInListRelations extends HeuristicSlotfillPostProcessor.Default {
    @SuppressWarnings("SimplifiableIfStatement")
    @Override
    public boolean pairwiseKeepLowerScoringFill(KBPEntity pivot, KBPSlotFill higherScoring, KBPSlotFill lowerScoring) {
      if (!higherScoring.key.hasKBPRelation() || !lowerScoring.key.hasKBPRelation()) { return true; }
      if (higherScoring.key.relationName.equals(lowerScoring.key.relationName) &&
          !higherScoring.key.slotValue.equals(lowerScoring.key.slotValue)) {
        // case: duplicate slot fills for a relation
        RelationType rel = higherScoring.key.kbpRelation();
        if (rel.cardinality != RelationType.Cardinality.SINGLE) {
          return true;
        } else if (higherScoring.key.slotValue.equals(lowerScoring.key.slotValue)) {
          return true;
        } else if (CollectionUtils.equalIfBothDefined(higherScoring.score, lowerScoring.score)) {
          // Same score; break ties arbitrarily
          return nonlocalFailure(lowerScoring, this.getClass());
        } else {
          return nonlocalFailure(lowerScoring, this.getClass());
        }
      }
      return true;
    }
  }

  /** Don't predict a low weight slot if there's another candidate of reputable weight */
  public static class RemoveLowWeightRelationUnlessOnlyOneOfType extends HeuristicSlotfillPostProcessor.Default {
    @Override
    public boolean pairwiseKeepLowerScoringFill(KBPEntity pivot, KBPSlotFill higherScoring, KBPSlotFill lowerScoring) {
      return !(higherScoring.key.relationName.equals(lowerScoring.key.relationName) &&
          lowerScoring.score.getOrElse(1.0) < 0.01);
    }
  }

  /** Respect the incmpatible relations declared in RelationType */
  public static class RespectDeclaredIncompatibilities extends HeuristicSlotfillPostProcessor.Default {
    @Override
    public boolean pairwiseKeepLowerScoringFill(KBPEntity pivot, KBPSlotFill higherScoring, KBPSlotFill lowerScoring) {
      if (!higherScoring.key.hasKBPRelation() || !lowerScoring.key.hasKBPRelation()) { return true; }
      RelationType candidateRelation = lowerScoring.key.kbpRelation();
      if (higherScoring.key.slotValue.equals(lowerScoring.key.slotValue)) {
        // Case: two slot fills share the same arg2
        RelationType existingRelation = higherScoring.key.kbpRelation();
        if (existingRelation != candidateRelation && !existingRelation.plausiblyCooccursWith(candidateRelation)) {
          logger.debug("" + lowerScoring + " failed " + this.getClass().getSimpleName());
          return nonlocalFailure(lowerScoring, this.getClass());
        }
      }
      return true;
    }
  }

  //
  // Hold One Out Checks
  //

  /**
   * Keep LOC_of_death only of date_of_death exists
   * This is needed because the noise level on LOC_of_death is very high, and
   * the one on date_of_death is low
   */
  public static class MitigateLocOfDeath extends HeuristicSlotfillPostProcessor.Default {
    @Override
    public boolean leaveOneOutKeepHeldOutSlot(KBPEntity pivot, IdentityHashSet<KBPSlotFill> others, KBPSlotFill candidate) {
      //noinspection LoopStatementThatDoesntLoop
      for (RelationType relation : candidate.key.tryKbpRelation()) {
        boolean isLocOfDeath = relation == RelationType.PER_CITY_OF_DEATH ||
            relation == RelationType.PER_COUNTRY_OF_DEATH ||
            relation == RelationType.PER_STATE_OR_PROVINCES_OF_DEATH;
        if (isLocOfDeath) {
          for (KBPSlotFill other : others) {
            if (other.key.tryKbpRelation().equalsOrElse(RelationType.PER_DATE_OF_DEATH, false)) { return true; }
          }
          return nonlocalFailure(candidate, this.getClass());
        } else {
          return true;
        }
      }
      return true;
    }
  }


}
