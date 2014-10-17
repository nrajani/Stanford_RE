package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.ir.StandardIR;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.kbp.slotfilling.ir.PostIRAnnotator;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.time.TimeAnnotations;
import edu.stanford.nlp.time.Timex;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;

import java.io.IOException;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * An interface for writing response files for various formats dictated
 * by different KBP years.
 *
 * @author Gabor Angeli
 */
public abstract class OfficialOutputWriter {

  /**
   * Yield a single line of the score file, with the necessary prefix already appended.
   * For 2009-2013 (at least -- maybe later too) this means the first three columns are already
   * populated.
   * @param builder The StringBuilder to append to
   * @param fill The slot fill to write
   */
  public abstract void yieldLine(StringBuilder builder, KBPSlotFill fill) throws IOException;

  /**
   * Output a set of relations to a file. This is the file to be read by the scoring script.
   * @param os The output stream to write to
   * @param rawRelations The relations to write
   * @param threshold The threshold above which to make a guess.
   */
  public void outputRelations(PrintStream os, String runId, Map<KBPOfficialEntity, Collection<KBPSlotFill>> rawRelations, Map<RelationType, Double> threshold) {
    // the output format specifies that the output has to be sorted by
    // query id
    List<KBPOfficialEntity> entities = new ArrayList<KBPOfficialEntity>();
    entities.addAll(rawRelations.keySet());
    Collections.sort(entities, new KBPOfficialEntity.QueryIdSorter());
    RelationType[] relationTypes = RelationType.values();
    Arrays.sort(relationTypes);

    // now, for each entity mention, output the block of text as expected
    for (KBPOfficialEntity entity : entities) {
      Collection<KBPSlotFill> mentions = rawRelations.get(entity);

      // first, build a map of the relations we know about...
      Map<String, List<KBPSlotFill>> mentionMap = new HashMap<String, List<KBPSlotFill>>();
      for (KBPSlotFill mention : mentions) {
        String relationName = officialRelationName(mention.key.kbpRelation());
        if (!mentionMap.containsKey(relationName)) mentionMap.put(relationName, new ArrayList<KBPSlotFill>());
        mentionMap.get(relationName).add(mention);
      }

      Set<String> relationsOutput = new HashSet<String>();
      for (RelationType relation : relationTypes) {
        // Get the official relation name for this relation (this changes from year to year)
        String officialRelationName = officialRelationName(relation);
        // Get the slot fills
        List<KBPSlotFill> slotFills = mentionMap.get(officialRelationName);

        // Don't output relations of incompatible type
        if (relation.entityType != entity.type) { continue; }
        // Don't output ignored relations
        if (entity.ignoredSlots.orCrash().contains(relation)) { continue; }
        // If we've collapsed multiple relations into one, don't output them again
        if (relationsOutput.contains(officialRelationName)) { continue; }
        relationsOutput.add(officialRelationName);

        // Create a common prefix, which doesn't change across years
        String prefix = entity.queryId.orCrash() + tab() + officialRelationName + tab() + runId + tab();
        if (slotFills == null || slotFills.size() == 0) {
          os.println(prefix + "NIL");
        } else {
          for (KBPSlotFill mention : slotFills) {
            // For each relation...
            if( !threshold.containsKey(mention.key.kbpRelation()) ||
                mention.score.orCrash() >= threshold.get(mention.key.kbpRelation())) {
              // ... if it's over the minimum threshold
              StringBuilder line = new StringBuilder(prefix);
              try {
                // ... append it to the line
                yieldLine(line, mention);
               
              } catch (IOException e) {
                throw new RuntimeException(e);
              }
              // Write line of output
              os.println(line.toString().trim());
            }
          }
        }
      }
    }
  }

  /**
   * The top level call for creating a slotfilling validation file from an input PrintStream
   * @param os The output target to write to.
   * @param originalFills The original set of slot fills corresponding to system guesses.
   * @param validSlotFills The set of valid slot fills, as judged by our slot fill validation.
   */
  public void outputValidSlotsForEntity(PrintStream os,
                                        List<KBPSlotValidator.KBPSlotFillQuery> originalFills,
                                        Set<KBPSlotFill> validSlotFills) {
    Iterator<KBPSlotValidator.KBPSlotFillQuery> candidates = originalFills.iterator();
    while (candidates.hasNext()) {
      KBPSlotValidator.KBPSlotFillQuery cand = candidates.next();
      os.print(cand.validationQueryID);
      os.print("\t");
      os.print(validSlotFills.contains(cand.fill) ? 1 : -1);
      if (candidates.hasNext()) os.print("\n");
    }
  }

  /**
   * Output the official relation name for the current year's KBP.
   *
   * @see OfficialOutputWriter#officialRelationName(RelationType, Props.YEAR)
   * @see Props#KBP_YEAR
   */
  public static String officialRelationName(RelationType relation) {
    return officialRelationName(relation, Props.KBP_YEAR);
  }

  /**
   * Convert a canonical KBP Relation to the string form expected for a particular year.
   * This handles things like minor lexical variations, and merging relation types together.
   * @param relation The relation to convert to an official string
   * @return The official string form of the relation
   */
  public static String officialRelationName(RelationType relation, Props.YEAR kbp_year) {
    String name = relation.canonicalName.replaceAll("SLASH", "/");
    switch (kbp_year) {
      case KBP2009:
        switch (relation) {
          case PER_CITIES_OF_RESIDENCE: name = "per:cities_of_residences"; break;
          case PER_STATE_OR_PROVINCES_OF_RESIDENCE: name = "per:stateorprovinces_of_residences"; break;
          case PER_COUNTRIES_OF_RESIDENCE: name = "per:countries_of_residences"; break;
          case ORG_CITY_OF_HEADQUARTERS:
          case ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS:
          case ORG_COUNTRY_OF_HEADQUARTERS:
            name = "org:headquarters";
            break;
          case PER_CITY_OF_BIRTH:
          case PER_STATE_OR_PROVINCES_OF_BIRTH:
          case PER_COUNTRY_OF_BIRTH:
            name = "per:place_of_birth";
            break;
        }
        break;
      case KBP2010: break;
      case KBP2011:
        switch (relation) {
          case ORG_POLITICAL_RELIGIOUS_AFFILIATION: name = "org:political,religious_affiliation"; break;
          case ORG_TOP_MEMBERS_SLASH_EMPLOYEES: name = "org:top_members,employees"; break;
          case ORG_NUMBER_OF_EMPLOYEES_SLASH_MEMBERS: name = "org:number_of_employees,members"; break;
        }
        break;
      case KBP2012:
        switch (relation) {
          case PER_STATE_OR_PROVINCES_OF_RESIDENCE: name = "per:statesorprovinces_of_residence"; break;
          case ORG_POLITICAL_RELIGIOUS_AFFILIATION: name = "org:political_religious_affiliation"; break;
          case ORG_TOP_MEMBERS_SLASH_EMPLOYEES: name = "org:top_members_employees"; break;
          case ORG_NUMBER_OF_EMPLOYEES_SLASH_MEMBERS: name = "org:number_of_employees_members"; break;
          case ORG_FOUNDED: name = "org:date_founded"; break;
          case ORG_DISSOLVED: name = "org:date_dissolved"; break;
        }
        break;
      case KBP2013:
        switch (relation) {
          case PER_EMPLOYEE_OF: name = "per:employee_or_member_of"; break;
          case PER_MEMBER_OF: name = "per:employee_or_member_of"; break;
          case PER_STATE_OR_PROVINCES_OF_RESIDENCE: name = "per:statesorprovinces_of_residence"; break;
          case ORG_POLITICAL_RELIGIOUS_AFFILIATION: name = "org:political_religious_affiliation"; break;
          case ORG_TOP_MEMBERS_SLASH_EMPLOYEES: name = "org:top_members_employees"; break;
          case ORG_NUMBER_OF_EMPLOYEES_SLASH_MEMBERS: name = "org:number_of_employees_members"; break;
          case ORG_FOUNDED: name = "org:date_founded"; break;
          case ORG_DISSOLVED: name = "org:date_dissolved"; break;
        }
        break;
      default:
        throw new IllegalStateException("Unknown year: " + Props.KBP_YEAR);
    }
    return name;
  }

  /**
   * Escape the slot value, which may contain characters which would corrupt the
   * output score file
   * @param raw The raw slot value
   * @return A slot value that is valid to add to the output score file
   */
  protected String escapeSlotValue(String raw) {
    return raw.replaceAll("\n", " ").replaceAll("\t", " ").replaceAll("\\s+", " ");
  }

  /**
   * Format a list of spans into a valid field for the output file.
   * @param spans The spans to format. The maximum number of spans is 2.
   *              These are character offset spans.
   * @return A valid String to be printed in the
   */
  protected String spansToOffsetString(Span... spans) {
    if (spans.length > 2) {
      throw new IllegalArgumentException("Too many spans for offset");
    }
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < spans.length; ++i) {
      b.append(spans[i].start()).append("-").append(spans[i].end());
      if (i < spans.length - 1) { b.append(","); }
    }
    return b.toString();
  }

  /** The character to use to separate fields of the score file */
  protected String tab() {
    switch (Props.KBP_YEAR) {
      case KBP2009:
      case KBP2010:
      case KBP2011:
      case KBP2012:
//        return " ";
      case KBP2013:
        return "\t";
      default: throw new IllegalStateException("Unknown year: " + Props.KBP_YEAR);
    }
  }

  /**
   * A writer to output the official response files for 2009
   */
  public static class OfficialOutputWriter2009 extends OfficialOutputWriter2010 { }

  /**
   * A writer to output the official response files for 2010
   */
  public static class OfficialOutputWriter2010 extends OfficialOutputWriter {
    @Override
    public void yieldLine(StringBuilder builder, KBPSlotFill fill) {
      if (fill.provenance.isDefined()) {
        builder.append(fill.provenance.get().docId).append(tab()).append(escapeSlotValue(fill.key.slotValue));
      } else {
        builder.append("UNKDOC").append(tab()).append(escapeSlotValue(fill.key.slotValue));

      }
    }
  }

  /**
   * A writer to output the official response files for 2011
   */
  public static class OfficialOutputWriter2011 extends OfficialOutputWriter2010 { }

  /**
   * A writer to output the official response files for 2012
   */
  public static class OfficialOutputWriter2012 extends OfficialOutputWriter2011 { }

  /**
   * A writer to output the official response files for 2013
   */
  public static class OfficialOutputWriter2013 extends OfficialOutputWriter {
    private static DecimalFormat df = new DecimalFormat("0.000");
    public final StandardIR ir;


    public OfficialOutputWriter2013(StandardIR ir) {
      this.ir = ir;
    }

    /** Offsets for a particular well-defined span */
    private Span characterOffset(List<CoreLabel> tokens, Span tokenOffset) {
      int start = tokens.get(tokenOffset.start()).get(CoreAnnotations.CharacterOffsetBeginAnnotation.class);
      int end = tokens.get(tokenOffset.end() - 1).get(CoreAnnotations.CharacterOffsetEndAnnotation.class);
      return new Span(start, end);

    }

    /** Offsets for a particular well-defined span */
    private Span characterOffset(CoreMap sentence, Span tokenOffset) {
      return characterOffset(sentence.get(CoreAnnotations.TokensAnnotation.class), tokenOffset);
    }

    /** Offsets for an entire provenance span */
    private Span characterOffset(CoreMap sentence, KBPRelationProvenance provenance,
                                 Span entityOffset, Span slotOffset) {
      int tokenStart = min(provenance.entityMentionInSentence.orCrash().start(),
                           provenance.slotValueMentionInSentence.orCrash().start());
      int tokenEnd = max(provenance.entityMentionInSentence.orCrash().end(),
                         provenance.slotValueMentionInSentence.orCrash().end());
      Span expectedCharacterOffsets = characterOffset(sentence, new Span(tokenStart, tokenEnd));
      // Make sure that the entity and slot value are completely subsumed by the sentence provenance
      return new Span(
          min(expectedCharacterOffsets.start(), entityOffset.start(), slotOffset.start()),
          max(expectedCharacterOffsets.end(), entityOffset.end(), slotOffset.end()));
    }

    /** Why this isn't the default min is beyond me */
    private int min(int... args) {
      int minSoFar = Integer.MAX_VALUE;
      for (int arg : args) { minSoFar = minSoFar < arg ? minSoFar : arg; }
      return minSoFar;
    }

    /** Why this isn't the default max is beyond me */
    private int max(int... args) {
      int maxSoFar = Integer.MIN_VALUE;
      for (int arg : args) { maxSoFar = maxSoFar > arg ? maxSoFar : arg; }
      return maxSoFar;
    }

    /**
     * Finds a literal string in the source document, to resolve coref.
     * A preference is given to respecting existing coref chains, but eventually
     * the method will back off to finding the first case-insensitive match
     * for the name in the document.
     *
     * @param doc The document to search within
     * @param key The key to annotate with respect to
     * @param entity True if we are finding the literal entity mention; false if we are finding the literal slot value mention
     * @return A character offset span for the antecedent
     */
    private Maybe<Span> resolveLiteralCoref(Annotation doc, KBTriple key, boolean entity) {
      new PostIRAnnotator(key.entityName, Maybe.Just(key.entityType.name), Maybe.Just(key.slotValue),
          key.slotType.isDefined() ? Maybe.Just(key.slotType.get().name) : Maybe.<String>Nothing(), true).annotate(doc);
      if (entity) {
        Pair<Integer, Span> spanInfo = doc.get(KBPAnnotations.CanonicalEntitySpanAnnotation.class);
        if (spanInfo != null) {
          return Maybe.Just(characterOffset(doc.get(CoreAnnotations.SentencesAnnotation.class).get(spanInfo.first), spanInfo.second));
        } else {
          return Maybe.Nothing();
        }
      } else {
        // Return regular antecedent
        Pair<Integer, Span> spanInfo = doc.get(KBPAnnotations.CanonicalSlotValueSpanAnnotation.class);
        if (spanInfo != null) {
          return Maybe.Just(characterOffset(doc.get(CoreAnnotations.SentencesAnnotation.class).get(spanInfo.first), spanInfo.second));
        } else {
          return Maybe.Nothing();
        }
      }
    }

    /** A simple regular expression for matching a well-formed year */
    public static final Pattern YEAR = Pattern.compile("[12]\\d\\d\\d");
    public static final Pattern DATETIME = Pattern.compile("<DATETIME>\\s*([^<]+)\\s*</DATETIME>");

    private Span resolveDocDate(Annotation doc, Span defaultReturnValue) {
      // Try to find a literal datetime
      String rawText = doc.get(CoreAnnotations.TextAnnotation.class);
      Matcher datetimeMatcher = DATETIME.matcher(rawText);
      if (datetimeMatcher.find()) {
        String datetimeString = datetimeMatcher.group(1).trim();
        return new Span(rawText.indexOf(datetimeString), rawText.indexOf(datetimeString) + datetimeString.length());
      }
      // Try to glean the datetime from the doc id

      // Scan for the doc date in the tokens
      List<CoreLabel> tokens = doc.get(CoreAnnotations.TokensAnnotation.class);
      for (int start = 0; start < tokens.size(); ++start) {
        CoreLabel token = tokens.get(start);
        if (token.containsKey(TimeAnnotations.TimexAnnotation.class)) {
          // Get time chunk
          Timex timex = token.get(TimeAnnotations.TimexAnnotation.class);
          int end = start + 1;
          while(Maybe.fromNull(tokens.get(end).get(TimeAnnotations.TimexAnnotation.class)).equalsOrElse(timex, false)) {
            end += 1;
          }
          // Check chunk
          Span candidateTokenSpan = new Span(start, end);
          if (YEAR.matcher(CoreMapUtils.sentenceSpanString(tokens, candidateTokenSpan)).find()) {
            return characterOffset(tokens, candidateTokenSpan);
          }
          start = end;
        }
      }
      return defaultReturnValue;
    }

    @Override
    public void yieldLine(StringBuilder builder, KBPSlotFill fill) throws IOException {
      // -- Variables
      // Get Provenances
      if (!fill.provenance.isDefined()) { throw new IllegalArgumentException("No provenance for slot fill: " + fill); }
      KBPRelationProvenance primaryProvenance = fill.provenance.get();
      Maybe<KBPRelationProvenance> secondaryProvenance = Maybe.Nothing(); // TODO(gabor) multipe provenances
      if(primaryProvenance==null){
          System.out.println("provenance is null "+builder);
          if (fill.provenance.isDefined()){
        	  System.out.println("provenance is defined");
          }
      }
      //System.out.println("yes");

      // Get Sentences
     /* Annotation doc =  null;//ir.officialIndex.fetchDocument(primaryProvenance.docId);
      CoreMap primarySentence = doc.get(CoreAnnotations.SentencesAnnotation.class).get(primaryProvenance.sentenceIndex.get());
      Maybe<CoreMap> secondarySentence = Maybe.Nothing();
      if (secondaryProvenance.isDefined()) {
        secondarySentence = Maybe.Just(doc.get(CoreAnnotations.SentencesAnnotation.class).get(primaryProvenance.sentenceIndex.get()));
      }
     */
      // -- Output

      // Column 4: DocID
      builder.append(primaryProvenance.docId).append(tab());

      // Column 5: Slot Fill
      builder.append(escapeSlotValue(fill.key.slotValue)).append(tab());

      // Column 6: Value Provenance
//      Span primaryValueOffset = characterOffset(primarySentence, primaryProvenance.slotValueMentionInSentence.orCrash());
        Span primaryValueOffset =  primaryProvenance.slotValueMentionInSentence.get();
      // resolve coref
      Span valueAntecedent = primaryValueOffset;
      builder.append(spansToOffsetString(primaryValueOffset));
      /*
      if (!CoreMapUtils.sentenceSpanString(primarySentence, primaryProvenance.slotValueMentionInSentence.orCrash()).equalsIgnoreCase(fill.key.slotValue)) {
        valueAntecedent = resolveLiteralCoref(doc, fill.key, false).getOrElse(primaryValueOffset);
      }
      if (Span.overlaps(primaryValueOffset, valueAntecedent)) {
        primaryValueOffset = valueAntecedent;
      }
      // resolve dates
      Span docDate = primaryValueOffset;
      if (fill.key.kbpRelation().isDateRelation() &&
          !YEAR.matcher(CoreMapUtils.sentenceSpanString(primarySentence, primaryProvenance.slotValueMentionInSentence.orCrash())).find()) {
        docDate = resolveDocDate(doc, primaryValueOffset);
      }
      if (Span.overlaps(primaryValueOffset, docDate)) {
        docDate = valueAntecedent;
      }
      // write slot
      if (!primaryValueOffset.equals(valueAntecedent)) {
        // case: coreferent
        builder.append(spansToOffsetString(primaryValueOffset, valueAntecedent));
      } else if (!primaryValueOffset.equals(docDate)) {
        builder.append(spansToOffsetString(primaryValueOffset, docDate));
      } else if (secondaryProvenance.isDefined() && secondarySentence.isDefined()) {
        // case: multiple provenances
        builder.append(spansToOffsetString(
            primaryValueOffset,
            characterOffset(secondarySentence.get(), secondaryProvenance.get().slotValueMentionInSentence.orCrash())
        ));
      } else {
        // case: single provenance
        builder.append(spansToOffsetString(primaryValueOffset));
      }
      */
      builder.append(tab());

      // Column 7: Entity Provenance
      //Span primaryEntityOffset = characterOffset(primarySentence, primaryProvenance.entityMentionInSentence.orCrash());
      Span primaryEntityOffset = primaryProvenance.entityMentionInSentence.get();
      builder.append(spansToOffsetString(primaryEntityOffset));
      /*
      // resolve coref
      Span entityAntecedent = primaryEntityOffset;
      if (!CoreMapUtils.sentenceSpanString(primarySentence, primaryProvenance.entityMentionInSentence.orCrash()).equalsIgnoreCase(fill.key.entityName)) {
        entityAntecedent = resolveLiteralCoref(doc, fill.key, true).getOrElse(entityAntecedent);
      }
      if (Span.overlaps(primaryEntityOffset, entityAntecedent)) {
        primaryEntityOffset = entityAntecedent;
      }
      // write slot
      if (!primaryEntityOffset.equals(entityAntecedent)) {
        // case: coreferent
        builder.append(spansToOffsetString(primaryEntityOffset, entityAntecedent));
      } else if (secondaryProvenance.isDefined() && secondarySentence.isDefined()) {
        // case: multiple provenances
        builder.append(spansToOffsetString(
            primaryEntityOffset,
            characterOffset(secondarySentence.get(), secondaryProvenance.get().entityMentionInSentence.orCrash())
        ));
      } else {
        // case: single provenance
        builder.append(spansToOffsetString(primaryEntityOffset));
      }
      */
      builder.append(tab());

      // Column 8: Justification Provenance
      if (fill.key.kbpRelation() == RelationType.ORG_ALTERNATE_NAMES || fill.key.kbpRelation() == RelationType.PER_ALTERNATE_NAMES) {
        builder.append(tab());  // We don't need to give provenance -- let's not shoot ourselves in the foot.
      } else {
        //builder.append(spansToOffsetString(characterOffset(primarySentence, primaryProvenance,primaryEntityOffset, primaryValueOffset))).append(tab());
    	  Span justOffset = primaryProvenance.justificationMention.get();
    	  builder.append(spansToOffsetString(justOffset));
    	  builder.append(tab());
      }
      // Column 9: Confidence Score
      double score = fill.score.getOrElse(0.5);
      if (score > 1.0 || score < 0.0) score = 1.0 / (1.0 + Math.exp( -score ));  // sigmoid score if out of bounds
      assert score <= 1.0 && score >= 0.0;
      builder.append(df.format(score));
    }
  }


}
