package edu.stanford.nlp.kbp.slotfilling.common;

import edu.stanford.nlp.ie.NumberNormalizer;
import edu.stanford.nlp.ie.machinereading.structure.EntityMention;
import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.kbp.slotfilling.evaluate.WorldKnowledgePostProcessor;
import edu.stanford.nlp.ling.CoreAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.*;

import java.lang.reflect.Array;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Pattern;

@SuppressWarnings("UnusedDeclaration")
public class Utils {


  private Utils() {} // static methods

  public static WorldKnowledgePostProcessor geography() { return WorldKnowledgePostProcessor.singleton(Props.TEST_CONSISTENCY_WORLDKNOWLEDGE_DIR); }


  public static String makeNERTag(NERTag et) {
    return ("ENT:" + et.name);
  }

  public static Maybe<NERTag> getNERTag(EntityMention e) {
    String typeString = e.getType();
    if (typeString == null) { typeString = e.getMentionType(); }
    if (typeString == null) { typeString = e.getSubType(); }
    if( typeString != null && typeString.length() > 0 ) {
        if (typeString.equals("ENT:PERSON"))  
          return Maybe.Just(NERTag.PERSON);
        else if (typeString.equals("ENT:ORGANIZATION"))  
          return Maybe.Just(NERTag.ORGANIZATION);
        else
          return NERTag.fromString(typeString);
    } else {
      return Maybe.Nothing();
    }
  }


  public static Maybe<String> getKbpId(EntityMention mention) {
    String id = mention.getObjectId();
    int kbpPos = id.indexOf("KBP");
    if( kbpPos >= 0 )
      return Maybe.Just( id.substring(kbpPos + "KBP".length()) );
    else
      return Maybe.Nothing();
  }

  public static KBPEntity getKbpEntity(EntityMention mention) {
    return KBPNew.entName(mention.getFullValue())
                 .entType(getNERTag(mention).orCrash())
                 .entId(getKbpId(mention))
                 .KBPEntity();
  }

  static Pattern escaper = Pattern.compile("([^a-zA-z0-9])"); // should also include ,:; etc.?
  /**
   * Builds a new string where characters that have special meaning in Java regexes are escaped
   */
  public static String escapeSpecialRegexCharacters(String s) {
    return escaper.matcher(s).replaceAll("\\\\$1");
  }

  public static String getMemoryUsage() {
    Runtime rt = Runtime.getRuntime();
    long total = rt.totalMemory() / (1024*1024);
    long free = rt.freeMemory() / (1024*1024);
    return String.format( "Used: %d MB, Free: %d MB, Total: %d MB", total - free, free, total );
  }


  public static Maybe<NERTag> inferFillType(RelationType relation) {
    switch (relation) {
      case PER_ALTERNATE_NAMES:
        return  Maybe.Just(NERTag.PERSON);
      case PER_CHILDREN:
        return  Maybe.Just(NERTag.PERSON);
      case PER_CITIES_OF_RESIDENCE:
        return  Maybe.Just(NERTag.CITY);
      case PER_CITY_OF_BIRTH:
        return  Maybe.Just(NERTag.CITY);
      case PER_CITY_OF_DEATH:
        return  Maybe.Just(NERTag.CITY);
      case PER_COUNTRIES_OF_RESIDENCE:
        return  Maybe.Just(NERTag.COUNTRY);
      case PER_COUNTRY_OF_BIRTH:
        return  Maybe.Just(NERTag.COUNTRY);
      case PER_COUNTRY_OF_DEATH:
        return  Maybe.Just(NERTag.COUNTRY);
      case PER_EMPLOYEE_OF:
        return  Maybe.Just(NERTag.ORGANIZATION);
      case PER_MEMBER_OF:
        return  Maybe.Just(NERTag.ORGANIZATION);
      case PER_ORIGIN:
        return  Maybe.Just(NERTag.NATIONALITY);
      case PER_OTHER_FAMILY:
        return  Maybe.Just(NERTag.PERSON);
      case PER_PARENTS:
        return  Maybe.Just(NERTag.PERSON);
      case PER_SCHOOLS_ATTENDED:
        return  Maybe.Just(NERTag.ORGANIZATION);
      case PER_SIBLINGS:
        return  Maybe.Just(NERTag.PERSON);
      case PER_SPOUSE:
        return  Maybe.Just(NERTag.PERSON);
      case PER_STATE_OR_PROVINCES_OF_BIRTH:
        return  Maybe.Just(NERTag.STATE_OR_PROVINCE);
      case PER_STATE_OR_PROVINCES_OF_DEATH:
        return  Maybe.Just(NERTag.STATE_OR_PROVINCE);
      case PER_STATE_OR_PROVINCES_OF_RESIDENCE:
        return  Maybe.Just(NERTag.STATE_OR_PROVINCE);
      case PER_AGE:
        return  Maybe.Just(NERTag.NUMBER);
      case PER_DATE_OF_BIRTH:
        return  Maybe.Just(NERTag.DATE);
      case PER_DATE_OF_DEATH:
        return  Maybe.Just(NERTag.DATE);
      case PER_CAUSE_OF_DEATH:
        return  Maybe.Just(NERTag.CAUSE_OF_DEATH);
      case PER_CHARGES:
        return  Maybe.Just(NERTag.CRIMINAL_CHARGE);
      case PER_RELIGION:
        return  Maybe.Just(NERTag.RELIGION);
      case PER_TITLE:
        return  Maybe.Just(NERTag.TITLE);
      case ORG_ALTERNATE_NAMES:
        return  Maybe.Just(NERTag.ORGANIZATION);
      case ORG_CITY_OF_HEADQUARTERS:
        return  Maybe.Just(NERTag.CITY);
      case ORG_COUNTRY_OF_HEADQUARTERS:
        return  Maybe.Just(NERTag.COUNTRY);
      case ORG_FOUNDED_BY:
        return  Maybe.Just(NERTag.PERSON);
      case ORG_MEMBER_OF:
        return  Maybe.Just(NERTag.ORGANIZATION);
      case ORG_MEMBERS:
        return  Maybe.Just(NERTag.PERSON);
      case ORG_PARENTS:
        return  Maybe.Just(NERTag.ORGANIZATION);
      case ORG_POLITICAL_RELIGIOUS_AFFILIATION:
        return  Maybe.Just(NERTag.RELIGION);
      case ORG_SHAREHOLDERS:
        return  Maybe.Just(NERTag.PERSON);
      case ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS:
        return  Maybe.Just(NERTag.STATE_OR_PROVINCE);
      case ORG_SUBSIDIARIES:
        return  Maybe.Just(NERTag.ORGANIZATION);
      case ORG_TOP_MEMBERS_SLASH_EMPLOYEES:
        return  Maybe.Just(NERTag.PERSON);
      case ORG_DISSOLVED:
        return  Maybe.Just(NERTag.DATE);
      case ORG_FOUNDED:
        return  Maybe.Just(NERTag.DATE);
      case ORG_NUMBER_OF_EMPLOYEES_SLASH_MEMBERS:
        return  Maybe.Just(NERTag.NUMBER);
      case ORG_WEBSITE:
        return Maybe.Just(NERTag.URL);
    }
    return Maybe.Nothing();
  }

  /**
   * Concatenate two arrays; also, a great example of Java Magic canceling out Java Failure.
   */
  public static <E> E[] concat(E[] a, E[] b) {
    @SuppressWarnings("unchecked") E[] rtn = (E[]) Array.newInstance(a.getClass().getComponentType(), a.length + b.length);
    System.arraycopy(a, 0, rtn, 0, a.length);
    System.arraycopy(b, 0, rtn, a.length, b.length);
    return rtn;
  }


  @SuppressWarnings({"ConstantConditions", "AssertWithSideEffects", "UnusedAssignment"})
  public static boolean assertionsEnabled() {
    boolean assertionsEnabled = false;
    assert assertionsEnabled = true;
    return assertionsEnabled;
  }


  public static boolean sameSlotFill(String candidate, String gold) {
    // Canonicalize strings
    candidate = candidate.trim().toLowerCase();
    gold = gold.trim().toLowerCase();
    // Special cases
    if (candidate == null || gold == null || candidate.trim().equals("")) {
      return false;
    }

    // Simple equality
    if (candidate.equals(gold)) {
      return true;
    }

    // Containment
    if (candidate.contains(gold) || gold.contains(candidate)) {
      return true;
    }

    // Else, give up
    return false;
  }


  /** I shamefully stole this from: http://rosettacode.org/wiki/Levenshtein_distance#Java --Gabor */
  public static int levenshteinDistance(String s1, String s2) {
    s1 = s1.toLowerCase();
    s2 = s2.toLowerCase();

    int[] costs = new int[s2.length() + 1];
    for (int i = 0; i <= s1.length(); i++) {
      int lastValue = i;
      for (int j = 0; j <= s2.length(); j++) {
        if (i == 0)
          costs[j] = j;
        else {
          if (j > 0) {
            int newValue = costs[j - 1];
            if (s1.charAt(i - 1) != s2.charAt(j - 1))
              newValue = Math.min(Math.min(newValue, lastValue), costs[j]) + 1;
            costs[j - 1] = lastValue;
            lastValue = newValue;
          }
        }
      }
      if (i > 0)
        costs[s2.length()] = lastValue;
    }
    return costs[s2.length()];
  }

  public static Iterator<String> randomInsults() { return randomInsults(42); }

  /** http://iome.me/fz/need%20a%20random%20sentence?/why%20not%20zoidberg? */
  public static Iterator<String> randomInsults(final int seed) {
    final HashSet<String> seen = new HashSet<String>();
    final String[] adj1 = {"warped", "babbling", "madcap", "whining", "wretched", "loggerheaded", "threadbare", "foul", "artless",
        "artless", "baudy", "beslumbering", "bootless", "churlish", "clouted", "craven", "dankish", "dissembling", "droning",
        "errant", "fawning", "fobbing", "froward", "frothy", "gleeking", "goatish", "gorbellied", "impertinent", "infectious",
        "jarring", "loggerheaded", "lumpish", "mammering", "mangled", "mewling", "paunchy", "pribbling", "puking", "puny",
        "qualling", "rank", "reeky", "roguish", "rutting", "saucy", "spleeny", "spongy", "surly", "tottering", "unmuzzled", "vain",
        "venomed", "villainous", "warped", "wayward", "weedy", "yeasty" };
    final String[] adj2 = {"toad-spotted", "guts-griping", "beef-witted", "ill-favored", "hare-brained", "fat-kidneyed", "white-bearded",
        "shrill-voiced", "base-court", "bat-fowling", "beef-witted", "beetle-headed", "boil-brained", "clapper-clawed",
        "clay-brained", "common-kissing", "crook-pated", "dismal-dreaming", "dizzy-eyed", "doghearted", "dread-bolted", "earth-vexing",
        "elf-skinned", "fat-kidneyed", "fen-sucked", "flap-mouthed", "fly-bitten", "folly-fallen", "fool-born", "full-gorged",
        "guts-griping", "half-faced", "hasty-witted", "hedge-born", "hell-hated", "idle-headed", "ill-breeding", "ill-nurtured",
        "knotty-pated", "milk-livered", "motley-minded", "onion-eyed", "plume-plucked", "pottle-deep", "pox-marked", "reeling-ripe",
        "rough-hewn", "rude-growing", "rump-fed", "shard-borne", "sheep-biting", "spur-galled", "swag-bellied", "tardy-gaited",
        "tickle-brained", "toad-spotted", "unchin-snouted", "weather-bitten" };
    final String[] nouns = {"jolt-head", "mountain-goat", "fat-belly", "malt-worm", "minnow", "so-and-so", "maggot-pie", "foot-licker", "land-fish",
        "apple-john", "baggage", "barnacle", "bladder", "boar-pig", "bugbear", "bum-bailey", "canker-blossom", "clack-dish", "clotpole",
        "coxcomb", "codpiece", "death-token", "dewberry", "flap-dragon", "flax-wench", "flirt-gill", "foot-licker", "fustilarian",
        "giglet", "gudgeon", "haggard", "harpy", "hedge-pig", "horn-beast", "hugger-mugger", "joithead", "lewdster", "lout", "maggot-pie",
        "malt-worm", "mammet", "measle", "minnow", "miscreant", "moldwarp", "mumble-news", "nut-hook", "pigeon-egg", "pignut",
        "puttock", "pumpion", "ratsbane", "scut", "skainsmate", "strumpet", "varlot", "vassal", "whey-face", "wagtail", };
    return new Iterator<String> () {
      private Random random = new Random(seed);
      @Override
      public boolean hasNext() {
        return true;
      }
      @Override
      public String next() {
        String a1 = adj1[random.nextInt(adj1.length)];
        String a2 = adj2[random.nextInt(adj2.length)];
        String n  = nouns[random.nextInt(nouns.length)];
        String candidate = "Curse thee, KBP, thou " + a1 + " " + a2 + " " + n;
        if (seen.contains(candidate)) { return next(); } else { seen.add(candidate); return candidate; }
      }
      @Override
      public void remove() {
      }
    };
  }


  public static Maybe<Double> noisyOr(Maybe<Double> score1, Maybe<Double> score2) {
    if( score1.isNothing() ) return score2;
    else if( score2.isNothing() ) return score1;
    else return Maybe.Just( 1 - (1 - score1.get()) * (1-score2.get()) );
  }



  private static final AtomicInteger entityMentionCount = new AtomicInteger(0);
  public static String makeEntityMentionId(Maybe<String> kbpId) {
    String id = "EM" + entityMentionCount.incrementAndGet();
    for (String idImpl : kbpId) { id += "-KBP" + idImpl; }
    return id;
  }

  /** Determine if a slot is close enough to any entity to be considered a valid candidate */
  public static boolean closeEnough(Span slotSpan, Collection<Span> entitySpans) {
    if (entitySpans.isEmpty()) { return true; }
    for (Span entitySpan : entitySpans) {
      if (slotSpan.end() <= entitySpan.start()
          && entitySpan.start() - slotSpan.end() < Props.MAX_DISTANCE_BETWEEN_ENTITY_AND_SLOT) {
        return true;
      } else if (entitySpan.end() <= slotSpan.start()
          && slotSpan.start() - entitySpan.end() < Props.MAX_DISTANCE_BETWEEN_ENTITY_AND_SLOT) {
        return true;
      }
    }
    return false;
  }

  public static String noSpecialChars(String original) {
    return original
        .replaceAll("\\\\", "")
        .replaceAll("\"", "")
        .replaceAll("-", " ")
        .toLowerCase()
        ;
  }
  
  public static boolean nearExactEntityMatch( String higherGloss, String lowerGloss ) {
    // case: slots have same relation, and that relation isn't an alternate name
    // Filter case sensitive match
    if (higherGloss.equalsIgnoreCase(lowerGloss)) { return true; }
    // Ignore certain characters
    else if (noSpecialChars(higherGloss).equalsIgnoreCase(noSpecialChars(lowerGloss))) { return true; }
    return false;
  }


  public static boolean approximateEntityMatch( KBPEntity entity, KBPEntity otherEntity ) {
    return Props.KBP_ENTITYLINKER.sameEntity( new EntityContext(entity), new EntityContext(otherEntity) );
  }

  private static String removeDisallowedAlternateNameVariants(String in) {
    return in.toLowerCase().replaceAll("corp.?", "").replaceAll("llc.?", "").replaceAll("inc.?", "")
        .replaceAll("\\s+", " ").trim();
  }

  public static boolean isValidAlternateName(String name1, String name2) {
    return !removeDisallowedAlternateNameVariants(name1).equals(removeDisallowedAlternateNameVariants(name2));

  }

  public static Maybe<Integer> getNumericValue( KBPSlotFill candidate ) {
    // Case: Rewrite mentions to their antecedents (person and organization)
    // If we already have provenance...
    if (candidate.provenance.isDefined() && candidate.provenance.get().containingSentenceLossy.isDefined() &&
            candidate.provenance.get().slotValueMentionInSentence.isDefined() ) {
      CoreMap lossySentence = candidate.provenance.get().containingSentenceLossy.get();
      Span slotSpan = candidate.provenance.get().slotValueMentionInSentence.get();
      List<CoreLabel> tokens = lossySentence.get(CoreAnnotations.TokensAnnotation.class);
      Maybe<List<CoreLabel>> provenanceSpan = Maybe.Just(tokens.subList(slotSpan.start(), slotSpan.end()));

      for (List<CoreLabel> valueSpan : provenanceSpan) {
        for (CoreLabel token : valueSpan) {
          if (token.containsKey(CoreAnnotations.NumericValueAnnotation.class)) {
            return Maybe.Just(token.get(CoreAnnotations.NumericValueAnnotation.class).intValue());
          }
        }
      }
    }
    // I'm trying my best here
    return Maybe.Just(NumberNormalizer.wordToNumber(candidate.key.slotValue).intValue());
    // TODO(arun) Catch the case where this isn't a number
  }

  /**
   * Tries very hard to match a given sentence fragment to its original sentence.
   * Whitespace is ignored on both the sentence tokenization and the gloss.
   * @param sentence The sentence to match against. The returned span will be in token offsets into this sentence
   * @param gloss The string to fit into the sentence somewhere.
   * @param guess An optional span that denotes where we think this gloss should go -- the closest matching gloss
   *              to this is returned.
   * @return Our best guess of where the gloss came from in the original sentence, or {@link Maybe.Nothing} if we couldn't find anything.
   */
  public static Maybe<Span> getTokenSpan(char[][] sentence, char[] gloss, Maybe<Span> guess) {
    // State (think finite state machine with multiple "heads" tracking progress)
    boolean[] heads = new boolean[gloss.length];
    int[] starts = new int[gloss.length];
    // Matches
    List<Span> finishedSpans = new ArrayList<Span>();
    // Initialize State
    Arrays.fill(heads, false);

    // Run FSA Matcher
    for (int tokI = 0; tokI < sentence.length; ++tokI) {
      for (int charI = 0; charI < sentence[tokI].length; ++charI) {
        // Initialize
        starts[0] = tokI;
        heads[0] = true;

        // (1) Scroll through whitespace
        for (int onPriceI = 1; onPriceI < gloss.length; ++onPriceI) {
          if (heads[onPriceI - 1] && Character.isWhitespace(gloss[onPriceI - 1])) {
            heads[onPriceI] = true;
            starts[onPriceI] = starts[onPriceI - 1];
          }
        }
        // (2) Check if we've whitespace'd to the end
        if (heads[gloss.length - 1] && Character.isWhitespace(gloss[gloss.length - 1])) {
          assert starts[gloss.length - 1] >= 0;
          finishedSpans.add(new Span(starts[gloss.length - 1], charI == 0 ? tokI : tokI + 1));
        }

        // (3) try for an exact match
        for (int onPriceI = gloss.length - 1; onPriceI >= 0; --onPriceI) {
          if (heads[onPriceI] || onPriceI == 0) {
            // Case: found an active partial match to potentially extend
            if (sentence[tokI][charI] == gloss[onPriceI]) {
              // Case: literal match
              if (onPriceI >= gloss.length - 1) {
                // Finished match
                assert starts[onPriceI] >= 0;
                finishedSpans.add(new Span(starts[onPriceI], tokI + 1));
              } else {
                // Move head
                heads[onPriceI + 1] = true;
                starts[onPriceI + 1] = starts[onPriceI];
              }
              if (!Character.isWhitespace(sentence[tokI][charI])) {
                // Either we matched whitespace, or invalidate this position
                heads[onPriceI] = false;
                starts[onPriceI] = -1;
              }
            }
          }
        }

        // (4) Scroll through whitespace (and potentially kill it!)
        for (int onPriceI = 1; onPriceI < gloss.length; ++onPriceI) {
          if (heads[onPriceI - 1] && Character.isWhitespace(gloss[onPriceI - 1])) {
            heads[onPriceI] = true;
            starts[onPriceI] = starts[onPriceI - 1];
          } else if (!heads[onPriceI - 1] && Character.isWhitespace(gloss[onPriceI - 1]) &&
              !Character.isWhitespace(sentence[tokI][charI])) {
            heads[onPriceI] = false;
            starts[onPriceI] = -1;
          }
        }
        // (5) Check if we've whitespace'd to the end
        if (heads[gloss.length - 1] && Character.isWhitespace(gloss[gloss.length - 1])) {
          assert starts[gloss.length - 1] >= 0;
          finishedSpans.add(new Span(starts[gloss.length - 1], tokI + 1));
        }
      }
    }

    // Find closest match (or else first match)
    // Shortcuts
    if (finishedSpans.size() == 0) { return Maybe.Nothing(); }
    if (finishedSpans.size() == 1) { return Maybe.Just(finishedSpans.get(0)); }
    if (guess.isDefined()) {
      // Case; find closest span
      Span toReturn = null;
      int min = Integer.MAX_VALUE;
      for (Span candidate : finishedSpans) {
        int distance = Math.abs(candidate.start() - guess.get().start()) + Math.abs(candidate.end() - guess.get().end());
        if (distance < min) {
          min = distance;
          toReturn = candidate;
        }
      }
      return Maybe.Just(toReturn);
    } else {
      // Case:
      return Maybe.Just(finishedSpans.get(0));
    }
  }

  /**
   * @see Utils#getTokenSpan(char[][], char[], Maybe)
   */
  private static Maybe<Span> getTokenSpan(List<CoreLabel> sentence, String glossString, Maybe<Span> guess, Class<? extends CoreAnnotation<String>> textAnnotation) {
    // Get characters from sentence
    char[][] tokens = new char[sentence.size()][];
    for (int i = 0; i < sentence.size(); ++i) {
      @SuppressWarnings("unchecked") char[] tokenChars = ((String) sentence.get(i).get((Class) textAnnotation)).toCharArray();
      tokens[i] = tokenChars;
    }
    // Call low-level [tested] implementation
    return getTokenSpan(tokens, glossString.toCharArray(), guess);

  }

  /**
   * Gets the best matching token span in the given sentence for the given gloss, with an optional
   * guessed span to help the algorithm out.
   *
   * @see Utils#getTokenSpan(char[][], char[], Maybe)
   */
  public static Maybe<Span> getTokenSpan(List<CoreLabel> sentence, String gloss, Maybe<Span> guess) {
    Maybe<Span> span = getTokenSpan(sentence, gloss, guess, CoreAnnotations.OriginalTextAnnotation.class);
    if (!span.isDefined()) {
      span = getTokenSpan(sentence, gloss, guess, CoreAnnotations.TextAnnotation.class);
    }
    return span;
  }


  /**
   * A utility data structure.
   * @see Utils#sortRelationsByPrior(java.util.Collection)
   */
  private static final Counter<String> relationPriors = new ClassicCounter<String>(){{
    for (RelationType rel : RelationType.values()) {
      setCount(rel.canonicalName, rel.priorProbability);
    }
  }};

  /**
   * Returns a sorted list of relations names, ordered by their prior probability.
   * This is guaranteed to return a stable order.
   * @param relations A collection of relations to sort.
   * @return A sorted list of the relations in descending order of prior probability.
   */
  public static List<String> sortRelationsByPrior(Collection<String> relations) {
    List<String> sorted = new ArrayList<String>(relations);
    Collections.sort(sorted, new Comparator<String>() {
      @Override
      public int compare(String o1, String o2) {
        double count1 = relationPriors.getCount(o1);
        double count2 = relationPriors.getCount(o2);
        if (count1 < count2) { return 1; }
        if (count2 < count1) { return -1; }
        return o1.compareTo(o2);
      }
    });
    return sorted;
  }

  /**
   * Checks if the given path <n>is</b> a loop. Note that there can be subloops within the path -- this is not checked for
   * by this function (see {@link Utils#doesLoopPath(java.util.Collection)}).
   * @param path The path, as a list of {@link KBPSlotFill}s, each of which represents an edge.
   * @return True if this path represents a complete loop.
   */
  public static boolean isLoopPath(List<KBPSlotFill> path) {
    return isLoop(CollectionUtils.lazyMap(path, new Function<KBPSlotFill, KBTriple>() {
      @Override
      public KBTriple apply(KBPSlotFill in) {
        return in.key;
      }
    }));
  }

  /**
   * Checks if the given path <n>is</b> a loop. Note that there can be subloops within the path -- this is not checked for
   * by this function (see {@link Utils#doesLoopPath(java.util.Collection)}).
   * @param path The path, as a list of {@link KBTriple}s, each of which represents an edge.
   * @return True if this path represents a complete loop.
   */
  public static boolean isLoop(List<KBTriple> path) {
    if (path.size() < 2) { return false; }
    // Get the first entity in the path
    KBPEntity firstEntity = path.get(0).getEntity();
    if (path.get(1).getEntity().equals(firstEntity) && !path.get(1).getSlotEntity().equalsOrElse(path.get(0).getSlotEntity().orNull(), false)) {
      firstEntity = path.get(0).getSlotEntity().orNull();
    }
    if (firstEntity == null) { return false; }
    // Get the last entity in the path
    KBPEntity lastEntity = path.get(path.size() - 1).getSlotEntity().orNull();
    if (lastEntity == null || path.get(path.size() - 2).getSlotEntity().equalsOrElse(lastEntity, false)) {
      lastEntity = path.get(path.size() - 1).getEntity();
    }
    // See if they match
    return firstEntity.equals(lastEntity);

  }

  /**
   * Checks if the given path <b>contains</b> a loop.
   * To check if a path is itself a loop, use {@link Utils#isLoopPath(java.util.List)}}.
   * @param path The path, as a set of {@link KBPSlotFill}s, each of which represents an edge.
   * @return True if this path contains a loop
   */
  public static boolean doesLoopPath(Collection<KBPSlotFill> path) {
    return doesLoop(CollectionUtils.lazyMap(path, new Function<KBPSlotFill, KBTriple>() {
      @Override
      public KBTriple apply(KBPSlotFill in) {
        return in.key;
      }
    }));
  }

  /**
   * Checks if the given path <b>contains</b> a loop.
   * To check if a path is itself a loop, use {@link Utils#isLoop(java.util.List)}}.
   * @param path The path, as a set of {@link KBTriple}s, each of which represents an edge.
   * @return True if this path contains a loop
   */
  public static boolean doesLoop(Collection<KBTriple> path) {
    Set<KBPEntity> entitiesInPath = new HashSet<KBPEntity>();
    for (KBTriple fill : path) {
      entitiesInPath.add(fill.getEntity());
      entitiesInPath.add(fill.getSlotEntity().orNull());
    }
    return entitiesInPath.size() <= path.size();
  }

  /**
   * Return all clauses which are valid antecedents for the passed clause.
   * For example, on input path A->B->C->A, it would return (A->B)+(B->C) and (A->B)+(C->A) and (B->C)+(C->A).
   * @param clause The input clause to find antecedents for.
   * @param normalize If true, normalize the resulting clauses, abstracting out entities (or renormalizing
   *                  to canonical form if the entities are already abstracted)
   * @return A set of valid antecedents, which should be added to the set of paths seen.
   */
  private static Set<Set<KBTriple>> getValidAntecedents(Collection<KBTriple> clause, boolean normalize) {
    Set<Set<KBTriple>> rtn = new HashSet<Set<KBTriple>>();
    HashSet<KBTriple> mutableClause = new HashSet<KBTriple>(clause);
    for (KBTriple consequent : clause) {  // for each candidate consequent
      mutableClause.remove(consequent);
      // This block just checks if both entities in the consequent are present somewhere
      // in the antecedents
      KBPEntity e1 = consequent.getEntity();
      KBPEntity e2 = consequent.getSlotEntity().orNull();  // careful with equals() order here
      boolean hasE1 = false;
      boolean hasE2 = false;
      for (KBTriple antecedent : mutableClause) {
        KBPEntity a1 = antecedent.getEntity();
        Maybe<KBPEntity> a2 = antecedent.getSlotEntity();
        if (a1.equals(e1) || a2.equalsOrElse(e1, false)) { hasE1 = true; }
        if (a1.equals(e2) || a2.equalsOrElse(e2, false)) { hasE2 = true; }
      }
      // If this is true, add the antecedents for this consequent as a valid antecedent
      if (hasE1 && hasE2) {
        rtn.add(normalize ? normalizeConjunction(mutableClause) : new HashSet<KBTriple>(mutableClause));
      }
      mutableClause.add(consequent);
    }
    return rtn;
  }

  /** @see Utils#getValidAntecedents(java.util.Collection, boolean) */
  public static Set<Set<KBTriple>> getValidAntecedents(Collection<KBTriple> clause) {
    return getValidAntecedents(clause, false);
  }

  /** @see Utils#getValidAntecedents(java.util.Collection, boolean) */
  public static Set<Set<KBTriple>> getValidNormalizedAntecedents(Collection<KBTriple> clause) {
    return getValidAntecedents(clause, true);
  }

  /**
   * Replace the variables of an inferential path with canonical variable names (x0, x1, ...)
   * @param conjunction - Conjunction of paths
   * @return - conjunction of facts with variables replaced.
   */
  public static Pair<Double, Set<KBTriple>> abstractConjunction(Collection<KBPSlotFill> conjunction) {
    List<KBPSlotFill> fills = new ArrayList<KBPSlotFill>(conjunction);
    // Sort slot fills by name.
    Collections.sort( fills, new Comparator<KBPSlotFill>() {
      @Override
      public int compare(KBPSlotFill o1, KBPSlotFill o2) {
        int relCmp = o1.key.relationName.compareTo(o2.key.relationName);
        if (relCmp != 0) { return relCmp; }
        int eCmp = o1.key.entityName.compareTo(o2.key.entityName);
        if (eCmp != 0) { return eCmp; }
        int sCmp = o1.key.slotValue.compareTo(o2.key.slotValue);
        if (sCmp != 0) { return sCmp; }
        return o1.compareTo(o2);  // some random stable ordering
      }
    });

    // Now copy the slot fills into this set
    Map<KBPEntity, String> mapper = new HashMap<KBPEntity,String>();
    double score = 1.0;
    Set<KBTriple> abstractConjunction = new HashSet<KBTriple>();
    for( KBPSlotFill fill : fills ) {
      score *= fill.score.getOrElse(1.0);
      assert !Double.isNaN(score);
      KBPEntity head = fill.key.getEntity();
      KBPEntity tail = fill.key.getSlotEntity().orCrash();
      if(! mapper.containsKey(head)) mapper.put( head, "x" + mapper.size() );
      if(! mapper.containsKey(tail)) mapper.put( tail, "x" + mapper.size() );

      KBTriple pred = KBPNew
              .entName(mapper.get(head))
              .entType(head.type)
              .slotValue(mapper.get(tail))
              .slotType(tail.type)
              .rel(fill.key.relationName).KBTriple();
      abstractConjunction.add( pred );
    }

    return Pair.makePair(score, abstractConjunction);
  }

  /**
   * Takes a potentially already abstracted conjunction, and normalizes it.
   * This is particularly relevant if the conjunction is a subset of a larger conjunction,
   * and the variable names may not agree (e.g., it may start on x_1 rather than x_0).
   * @param conjunction Conjunction of paths to normalize, potentially already normalized once.
   *                    This input will not be mutated.
   * @return conjunction of facts with variables replaced. This is guaranteed to always be in a
   *           canonical form. This set is a copy of the input conjunction.
   */
  public static Set<KBTriple> normalizeConjunction(Collection<KBTriple> conjunction) {
    List<KBTriple> fills = new ArrayList<KBTriple>(conjunction);
    // Sort slot fills by name.
    Collections.sort( fills, new Comparator<KBTriple>() {
      @Override
      public int compare(KBTriple o1, KBTriple o2) {
        return o1.relationName.compareTo(o2.relationName);
      }
    });

    // Now copy the triples into this set
    Map<KBPEntity, String> mapper = new HashMap<KBPEntity,String>();
    double score = 1.0;
    Set<KBTriple> abstractConjunction = new HashSet<KBTriple>();
    for( KBTriple fill : fills ) {
      KBPEntity head = fill.getEntity();
      KBPEntity tail = fill.getSlotEntity().orCrash();
      if(! mapper.containsKey(head)) mapper.put( head, "x" + mapper.size() );
      if(! mapper.containsKey(tail)) mapper.put( tail, "x" + mapper.size() );

      KBTriple pred = KBPNew
          .entName(mapper.get(head))
          .entType(head.type)
          .slotValue(mapper.get(tail))
          .slotType(tail.type)
          .rel(fill.relationName).KBTriple();
      abstractConjunction.add( pred );
    }

    return abstractConjunction;
  }
}
