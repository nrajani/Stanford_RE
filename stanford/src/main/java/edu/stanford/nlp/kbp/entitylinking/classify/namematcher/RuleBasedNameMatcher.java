package edu.stanford.nlp.kbp.entitylinking.classify.namematcher;

import edu.stanford.nlp.dcoref.CorefChain;
import edu.stanford.nlp.dcoref.Dictionaries;
import edu.stanford.nlp.dcoref.Mention;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Name matcher
 *
 * @author Angel Chang
 */
public class RuleBasedNameMatcher extends NameMatcher {

  private final static String DEFAULT_FEMALE_NAMES =  "/scr/nlp/data/tackbp2013/data/worldknowledge/kbp_female_names.txt";
  private final static String DEFAULT_MALE_NAMES =  "/scr/nlp/data/tackbp2013/data/worldknowledge/kbp_male_names.txt";
  private final static String DEFAULT_ASCII_MAP = "/scr/nlp/data/tackbp2013/solr/cores/conf/mapping-FoldToASCII.txt";
//  private final static String DEFAULT_FEMALE_NAMES =  "C:\\code\\NLP\\kbp\\javanlp\\models\\kbp_female_names.txt";
//  private final static String DEFAULT_MALE_NAMES =  "C:\\code\\NLP\\kbp\\javanlp\\models\\kbp_male_names.txt";
//  private final static String DEFAULT_ASCII_MAP = "C:\\code\\NLP\\kbp\\javanlp\\models\\mapping-FoldToASCII.txt";

  private static final Pattern delimiterPattern = Pattern.compile("(\\s+|_+)");
  private static final Pattern periodPattern = Pattern.compile("\\.");
  private static final Pattern discardPattern = Pattern.compile("[-._]");
  private static final Pattern punctPattern = Pattern.compile("\\p{Punct}");

  private static final Set<String> PERSON_TYPE = CollectionUtils.asSet(new String[]{"PERSON", "PER"});

  // Known list of aliases
  CollectionValuedMap<String,String> aliases; // Map from name to canonical form(s)
  Map<Character,String> charMap; // Normalization map from diacriticals to ascii
  // Map from first name to variants of the name (typically longer is the full form)
  CollectionValuedMap<String, String> personalNameVariants = new CollectionValuedMap<String, String>();

  // Names compatible with female/male genders
  // Unisex names goes in both sets
  CollectionValuedMap<Dictionaries.Gender, String> personalNamesByGender = new CollectionValuedMap<Dictionaries.Gender, String>();
  boolean normalize = true;

  public RuleBasedNameMatcher() {
    // initialize with default properties
    this.init(null, new Properties());
  }

  public RuleBasedNameMatcher(String name, Properties properties) {
    this.init(name, properties);
  }

  public void init(String name, Properties props)
  {
    String prefix = (name != null)? name + ".":"";
    String femaleNicknames = props.getProperty(prefix + "nicknames.female", DEFAULT_FEMALE_NAMES);
    String maleNicknames = props.getProperty(prefix + "nicknames.male", DEFAULT_MALE_NAMES);
    String charMapFile = props.getProperty(prefix + "charmapping", DEFAULT_ASCII_MAP);
    String[] aliases = PropertiesUtils.getStringArray(props, prefix + "aliases");
    try {
      readPersonalNameVariants(femaleNicknames, Dictionaries.Gender.FEMALE);
      readPersonalNameVariants(maleNicknames, Dictionaries.Gender.MALE);
      readAliases(aliases);
      if (normalize) readCharMap(charMapFile);
    } catch (IOException ex) {
      throw new RuntimeException("Error initializing RuleBasedNameMatcher " + name, ex);
    }
    boolean useDict = Boolean.parseBoolean(props.getProperty(prefix + "usedict", "false"));
  }

  public Counter<String> phraseDictLookup(String word) {
    return new ClassicCounter<String>();
  }

//  public List<String> wikiDictLookup(String word) {
//    SolrDict solrDict = new SolrDict(solrUrl);
//    solrDict.setDictType(Dict.DictType.EXCT);
//    Dict.DictValue dictValue = solrDict.get(word);
//    dictValue.
//    Dict heurDict = CombinedDict.heuristicCombine(dict, null, null);
//    Collection<String> heurWords = heurDict.getWords(title);
//  }
//
//  public void setupWikiDict(String solrUrl) {
//    SolrDict solrDict = new SolrDict(solrUrl);
//    solrDict.setDictType(Dict.DictType.EXCT);
//    Dict heurDict = CombinedDict.heuristicCombine(dict, null, null);
//    Collection<String> heurWords = heurDict.getWords(title);
//
//  }

  private static final Pattern COMMA_DELIM_PATTERN = Pattern.compile("\\s*,\\s*");
  public void readAliases(String... filenames) throws IOException {
    // Read in input of aliases
    aliases = new CollectionValuedMap<String, String>();
    for (String filename:filenames) {
      BufferedReader br = IOUtils.getBufferedFileReader(filename);
      String line = null;
      while ((line = br.readLine()) != null) {
        // Same entity
        String[] fields = COMMA_DELIM_PATTERN.split(line);
        for (String field: fields) {
          for (String alt: fields) {
            aliases.add(field, alt);
          }
        }
      }
    }
  }

  private final static Pattern tabPattern = Pattern.compile("\t");
  private final static Pattern commaPattern = Pattern.compile("\\s*,\\s*");
  public void readPersonalNameVariants(String filename, Dictionaries.Gender gender) throws IOException {
    BufferedReader br = IOUtils.getBufferedFileReader(filename);
    String line;
    while ((line = br.readLine()) != null) {
      line = line.toUpperCase();
      String[] names = tabPattern.split(line,2);
      String[] mainNames = commaPattern.split(names[0]);
      Set<String> nameSet = new HashSet<String>();
      nameSet.addAll(Arrays.asList(mainNames));
      if (names.length > 1) {
        // Has name variants
        String[] variants = commaPattern.split(names[1]);
        // Store our variants away
        nameSet.addAll(Arrays.asList(variants));
      }
      // Add to table of personal names....
      for (String name: nameSet) {
        personalNameVariants.addAll(name, nameSet);
      }
      personalNamesByGender.addAll(gender, nameSet);

    }
    br.close();
  }

  private final static Pattern charMapPattern = Pattern.compile("\\s*\"(.+)\"\\s*=>\\s*\"(.+)\"\\s*");
  public void readCharMap(String filename) throws IOException {
    charMap = new HashMap<Character, String>();
    BufferedReader br = IOUtils.getBufferedFileReader(filename);
    int lineno = 0;
    String line;
    while ((line = br.readLine()) != null) {
      line = line.trim();
      // Skip comments and empty lines
      if (line.startsWith("#") || line.isEmpty()) continue;
      Matcher m = charMapPattern.matcher(line);
      if (m.matches()) {
        String c = m.group(1);
        String repl = m.group(2);
        if (c.length() == 1) {
          charMap.put(c.charAt(0), repl);
        } else if (c.length() == 6 && c.startsWith("\\u")) {
          int i = Integer.parseInt(c.substring(2,6), 16);
          char[] chars = Character.toChars(i);
          if (chars.length == 1) {
            charMap.put(chars[0], repl);
          } else {
            Redwood.log(Redwood.WARN, "Unsupported multi-character substitution (" + filename + ":" + lineno + "): " + line);
          }
        } else {
          throw new RuntimeException("Unexpected line (" + filename + ":" + lineno + "): " + line);
        }
      } else {
        throw new RuntimeException("Unexpected line (" + filename + ":" + lineno + "): " + line);
      }
      lineno++;
    }
    br.close();
  }

  // Public functions

  /**
   * Given two names, returns a score from 0 to 1 indicating
   * how likely they are to refer to the same entity
   * @param name1
   * @param name2
   */
  public double getMatchScore(String name1, String name2)
  {
    NameMatcher.MatchType matchType = getMatchType(name1, name2);
    if (matchType != null) {
      return 1;
    } else {
      return 0;
    }
  }

  public double getMatchScore(Pair<String,String> mention1, Pair<String,String> mention2) {
    NameMatcher.MatchType matchType = getMatchType(mention1, mention2);
    if (matchType != null) {
      return 1;
    } else {
      return 0;
    }
  }

  public double getMatchScore(CorefChain.CorefMention m1, CorefChain.CorefMention m2)
  {
    // TODO: Get if the mention is a person
    NameMatcher.MatchType matchType = getMatchType(m1.mentionSpan, m2.mentionSpan);
    if (matchType != null) {
      return 1;
    } else {
      return 0;
    }
  }

  /**
   * Given two names, returns most likely relationship of name1 to name2
   * @param name1
   * @param name2
   */
  public NameMatcher.MatchType getMatchType(String name1, String name2)
  {
    if (normalize) {
      name1 = normalize(name1);
      name2 = normalize(name2);
    }
    return getMatchTypeAfterNormalization(name1, name2);
  }
  /**
   * Checks if two mentions (represented as pair of name, ner tag) matches
   * @param mention1
   * @param mention2
   * @return
   */
  public MatchType getMatchType(Pair<String,String> mention1, Pair<String,String> mention2) {
    String name1 = mention1.first;
    String name2 = mention2.first;
    if (normalize) {
      name1 = normalize(name1);
      name2 = normalize(name2);
    }
    MatchType m = getMatchType(name1, name2);
    if (m == null) {
      if (PERSON_TYPE.contains(mention1.second()) && PERSON_TYPE.contains(mention2.second())) {
        m = getMatchTypeForPerson(name1, name2);
        if (m == null) m = MatchType.INCOMPATIBLE;
      }
    }
//    if (m != null) Redwood.log("Match " + mention1 + " with " + mention2 + " => " + m);
    return m;
  }

  public MatchType getMatchType(Mention mention1, Mention mention2) {
    // for now, just call the pair version
    String n1 = mention1.nerName();
    if (n1 == null) n1 = mention1.spanToString();
    String n2 = mention2.nerName();
    if (n2 == null) n2 = mention2.spanToString();
    return getMatchType(Pair.makePair(n1.trim(), mention1.nerString),
            Pair.makePair(n2.trim(), mention2.nerString));
  }

  public String normalize(String str) {
    return normalizeChars(str);
  }

  // Normalizes a name using our charMap
  public String normalizeChars(String str) {
    String res = str;
    if (!charMap.isEmpty()) {
      StringBuilder sb = null;
      int processed = 0;
      for (int i = 0; i < str.length(); i++) {
        char c = str.charAt(i);
        String repl = charMap.get(c);
        if (repl != null) {
          if (sb == null) { sb = new StringBuilder(); }
          if (processed < i-1) {
            sb.append(str.substring(processed, i));
          }
          sb.append(repl);
          processed = i+1;
        }
      }
      if (sb != null) {
        if (processed < str.length()-1)
          sb.append(str.substring(processed, str.length()));
        res = sb.toString();
      }
    }
    return res;
  }

  private NameMatcher.MatchType getMatchTypeAfterNormalization(String name1, String name2)
  {
    if (name1.equalsIgnoreCase(name2)) {
      return MatchType.EXACT;
    }
    String[] name2parts = delimiterPattern.split(name2);
    String[] name1parts = delimiterPattern.split(name1);
    if (isAcronymImpl(name1, getMainStrs(Arrays.asList(name2parts)))) {
      return MatchType.LESS_SPECIFIC;
    }
    if (isAcronymImpl(name2, getMainStrs(Arrays.asList(name1parts)))) {
      return MatchType.MORE_SPECIFIC;
    }
    if (name1parts.length >= name2parts.length) {
      if (mainPartsAppearsIn(name2parts, name1parts)) {
        return MatchType.MORE_SPECIFIC;
      }
    } else {
      if (mainPartsAppearsIn(name1parts, name2parts)) {
        return MatchType.LESS_SPECIFIC;
      }
    }

    if (aliases != null) {
      Collection<String> cnames1 = aliases.get(name1);
      if (cnames1 == null) { cnames1 = CollectionUtils.asSet(new String[] { name1 }); }
      Collection<String> cnames2 = aliases.get(name2);
      if (cnames2 == null) { cnames2 = CollectionUtils.asSet(new String[] { name2 }); }
      for (String cname1: cnames1) {
        for (String cname2: cnames2) {
          if (cname1.equalsIgnoreCase(cname2)) {
            return MatchType.COMPATIBLE;
          }
        }
      }
    }

    return null;
  }

  private MatchType getMatchTypeForPerson(String name1, String name2)
  {
    // Turn our names to upper case
//    name1 = name1.toUpperCase();
//    name2 = name2.toUpperCase();
    // Specific name matching for people
    String[] name2parts = delimiterPattern.split(name2);
    String[] name1parts = delimiterPattern.split(name1);
    return getMatchTypeForPerson(name1parts, name2parts);
  }

  private MatchType getMatchTypeForPerson(String[] name1parts, String[] name2parts) {
    name1parts = reorderPersonName(name1parts);
    name2parts = reorderPersonName(name2parts);
    String[] lessNameParts = name1parts;
    String[] moreNameParts = name2parts;
    boolean isFlipSpecificity = false;
    if (name1parts.length > name2parts.length) {
      lessNameParts = name2parts;
      moreNameParts = name1parts;
      isFlipSpecificity = true;
    }
    if (lessNameParts.length == moreNameParts.length) {
      // same length name...
      MatchType m = checkPersonalName(lessNameParts[0], moreNameParts[0]);
      if (m != null) {
        // check rest
        for (int i = 1; i < lessNameParts.length; i++) {
          if (!lessNameParts[i].equalsIgnoreCase(moreNameParts[i])) return null;
        }
      }
      return flipSpecificity(m, isFlipSpecificity);
    } else {
      if (lessNameParts.length == 1) {
        // Check this name matches one of the other names...
        for (int i = 0; i < moreNameParts.length; i++) {
          MatchType m = checkPersonalName(lessNameParts[0], moreNameParts[i]);
          if (m != null) return flipSpecificity(m, isFlipSpecificity);
        }
      } else if (lessNameParts.length == 2) {
        // Check if last name matches
        if (lessNameParts[lessNameParts.length-1].equalsIgnoreCase(moreNameParts[moreNameParts.length-1])) {
          // Check this name matches one of the other names...
          for (int i = 0; i < moreNameParts.length-1; i++) {
            MatchType m = checkPersonalName(lessNameParts[0], moreNameParts[i]);
            if (m != null) return flipSpecificity(m, isFlipSpecificity);
          }
        }
      } else {
        // TODO: implement something here
      }
      return null;
    }
  }

  private MatchType flipSpecificity(MatchType m, boolean isFlip) {
    if (isFlip) {
      if (MatchType.LESS_SPECIFIC.equals(m)) return MatchType.MORE_SPECIFIC;
      if (MatchType.MORE_SPECIFIC.equals(m)) return MatchType.LESS_SPECIFIC;
    }
    return m;
  }

  private MatchType checkPersonalName(String s1, String s2) {
    if (s1.equalsIgnoreCase(s2)) return MatchType.EXACT;
    Collection c = personalNameVariants.get(s1.toUpperCase());
    if (c != null) {
      if (c.contains(s2.toUpperCase())) return MatchType.COMPATIBLE;
    }
    if ( !s1.isEmpty() && !s2.isEmpty() &&
        ( s1.length() == 1 || s2.length() == 1) )  {
      char c1 = Character.toUpperCase(s1.charAt(0));
      char c2 = Character.toUpperCase(s2.charAt(0));
      if (c1 == c2) {
        return (s1.length() < s2.length())? MatchType.LESS_SPECIFIC: MatchType.MORE_SPECIFIC;
      }
    }
    return null;
  }

  private String[] reorderPersonName(String[] nameparts) {
    // put first name in front
    if (nameparts.length == 3 && ",".equals(nameparts[1])) {
      return new String[]{nameparts[2], nameparts[0]};
    }
    return nameparts;
  }

  private String getCanonical(String name) {
    // Lookup name in alias map
    Collection<String> cnames = aliases.get(name);
    String cname = (cnames != null && !cnames.isEmpty())? cnames.iterator().next():null;
    if (cname == null) cname = name;
    return cname;
  }

  private static boolean mainPartsAppearsIn(String[] parts1, String[] parts2)
  {
    boolean matched = false;
    for (String p:parts1) {
      if( !p.isEmpty() && (p.length() >= 4 || Character.isUpperCase(p.charAt(0))) ) {
        boolean ok = false;
        for (String p2:parts2) {
          if (p2.equalsIgnoreCase(p)) {
            ok = true;
            break;
          }
        }
        if (!ok) { return false; }
        matched = true;
      }
    }
    return matched;
  }

  private static List<String> getTokenStrs(List<CoreLabel> tokens)
  {
    List<String> mainTokenStrs = new ArrayList<String>(tokens.size());
    for (CoreLabel token:tokens) {
      String text = token.get(CoreAnnotations.TextAnnotation.class);
      mainTokenStrs.add(text);
    }
    return mainTokenStrs;
  }

  private static List<String> getMainTokenStrs(List<CoreLabel> tokens)
  {
    List<String> mainTokenStrs = new ArrayList<String>(tokens.size());
    for (CoreLabel token:tokens) {
      String text = token.get(CoreAnnotations.TextAnnotation.class);
      if (!text.isEmpty() && ( text.length() >= 4 || Character.isUpperCase(text.charAt(0))) ) {
        mainTokenStrs.add(text);
      }
    }
    return mainTokenStrs;
  }

  private static List<String> getMainTokenStrs(String[] tokens)
  {
    List<String> mainTokenStrs = new ArrayList<String>(tokens.length);
    for (String text:tokens) {
      if ( !text.isEmpty() && ( text.length() >= 4 || Character.isUpperCase(text.charAt(0)) ) ) {
        mainTokenStrs.add(text);
      }
    }
    return mainTokenStrs;
  }

  private static List<String> getMainStrs(List<String> tokens)
  {
    List<String> mainTokenStrs = new ArrayList<String>(tokens.size());
    for (String text:tokens) {
      if ( !text.isEmpty() && (text.length() >= 4 || Character.isUpperCase(text.charAt(0))) ) {
        mainTokenStrs.add(text);
      }
    }
    return mainTokenStrs;
  }

  public static boolean isAcronym(String str, String[] tokens) {
    return isAcronymImpl(str, Arrays.asList(tokens));
  }
  // Public static utility methods
  public static boolean isAcronymImpl(String str, List<String> tokens)
  {
    str = discardPattern.matcher(str).replaceAll("");
    if (str.length() == tokens.size()) {
      for (int i = 0; i < str.length(); i++) {
        char ch = Character.toUpperCase(str.charAt(i));
        if ( !tokens.get(i).isEmpty() &&
            Character.toUpperCase(tokens.get(i).charAt(0)) != ch ) {
          return false;
        }
      }
      return true;
    } else {
      return false;
    }
  }

  public static boolean isAcronym(String str, List<? extends Object> tokens)
  {
    List<String> strs = new ArrayList<String>(tokens.size());
    for (Object tok : tokens) { strs.add(tok instanceof String ? tok.toString() : ((CoreLabel) tok).word()); }
    return isAcronymImpl(str, strs);
  }

  /**
   * Returns true if either chunk1 or chunk2 is acronym of the other
   * @return true if either chunk1 or chunk2 is acronym of the other
   */
  public static boolean isAcronym(CoreMap chunk1, CoreMap chunk2)
  {
    String text1 = chunk1.get(CoreAnnotations.TextAnnotation.class);
    String text2 = chunk2.get(CoreAnnotations.TextAnnotation.class);
    if (text1.length() <= 1 || text2.length() <= 1) { return false; }
    List<String> tokenStrs1 = getTokenStrs(chunk1.get(CoreAnnotations.TokensAnnotation.class));
    List<String> tokenStrs2 = getTokenStrs(chunk2.get(CoreAnnotations.TokensAnnotation.class));
    boolean isAcro = isAcronymImpl(text1, tokenStrs2) || isAcronymImpl(text2, tokenStrs1);
    if (!isAcro) {
      tokenStrs1 = getMainTokenStrs(chunk1.get(CoreAnnotations.TokensAnnotation.class));
      tokenStrs2 = getMainTokenStrs(chunk2.get(CoreAnnotations.TokensAnnotation.class));
      isAcro = isAcronymImpl(text1, tokenStrs2) || isAcronymImpl(text2, tokenStrs1);
    }
    return isAcro;
  }

  /** @see edu.stanford.nlp.kbp.entitylinking.classify.namematcher.RuleBasedNameMatcher#isAcronym(edu.stanford.nlp.util.CoreMap, edu.stanford.nlp.util.CoreMap) */
  public static boolean isAcronym(String[] chunk1, String[] chunk2)
  {
    String text1 = StringUtils.join(chunk1);
    String text2 = StringUtils.join(chunk2);
    if (text1.length() <= 1 || text2.length() <= 1) { return false; }
    List<String> tokenStrs1 = Arrays.asList(chunk1);
    List<String> tokenStrs2 = Arrays.asList(chunk2);
    boolean isAcro = isAcronymImpl(text1, tokenStrs2) || isAcronymImpl(text2, tokenStrs1);
    if (!isAcro) {
      tokenStrs1 = getMainTokenStrs(chunk1);
      tokenStrs2 = getMainTokenStrs(chunk2);
      isAcro = isAcronymImpl(text1, tokenStrs2) || isAcronymImpl(text2, tokenStrs1);
    }
    return isAcro;
  }
  
  /** True if candidate is a reasonable alias of entity.
   *  E.g.  Zondervan is an alias of Zondervan Publishing, Inc.
   * 
   * @param entity
   * @param candidate
   */
  public static boolean isAlias(String entity, String candidate) {
    //TODO (kevin) implement
    return false;
  }
}
