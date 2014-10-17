package edu.stanford.nlp.kbp.entitylinking.classify.namematcher;

import edu.stanford.nlp.dcoref.Mention;
import edu.stanford.nlp.dcoref.MentionMatcher;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Pair;

import java.util.Properties;

/**
 * Returns score about whether str1 refers to the same object as str2
 * - Other types of relationship ( str1 is more specific than str2  or str2 more specific than str1
 *                                 str1/str2 are hypernym/hyponym)
 * @author Angel Chang
 */
public abstract class NameMatcher implements MentionMatcher {

  public enum MatchType { EXACT, COMPATIBLE, MORE_SPECIFIC, LESS_SPECIFIC, INCOMPATIBLE }
  // EXACT - Names are exactly the same
  // MORE_SPECIFIC - Name1 is more specific than name2:
  //   George H. W. Bush MORE_SPECIFIC George Bush MORE_SPECIFIC Bush
  // LESS_SPECIFIC - Name1 is less specific than name2:
  // COMPATIBLE - Two names are compatible in some other way

  public void init(String name, Properties props)
  {
  }

  /**
   * Given two names, returns a score from 0 to 1 indicating
   * how likely they are to refer to the same entity
   * @param name1
   * @param name2
   */
  public double getMatchScore(String name1, String name2)
  {
    return 0;
  }

  /**
   * Given two pairs of names and NER type, returns a score from 0 to 1 indicating
   * how likely they are to refer to the same entity
   * @param mention1
   * @param mention2
   */
  public double getMatchScore(Pair<String,String> mention1, Pair<String,String> mention2)
  {
    return 0;
  }

  /**
   * Given two names, returns most likely relationship of name1 to name2
   * @param name1
   * @param name2
   */
  public NameMatcher.MatchType getMatchType(String name1, String name2)
  {
    return null;
  }

  /**
   * Given two names indicate relationship of name1 to name2
   * @param name1
   * @param name2
   */
  public Counter<NameMatcher.MatchType> getMatchTypeScores(String name1, String name2)
  {
    return null;
  }

  //public Counter<String> getMatchFeatures(String name1, String name2)
  //{
  //  return null;
  //}


  // Integration with coref system
  public Boolean isCompatible(Mention mention1, Mention mention2) {
    MatchType m = getMatchType(mention1, mention2);
    if (m != null) {
      // Explictly mark mentions as compatible or incompatible
      return (m == MatchType.INCOMPATIBLE)? false:true;
    } else {
      // Not sure, return null
      return null;
    }
  }

  public MatchType getMatchType(Mention mention1, Mention mention2) {
    return getMatchType(mention1.spanToString(), mention2.spanToString());
  }


}
