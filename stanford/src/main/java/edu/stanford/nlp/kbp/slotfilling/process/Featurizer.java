package edu.stanford.nlp.kbp.slotfilling.process;

import java.util.*;

import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.logging.Redwood;

/**
 * @author kevinreschke
 */
public abstract class Featurizer {
  
  protected static final Redwood.RedwoodChannels logger = Redwood.channels("Featurize");
  
  /** Build datums from |annotation| for relation mentions headed by |entity|
   *  A relation classifier is supplied for optional filtering. */
  public HashMap<KBPair,SentenceGroup> featurize(Annotation annotation, Maybe<RelationFilter> relationFilter) {
    HashMap<KBPair, SentenceGroup> featurized = new HashMap<KBPair, SentenceGroup>();
    for (Map.Entry<KBPair, Pair<SentenceGroup, List<CoreMap>>> entry : featurizeWithSentences(annotation, relationFilter).entrySet()) {
      featurized.put(entry.getKey(), entry.getValue().first);
    }
    return featurized;
  }
  
  /** Featurize with null relation filter */
  public HashMap<KBPair,SentenceGroup> featurize(Annotation annotation) {
    return featurize(annotation, Maybe.<RelationFilter>Nothing());
  }
  
  
  public Map<KBPair, Pair<SentenceGroup, List<CoreMap>>> featurizeWithSentences(Annotation annotation, Maybe<RelationFilter> relationFilter) {
    
    //debug: count duplicate sentences
    Counter<String> dupSents = new ClassicCounter<String>();
    
    HashMap<KBPair,Pair<SentenceGroup,List<CoreMap>>> datums = new HashMap<KBPair, Pair<SentenceGroup, List<CoreMap>>>();
    for (CoreMap sentence : annotation.get(SentencesAnnotation.class)) {
      
      //debug: count duplicate sentences
      dupSents.incrementCount(CoreMapUtils.sentenceToMinimalString(sentence));
      
      for(SentenceGroup sg : featurizeSentence(sentence, relationFilter)) {
        KBPair key = sg.key;
        if( !datums.containsKey(key) ) {
          //TODO (arun) change to SentenceGroup.empty
          datums.put(key, Pair.makePair(sg, (List<CoreMap>) new ArrayList<CoreMap>()));
        }
        else {
          datums.get(key).first.merge(sg);
        }
        datums.get(key).second.add(sentence);
      }
    }
    
    //debug: count duplicate sentences
    if(Counters.max(dupSents) > 1) {
      logger.debug("Duplicate sentences in featurizer");
      logger.debug("Counts of Counts:\n"+Counters.toVerticalString(Counters.getCountCounts(dupSents)));
      logger.debug("Counts:\n"+Counters.toVerticalString(dupSents));
    }
    
    return datums;
  }
  
  /**
   * Build datums for relations found in |sentence| and headed by |entity|.
   * 
   * @param sentence The sentence to featurize
   * @param relationFilter  Optional relation filter for within-sentence filtering
   * @return List of singleton sentence groups (each with a single datum).  
   */
  public abstract List<SentenceGroup> featurizeSentence(CoreMap sentence, Maybe<RelationFilter> relationFilter);
}
