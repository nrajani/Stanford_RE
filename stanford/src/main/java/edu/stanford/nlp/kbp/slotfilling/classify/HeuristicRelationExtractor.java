package edu.stanford.nlp.kbp.slotfilling.classify;

import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Function;
import edu.stanford.nlp.util.Pair;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Collection;

import static edu.stanford.nlp.util.logging.Redwood.Util.log;

/**
 * A base class for some heuristic, rule-based relation extractors.
 *
 * @author Gabor Angeli
 */
public abstract class HeuristicRelationExtractor extends RelationClassifier {

  public abstract Collection<Pair<String, Integer>> extractRelations(KBPair key, CoreMap[] input);

  @Override
  public Counter<Pair<String,Maybe<KBPRelationProvenance>>> classifyRelations(SentenceGroup input, Maybe<CoreMap[]> rawSentences) {
    Counter<Pair<String, Maybe<KBPRelationProvenance>>> rels = new ClassicCounter<Pair<String, Maybe<KBPRelationProvenance>>>();
    for (CoreMap[] sentences : rawSentences) {
      for (Pair<String, Integer> rel : extractRelations(input.key, sentences)) {
        rels.setCount(Pair.makePair(rel.first, KBPRelationProvenance.compute(sentences[rel.second], KBPNew.from(input.key).rel(rel.first).KBTriple())), Double.POSITIVE_INFINITY);
      }
    }
    return rels;
  }

  @Override
  public TrainingStatistics train(KBPDataset<String, String> trainSet) {
    // NOOP
    log("training for a heuristic relation classifier is a noop");
    return TrainingStatistics.empty();
  }

  @Override
  public void load(ObjectInputStream in) throws IOException, ClassNotFoundException {
    // NOOP
    assert in == null;
  }

  @Override
  public void save(ObjectOutputStream out) throws IOException {
    // NOOP
  }


  public static final Function<Pair<KBPair, CoreMap[]>, Counter<Pair<String, Maybe<KBPRelationProvenance>>>> allExtractors;
  static {
    // Extractors to run
    final HeuristicRelationExtractor[] extractors = new HeuristicRelationExtractor[]{
        new TokenRegexExtractor()
    };
    // Set the allExtractors variable
    allExtractors = new Function<Pair<KBPair, CoreMap[]>, Counter<Pair<String, Maybe<KBPRelationProvenance>>>>() {
      @Override
      public Counter<Pair<String, Maybe<KBPRelationProvenance>>> apply(Pair<KBPair, CoreMap[]> in) {
        Counter<Pair<String, Maybe<KBPRelationProvenance>>> extractions = new ClassicCounter<Pair<String, Maybe<KBPRelationProvenance>>>();
        for (HeuristicRelationExtractor extractor : extractors) {
          for (Pair<String, Integer> rel : extractor.extractRelations(in.first, in.second)) {
            extractions.setCount(Pair.makePair(rel.first, KBPRelationProvenance.compute(in.second[rel.second], KBPNew.from(in.first).rel(rel.first).KBTriple())), 1.0);
          }
        }
        return extractions;
      }
    };
  }
}
