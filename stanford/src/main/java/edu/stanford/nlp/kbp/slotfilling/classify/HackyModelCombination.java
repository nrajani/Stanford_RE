package edu.stanford.nlp.kbp.slotfilling.classify;

import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Pair;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Map;
import java.util.Properties;

/**
 * A classifier which combines a number of models, taking guesses from each
 * according to what the relation type is.
 *
 * In particular, as of 2013-07-20, it's taking PER_TITLE and
 * ORG_TOP_EMPLOYEES from the old model and everything else from the
 * new model.
 *
 * @author Gabor Angeli (please don't judge me!)
 */
public class HackyModelCombination extends RelationClassifier {
  public final RelationClassifier oldModel;
  public final JointBayesRelationExtractor newModel;

  public HackyModelCombination(Properties props) {
    this.oldModel = Props.TRAIN_MODEL.load(Props.KBP_MODEL_PATH, props);
    this.newModel = ModelType.JOINT_BAYES.load("/scr/nlp/data/tackbp2013/models/2013-07-20/kbp_relation_model.JOINT_BAYES.15.ser", props);
  }

  public Counter<Pair<String, Maybe<KBPRelationProvenance>>> classifyRelations(SentenceGroup input, Maybe<CoreMap[]> rawSentences) {
    ClassicCounter<Pair<String, Maybe<KBPRelationProvenance>>> combinedCounts = new ClassicCounter<Pair<String, Maybe<KBPRelationProvenance>>>();
    Counter<Pair<String, Maybe<KBPRelationProvenance>>> oldOutput = oldModel.classifyRelations(input, rawSentences);
    Counter<Pair<String, Maybe<KBPRelationProvenance>>> newOuptut = newModel.classifyRelations(input, rawSentences);
    for (Map.Entry<Pair<String, Maybe<KBPRelationProvenance>>, Double> entry : oldOutput.entrySet()) {
      if (RelationType.fromString(entry.getKey().first).equalsOrElse(RelationType.ORG_TOP_MEMBERS_SLASH_EMPLOYEES, false) ||
          RelationType.fromString(entry.getKey().first).equalsOrElse(RelationType.PER_TITLE, false)) {
        combinedCounts.setCount(entry.getKey(), entry.getValue());
      }
    }
    for (Map.Entry<Pair<String, Maybe<KBPRelationProvenance>>, Double> entry : newOuptut.entrySet()) {
      if (!RelationType.fromString(entry.getKey().first).equalsOrElse(RelationType.ORG_TOP_MEMBERS_SLASH_EMPLOYEES, false) &&
          !RelationType.fromString(entry.getKey().first).equalsOrElse(RelationType.PER_TITLE, false)) {
        combinedCounts.setCount(entry.getKey(), entry.getValue());
      }
    }
    return combinedCounts;
  }

  @Override
  public TrainingStatistics train(KBPDataset<String, String> trainSet) {
    throw new AbstractMethodError();
  }

  @Override
  public void load(ObjectInputStream in) throws IOException, ClassNotFoundException {
    /* do nothing */
  }

  @Override
  public void save(ObjectOutputStream out) throws IOException {
    /* dear God, certainly do nothing */
  }
}
