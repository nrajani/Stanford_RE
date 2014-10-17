package edu.stanford.nlp.kbp.slotfilling.classify;

import edu.stanford.nlp.kbp.slotfilling.common.Maybe;
import edu.stanford.nlp.util.MetaClass;

import java.io.IOException;
import java.util.Properties;

import static edu.stanford.nlp.util.logging.Redwood.Util.fatal;
import static edu.stanford.nlp.util.logging.Redwood.Util.log;

public enum ModelType {
  LR_INC            ("lr_inc",            true,  OneVsAllRelationExtractor.class,       new Object[]{}),     // LR with incomplete information (used at KBP 2011)
  PERCEPTRON        ("perceptron",        false, PerceptronExtractor.class,             new Object[]{}),     // boring local Perceptron
  PERCEPTRON_INC    ("perceptron_inc",    false, PerceptronExtractor.class,             new Object[]{}),     // local Perceptron with incomplete negatives
  AT_LEAST_ONCE     ("at_least_once",     false, HoffmannExtractor.class,               new Object[]{}),     // (Hoffman et al, 2011)
  AT_LEAST_ONCE_INC ("at_least_once_inc", false, PerceptronExtractor.class,             new Object[]{}),     // AT_LEAST_ONCE with incomplete information
  LOCAL_BAYES       ("local_bayes",       false, JointBayesRelationExtractor.class,     new Object[]{true}), // Mintz++
  JOINT_BAYES       ("joint_bayes",       false, JointBayesRelationExtractor.class,     new Object[]{}),     // MIML-RE
  ROBUST_LR         ("robust_lr",         true,  OneVsAllRelationExtractor.class,       new Object[]{true}), // robust LR with shift parameters
  ENSEMBLE          ("ensemble",          false, EnsembleRelationExtractor.class,       new Object[]{}),     // ensemble classifier
  TOKENREGEX        ("tokenregex",        true,  TokenRegexExtractor.class,             new Object[]{}),     // heuristic token regex extractor

  // Not real models, but rather just for debugging
  GOLD              ("gold",              false, GoldClassifier.class,                  new Object[]{});     // memorize the data

  public final String name;
  public final boolean isLocallyTrained;
  public final Class<? extends RelationClassifier> modelClass;
  public final Object[] constructorArgs;

  ModelType(String name, boolean isLocallyTrained, Class<? extends RelationClassifier> modelClass, Object[] constructorArgs) {
    this.name = name;
    this.isLocallyTrained = isLocallyTrained;
    this.modelClass = modelClass;
    this.constructorArgs = constructorArgs;
  }

  /**
   * Construct a new model of this type.
   * @param props The properties file to use in the construction of the model.
   * @param <A> The type of the model being constructed. The user must make sure this type is valid.
   * @return A new model of the type specified by this ModelType.
   */
  public <A extends RelationClassifier> A construct(Properties props) {
    // Create MetaClass for loading
    log("constructing new model of type " + modelClass.getSimpleName());
    MetaClass clazz = new MetaClass(modelClass);
    // Create arguments
    Object[] args = new Object[constructorArgs.length + 1];
    args[0] = props;
    System.arraycopy(constructorArgs, 0, args, 1, constructorArgs.length);
    // Create instance
    try {
      return clazz.createInstance(args);
    } catch (MetaClass.ConstructorNotFoundException e) {
      fatal("classifier of type " + modelClass.getSimpleName() + " has no constructor "  + modelClass.getSimpleName() + "(Properties, ...)");
    }
    throw new IllegalStateException("code cannot reach here");
  }

  /**
   * Load a model of this type from a path.
   * @param path The path to the model being loaded.
   * @param props The properties file to use when loading the model.
   * @param <A> The type of the model being constructed. The user must make sure this type is valid.
   * @return The [presumably trained] model specified by the path (and properties).
   */
  @SuppressWarnings("unchecked")
  public <A extends RelationClassifier> A load(String path, Properties props) {
    assert path != null;
    log("loading model of type " + modelClass.getSimpleName() + " from " + path);
    try {
      // Route this call to Load to a call of AbstractModel.load
      return RelationClassifier.load(path, props, (Class<A>) modelClass);
    } catch (IOException e) {
      e.printStackTrace();
      fatal("IOException while loading model of type: " + modelClass.getSimpleName() + " at " + path);
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
      fatal("Could not find class: " + modelClass.getSimpleName() + " at " + path);
    }
    throw new IllegalStateException("code cannot reach here");
  }

  public static Maybe<ModelType> fromString(String name) {
    for (ModelType slot : ModelType.values()) {
      if (slot.name.equals(name)) return Maybe.Just(slot);
    }
    return Maybe.Nothing();
  }
}
