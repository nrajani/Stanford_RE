package edu.stanford.nlp.kbp.slotfilling;

import edu.stanford.nlp.kbp.slotfilling.classify.HackyModelCombination;
import edu.stanford.nlp.kbp.slotfilling.classify.RelationClassifier;
import edu.stanford.nlp.kbp.slotfilling.common.Maybe;
import edu.stanford.nlp.kbp.slotfilling.common.Props;
import edu.stanford.nlp.kbp.slotfilling.evaluate.KBPEvaluator;
import edu.stanford.nlp.kbp.slotfilling.evaluate.KBPSlotValidator;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.ir.StandardIR;
import edu.stanford.nlp.kbp.slotfilling.process.KBPProcess;
import edu.stanford.nlp.kbp.slotfilling.train.KBPTrainer;
import edu.stanford.nlp.util.Execution;
import edu.stanford.nlp.util.Function;
import edu.stanford.nlp.util.logging.Redwood;

import java.util.Properties;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * This is the entry point for training and testing the KBP Slotfilling system on the official evaluations.
 *
 * @author Gabor Angeli
 */
public class SlotfillingSystem {

  protected static final Redwood.RedwoodChannels logger = Redwood.channels("MAIN");

  public final Properties props;

  public SlotfillingSystem(Properties props) {
    this.props = props;
    if (props.getProperty("annotators") == null) {
      props.setProperty("annotators", "tokenize, ssplit, pos, parse");
    }
  }


  //
  // Dependency Graph
  //
  // ir             classify
  //  |-> process---    |
  //  |     v      |    |
  //  |-> train <-------|
  //  |            v    |
  //  ------> evaluate <-
  //

  //
  // IR Component
  //
  private Maybe<StandardIR> querier = Maybe.Nothing();
  public KBPIR getIR() {
    // If cached, return
    //noinspection LoopStatementThatDoesntLoop
    for (KBPIR q : this.querier) { return q; }
    forceTrack("Creating Querier");
    // Create new querier
    this.querier = Maybe.Just(new StandardIR());
    // Return
    endTrack("Creating Querier");
    return querier.get();
  }

  //
  // Process Component
  //
  private Maybe<KBPProcess> process = Maybe.Nothing();
  public synchronized KBPProcess getProcess() {
    // If cached, return
    //noinspection LoopStatementThatDoesntLoop
    for (KBPProcess p : this.process) { return p; }
    forceTrack("Creating Process");
    // Create new processor
    this.process = Maybe.Just(new KBPProcess(this.props, this.getIR()));
    // Return
    endTrack("Creating Process");
    return process.get();
  }

  //
  // Classify Component
  //
  private Maybe<RelationClassifier> classifier = Maybe.Nothing();
  public synchronized RelationClassifier getClassifier(Maybe<String> modelFilename) {
    // If cached, return
    //noinspection LoopStatementThatDoesntLoop
    for (RelationClassifier m : this.classifier) { return m; }
    forceTrack("Creating Classifier");
    // Create new classifier
    RelationClassifier classifier;
    if (modelFilename.isDefined()){
      startTrack("Loading Classifier");
      if (Props.HACKS_HACKYMODELCOMBINATION) {
        classifier = new HackyModelCombination(props);
      } else {
        classifier = Props.TRAIN_MODEL.load(modelFilename.get(), props);
      }
      endTrack("Loading Classifier");
    } else {
      startTrack("Constructing Classifier");
      classifier = Props.TRAIN_MODEL.construct(props);
      endTrack("Constructing Classifier");
    }
    this.classifier = Maybe.Just(classifier);
    // Return
    endTrack("Creating Classifier");
    return this.classifier.get();
  }

  /** Create a new classifier */
  public RelationClassifier getNewClassifier() { return getClassifier(Maybe.<String>Nothing()); }
  /** Load an existing classifier */
  public RelationClassifier getTrainedClassifier() { return getClassifier(Maybe.Just(Props.KBP_MODEL_PATH)); }

  //
  // Training Component
  //
  private Maybe<edu.stanford.nlp.kbp.slotfilling.train.KBPTrainer> trainer = Maybe.Nothing();
  public edu.stanford.nlp.kbp.slotfilling.train.KBPTrainer getTrainer() {
    // If cached, return
    //noinspection LoopStatementThatDoesntLoop
    for (edu.stanford.nlp.kbp.slotfilling.train.KBPTrainer t : this.trainer) { return t; }
    forceTrack("Creating Trainer");
    // Create new trainer
    this.trainer = Maybe.Just(new KBPTrainer(props, getIR(), getProcess(), Props.TRAIN_MODEL.construct(props)));
    // Return
    endTrack("Creating Trainer");
    return this.trainer.get();
  }

  //
  // Evaluation Component
  //
  private Maybe<KBPEvaluator> evaluator = Maybe.Nothing();
  public synchronized KBPEvaluator getEvaluator() {
    // If cached, return
    //noinspection LoopStatementThatDoesntLoop
    for (KBPEvaluator e : this.evaluator) { return e; }
    forceTrack("Creating Evaluator");
    // Create new evaluator
    this.evaluator = Maybe.Just(new KBPEvaluator(props, getIR(), getProcess(), getTrainedClassifier()));
    // Return
    endTrack("Creating Evaluator");
    return this.evaluator.get();
  }

  //
  // Evaluation Component
  //
  private Maybe<KBPSlotValidator> validator = Maybe.Nothing();
  public synchronized KBPSlotValidator getSlotValidator() {
    // If cached, return
    //noinspection LoopStatementThatDoesntLoop
    for (KBPSlotValidator e : this.validator) { return e; }
    forceTrack("Creating Validator");
    // Create new evaluator
    this.validator = Maybe.Just(new KBPSlotValidator(props, getIR(), getProcess(), getTrainedClassifier()));
    // Return
    endTrack("Creating Validator");
    return this.validator.get();
  }

  /**
   * A utility method for various common tasks one may wish to perform with
   * a slotfilling system, but which are not part of the core functionaility and in general
   * don't depend on the KBP infrastructure (for the latter, calling methods in the various
   * components is preferred.
   *
   * @return A utility class from which many useful methods can be called.
   */
//  public SlotfillingTasks go() { return new SlotfillingTasks(this); }


  /**
   * The main operation to do when calling SlotfillingSystem
   */
  public static enum RunMode { TRAIN_ONLY, EVALUATE_ONLY, TRAIN_AND_EVALUATE, VALIDATE, DO_NOTHING }

  /**
   * A main router to various modes of running the program
   * @param mode The mode to run the program in
   * @param props The properties to run the program with
   * @throws Exception If something goes wrong
   */
  public static void runProgram(RunMode mode, Properties props) throws Exception {
    SlotfillingSystem instance = new SlotfillingSystem(props);

    logger.log(FORCE, BOLD, BLUE, "run mode: " + mode);
    boolean evaluate = true;
    switch(mode){
      case TRAIN_ONLY:
        evaluate = false;
      case TRAIN_AND_EVALUATE:
        instance.getTrainer().run();
      case EVALUATE_ONLY:
        if (evaluate) {
          instance.classifier = Maybe.Nothing(); // clear old classifier
          instance.getEvaluator().run();
        }
        break;
      case VALIDATE:
        instance.getSlotValidator().run();
        break;
      default:
        logger.fatal("Unknown run mode: " + mode);
    }
  }

  /**
   * A central method which takes command line arguments, and starts a program.
   * This method handles parsing the command line arguments, and setting the options in Props,
   * and any options from calling classes (working up the stack trace).
   * @param toRun The function to run, containing the implementation of the program
   * @param props The properties file to run the system with.
   */
  public static void exec(final Function<Properties, Object> toRun, final Properties props) {
    // Set options classes
    StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();
    Execution.optionClasses = new Class<?>[stackTrace.length +1];
    Execution.optionClasses[0] = Props.class;
    for (int i=0; i<stackTrace.length; ++i) {
      try {
        Execution.optionClasses[i+1] = Class.forName(stackTrace[i].getClassName());
      } catch (ClassNotFoundException e) { logger.fatal(e); }
    }
    // Start Program
    Execution.exec(new Runnable() { @Override public void run() {
      Props.initializeAndValidate();
      toRun.apply(props);
      try {
		SlotfillingSystem.runProgram(SlotfillingSystem.RunMode.EVALUATE_ONLY,props);
	} catch (Exception e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}
    } }, props);
  }

}
