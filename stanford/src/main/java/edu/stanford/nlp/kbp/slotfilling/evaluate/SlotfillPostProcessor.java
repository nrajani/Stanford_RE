package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A class to
 *   (1) filter impossible slot fills (given other slot fills), and
 *   (2) add in likely true slot fills.
 *
 * @author Gabor Angeli
 */
public abstract class SlotfillPostProcessor {


  /**
   * Post process an entire answer set. By default, this post-processes every entry in the result set.
   * This should happen *BEFORE* provenance finding.
   * @param candidateAnswers The candidate answer set
   * @return A new candidate answer set, corresponding to the post-processing..
   */
  public Map<KBPEntity, List<KBPSlotFill>> postProcess(Map<KBPEntity, List<KBPSlotFill>> candidateAnswers, GoldResponseSet checklist) {
    Map<KBPEntity, List<KBPSlotFill>> filtered = new HashMap<KBPEntity, List<KBPSlotFill>>();
    for (Map.Entry<KBPEntity, List<KBPSlotFill>> entry : candidateAnswers.entrySet()) {
      filtered.put(entry.getKey(), postProcess(entry.getKey(), entry.getValue(), checklist));
    }
    return filtered;
  }

  public final Map<KBPEntity, List<KBPSlotFill>> postProcess(Map<KBPEntity, List<KBPSlotFill>> candidateAnswers) {
    return postProcess(candidateAnswers, GoldResponseSet.empty());
  }

  /**
   * Given an entity and a set of candidate slot fills, return a new list of slot fills
   * with the 'filters' applied as specified by this class.
   * This should happen *BEFORE* provenance finding.
   * @param pivot The KBP entity to fill slots for.
   * @param slotFills The slots being filled.
   * @return A new list corresponding to the result of the post processing task.
   */
  protected abstract List<KBPSlotFill> postProcess(KBPEntity pivot, List<KBPSlotFill> slotFills, GoldResponseSet checklist);

  protected final List<KBPSlotFill> postProcess(KBPEntity pivot, List<KBPSlotFill> slotFills) {
    return postProcess(pivot, slotFills, GoldResponseSet.empty());

  }

  /**
   * Chain post-processing tasks together
   * @param alsoProcess The additional process to run
   * @return A processor which runs the tasks in this processor, and then also the ones in the argument.
   */
  public SlotfillPostProcessor and(final SlotfillPostProcessor alsoProcess) {
    final SlotfillPostProcessor outer = this;
    return new SlotfillPostProcessor() {
      @Override
      protected List<KBPSlotFill> postProcess(KBPEntity pivot, List<KBPSlotFill> slotFills, GoldResponseSet checklist) {
        List<KBPSlotFill> interm = outer.postProcess(pivot, slotFills, checklist);
        return alsoProcess.postProcess(pivot, interm);
      }
    };
  }

  public static SlotfillPostProcessor all(SlotfillPostProcessor... procs) {
    SlotfillPostProcessor proc = procs[0];
    for (SlotfillPostProcessor proc1 : procs) {
      if (proc1 != null) { proc = proc.and(proc1); }
    }
    return proc;
  }

  /**
   * A set of unary filters which do not require external knowledge (e.g., IR) and do not require nonlocal
   * information about other candidate slot fills.
   * Note that these only encompass the filtering processors; slot fills are not intended to be rewritten
   * as a result of running this processor
   */
  public static final SlotfillPostProcessor unaryFilters = SlotfillPostProcessor.all(
      // Singleton consistency
      new HeuristicSlotfillPostProcessors.FilterUnrelatedURL(), // This should be early in the list, as it rewrites stuff
      new HeuristicSlotfillPostProcessors.RespectRelationTypes(),
      new HeuristicSlotfillPostProcessors.FilterIgnoredSlots(),
      new HeuristicSlotfillPostProcessors.SanityCheckFilter(),
      new HeuristicSlotfillPostProcessors.ConformToGuidelinesFilter(),
      Props.TEST_CONSISTENCY_WORLDKNOWLEDGE_DIR.exists()
          ? WorldKnowledgePostProcessor.singleton(Props.TEST_CONSISTENCY_WORLDKNOWLEDGE_DIR).configure(true, false, false, false)
          : null
      );

  /**
   * A set of rewrite rules which do not require external knowledge (e.g., IR) and do not require nonlocal
   * information about other candidate slot fills.
   * These are generally quick fixes for common mistakes, or necessary rewrites to fit the task specification
   * (e.g., the slot fill should be the canonical mention).
   */
  public static final SlotfillPostProcessor unaryRewrites = SlotfillPostProcessor.all(
      // Rewrites (only really needed in phase 1)
      new HeuristicSlotfillPostProcessors.CanonicalMentionRewrite(),
      new HeuristicSlotfillPostProcessors.ExpandToMaximalPhraseRewrite(),
      new HeuristicSlotfillPostProcessors.TopEmployeeRewrite(),
      new HeuristicSlotfillPostProcessors.BornInRewrite()
  );

  /**
   * <p>The full set of unary factors, without dependence on any external resources or
   * global information from other candidate slot fills.
   * This includes the unary filters and the unary rewrites.</p>
   * @see SlotfillPostProcessor#unaryFilters
   * @see SlotfillPostProcessor#unaryRewrites
   */
  public static final SlotfillPostProcessor unary = SlotfillPostProcessor.all(
      unaryFilters, unaryRewrites
  );

  /**
   * <p>The full set of unary factors, with certain filters requiring access to the IR component.
   * This includes the unary filters and the unary rewrites.</p>
  * <p>This is the first pass in the standard evaluation framework, filtering out (and/or rewriting)
  * impossible slots.</p>
   * @see SlotfillPostProcessor#unaryFilters
   * @see SlotfillPostProcessor#unaryRewrites
   * @see HeuristicSlotfillPostProcessors.FilterAlreadyKnownSlots
  */
  public static SlotfillPostProcessor unary(KBPIR ir) {
    if (Props.KBP_YEAR == Props.YEAR.KBP2009 || Props.KBP_YEAR == Props.YEAR.KBP2010 || Props.KBP_YEAR == Props.YEAR.KBP2011 ||
        Props.KBP_YEAR == Props.YEAR.KBP2012) {
      return SlotfillPostProcessor.all(unaryRewrites, unaryFilters, new HeuristicSlotfillPostProcessors.FilterAlreadyKnownSlots(ir));
    } else {
      return SlotfillPostProcessor.all(unaryRewrites, unaryFilters);
    }
  }

  /**
   * Only filters which require global information, excluding the unary filters.
   */
  public static final SlotfillPostProcessor globalOnly = SlotfillPostProcessor.all(
      // Larger Factors
      new HeuristicSlotfillPostProcessors.NoDuplicates(),
      new HeuristicSlotfillPostProcessors.NoDuplicatesApproximate(),
      new HeuristicSlotfillPostProcessors.RespectDeclaredIncompatibilities(),
      new HeuristicSlotfillPostProcessors.MitigateLocOfDeath(),
      Props.TEST_CONSISTENCY_WORLDKNOWLEDGE_DIR.exists()
        ? WorldKnowledgePostProcessor.singleton(Props.TEST_CONSISTENCY_WORLDKNOWLEDGE_DIR).configure(true, true, true, true)
        : null,
      new HeuristicSlotfillPostProcessors.DuplicateRelationOnlyInListRelations()
  );

  /**
   * Only filters which require global information, excluding the unary filters.
   */
  public static SlotfillPostProcessor globalOnly(KBPIR ir) { return SlotfillPostProcessor.all(
      // Larger Factors
      new HeuristicSlotfillPostProcessors.NoDuplicates(),
      new HeuristicSlotfillPostProcessors.NoDuplicatesApproximate(ir),
      new HeuristicSlotfillPostProcessors.RespectDeclaredIncompatibilities(),
      new HeuristicSlotfillPostProcessors.MitigateLocOfDeath(),
      Props.TEST_CONSISTENCY_WORLDKNOWLEDGE_DIR.exists()
          ? WorldKnowledgePostProcessor.singleton(Props.TEST_CONSISTENCY_WORLDKNOWLEDGE_DIR).configure(true, true, true, true)
          : null,
      new HeuristicSlotfillPostProcessors.DuplicateRelationOnlyInListRelations()
  ); }

  /**
   * All filters, save those requiring external information (e.g., IR)
   */
  public static final SlotfillPostProcessor global = SlotfillPostProcessor.all(unary, globalOnly);


  /**
   * <p>All filters we know about, including those requiring access to external components.
   *    This is the combination of the unary and globalOnly processors.</p>
   * <p>This is the second pass in the standard evaluation framework, enforcing global consistency</p>
   * @see SlotfillPostProcessor#unary(edu.stanford.nlp.kbp.slotfilling.ir.KBPIR)
   * @see SlotfillPostProcessor#globalOnly
   * @see HeuristicSlotfillPostProcessors.FilterAlreadyKnownSlots
   */
  public static SlotfillPostProcessor global(KBPIR ir) {
    return SlotfillPostProcessor.all(unary(ir), globalOnly(ir));
  }

  public static SlotfillPostProcessor validators = SlotfillPostProcessor.all(
      unaryFilters,
      new HeuristicSlotfillPostProcessors.RespectDeclaredIncompatibilities(),
      new HeuristicSlotfillPostProcessors.MitigateLocOfDeath(),
      Props.TEST_CONSISTENCY_WORLDKNOWLEDGE_DIR.exists()
          ? WorldKnowledgePostProcessor.singleton(Props.TEST_CONSISTENCY_WORLDKNOWLEDGE_DIR).configure(true, true, true, true)
          : null,
      new HeuristicSlotfillPostProcessors.DuplicateRelationOnlyInListRelations()
  );
}
