package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.math.ArrayMath;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Function;
import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.Pair;

import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * A class for simple heuristic consistency checks on slot fills, in the same vein
 * as the old system.
 *
 * @author Gabor Angeli
 */
public abstract class HeuristicSlotfillPostProcessor extends SlotfillPostProcessor {

  /**
   * A class to manage modifications to the active slots vector.
   * On creation, it will mutate the passed slotsActive array;
   * when the modification is complete, call restoreAndReturn,
   * which passes on the passed return value, but also restores the
   * state of the slotsActive vector.
   */
  private static class GibbsState {
    private final boolean[] slotsActive;
    private final boolean savedDeactivatedValue;
    private final boolean savedActivatedValue;
    private final boolean savedActivated2Value;
    private final int toDeactivate;
    private final int toActivate;
    private final int toActivate2;

    private boolean isRestored = false;

    public GibbsState(boolean[] slotsActive, int toDeactivate, int toActivate, int toActivate2) {
      this.slotsActive = slotsActive;
      this.toDeactivate = toDeactivate;
      this.toActivate = toActivate;
      this.toActivate2 = toActivate2;
      this.savedDeactivatedValue = slotsActive[toDeactivate];
      this.savedActivatedValue = slotsActive[toActivate];
      this.savedActivated2Value = slotsActive[toActivate2];
      slotsActive[toDeactivate] = false;
      slotsActive[toActivate] = true;
      slotsActive[toActivate2] = true;
    }

    public <E> E restoreAndReturn(E valueToReturn) {
      if (isRestored) {
        throw new IllegalStateException("Using a Gibbs state twice!");
      }
      slotsActive[toDeactivate] = savedDeactivatedValue;
      slotsActive[toActivate] = savedActivatedValue;
      slotsActive[toActivate2] = savedActivated2Value;
      isRestored = true;
      return valueToReturn;
    }
  }

  private boolean blockGibbsCanTransition(KBPEntity pivot, KBPSlotFill[] slotFills, GibbsState state) {
    return state.restoreAndReturn(isConsistent(pivot, slotFills, state.slotsActive));
  }

  private boolean isConsistent(KBPEntity pivot, KBPSlotFill[] slotFills, boolean[] slotsActive) {
    // -- Singleton Consistency
    for (int i = 0; i < slotFills.length; ++i) {
      if (slotsActive[i] && !isValidSlotAndRewrite(pivot, slotFills[i]).isDefined()) { return false; }
    }

    // -- Pairwise Consistency
    for (int i = 0; i < slotFills.length; ++i) {
      for (int j = i + 1; j < slotFills.length; ++j) {
        if (slotsActive[i] && slotsActive[j] &&
            !pairwiseKeepLowerScoringFill(pivot, slotFills[i], slotFills[j])) {
          return false;
        }
      }
    }

    // -- Hold-one-out Consistency
    // (create set)
    IdentityHashSet<KBPSlotFill> others = new IdentityHashSet<KBPSlotFill>();
    for (int i = 0; i < slotFills.length; ++i) {
      if (slotsActive[i]) { others.add(slotFills[i]); }
    }
    // (check consistency for all active elements)
    for (int i = 0; i < slotFills.length; ++i) {
      if (slotsActive[i]) {
        others.remove(slotFills[i]);
        if (!leaveOneOutKeepHeldOutSlot(pivot, others, slotFills[i])) { return false; }
        others.add(slotFills[i]);
      }
    }

    // -- Everything Passes
    return true;
  }

  @Deprecated
  private List<KBPSlotFill> heuristicNonlocalDependencies(KBPEntity pivot, List<KBPSlotFill> slotFills, GoldResponseSet checklist) {
    List<KBPSlotFill> filteredSlots = slotFills;

    // -- Pairwise Consistency
    // (sort slots)
    KBPSlotFill[] sortedSlots = filteredSlots.toArray(new KBPSlotFill[filteredSlots.size()]);
    Arrays.sort(sortedSlots);
    boolean[] deadSlots = new boolean[sortedSlots.length];
    List<KBPSlotFill> pairwiseFiltered = new LinkedList<KBPSlotFill>();
    // (kill off slots -- lowest score on the block first)
    for (int betterI = 0; betterI<sortedSlots.length; ++betterI) {
      if (!deadSlots[betterI]) {
        for (int worseI = sortedSlots.length - 1; worseI > betterI; --worseI) {
          if (!deadSlots[worseI] && !pairwiseKeepLowerScoringFill(pivot, sortedSlots[betterI], sortedSlots[worseI])) {
            debug("removing slot: " + sortedSlots[worseI].fillToString() + " (incompatible with " + sortedSlots[betterI].fillToString() + ")");
            checklist.discardInconsistent(sortedSlots[worseI]);
            deadSlots[worseI] = true;
          }
        }
      }
    }
    // (update filtered slots)
    filteredSlots = new ArrayList<KBPSlotFill>();
    for (int i = 0; i < deadSlots.length; ++i) {
      if (!deadSlots[i]) {
        filteredSlots.add(sortedSlots[i]);
      } else {
        pairwiseFiltered.add(sortedSlots[i]);
      }
    }

    // -- Leave-One-Out Consistency
    // (create data structures)
    Collections.sort(filteredSlots);
    IdentityHashSet<KBPSlotFill> otherSlots = new IdentityHashSet<KBPSlotFill>();
    otherSlots.addAll(filteredSlots);
    // (kill off slots -- lowest priority on the block first)
    boolean[] inconsistentSlots = new boolean[filteredSlots.size()];
    for( int enPrise = filteredSlots.size() - 1; enPrise >= 0; --enPrise) {
      otherSlots.remove(filteredSlots.get(enPrise));
      if (!leaveOneOutKeepHeldOutSlot(pivot, otherSlots, filteredSlots.get(enPrise))) {
        debug("removing slot: " + filteredSlots.get(enPrise).fillToString() + " (incompatible with existing slots)");
        inconsistentSlots[enPrise] = true;
        checklist.discardInconsistent(filteredSlots.get(enPrise));
      } else {
        otherSlots.add(filteredSlots.get(enPrise));
      }
    }
    // (update filtered slots)
    List<KBPSlotFill> newFilteredSlots = new ArrayList<KBPSlotFill>();
    for (int i = 0; i < inconsistentSlots.length; ++i) {
      if (!inconsistentSlots[i]) {
        newFilteredSlots.add(filteredSlots.get(i));
      }
    }
    filteredSlots = newFilteredSlots;

    // -- Try to add back pairwise consistency
    // (create new restored list)
    List<KBPSlotFill> restoredSlots = new ArrayList<KBPSlotFill>();
    for (KBPSlotFill fill : filteredSlots) { restoredSlots.add(fill); }
    for (KBPSlotFill removedSlot : pairwiseFiltered) {
      boolean allow = true;
      for (KBPSlotFill existing : restoredSlots) {
        if (!this.isValidSlotAndRewrite(pivot, removedSlot).isDefined() ||
            !this.pairwiseKeepLowerScoringFill(pivot, existing, removedSlot)) {
          allow = false;
        }
      }
      if (allow) {
        debug("restoring slot: " + removedSlot + " (pairwise consistent again)");
        checklist.undoDiscardInconsistent(removedSlot);
        restoredSlots.add(removedSlot);
      }
    }
    filteredSlots = restoredSlots;

    return filteredSlots;
  }


  private int greedyEnableSlotsInPlace(KBPEntity pivot, KBPSlotFill[] sortedSlots, boolean[] slotsActive) {
    int slotsEnabled = 0;
    for (int i = 0; i < sortedSlots.length; ++i) {
      if (blockGibbsCanTransition(pivot, sortedSlots, new GibbsState(slotsActive, i, i, i))) {
        debug("greedily enabling " + sortedSlots[i]);
        slotsActive[i] = true;
        slotsEnabled += 1;
      } else {
        assert !slotsActive[i];
      }
    }
    return slotsEnabled;
  }


  private List<KBPSlotFill> filterStep(KBPEntity pivot, List<KBPSlotFill> slotFills, GoldResponseSet checklist) {
    startTrack("Gibbs Filtering");
    List<KBPSlotFill> filteredSlots = slotFills;

    // -- Singleton Consistency
    List<KBPSlotFill> withBlatantViolationsFiltered = new ArrayList<KBPSlotFill>();
    for (KBPSlotFill slotFill : filteredSlots) {
      Maybe<KBPSlotFill> maybeRewritten = isValidSlotAndRewrite(pivot, slotFill);
      for (KBPSlotFill rewritten : maybeRewritten) {
        if (!rewritten.equals(slotFill)) {
          checklist.discardRewritten(slotFill);
          checklist.registerResponse(rewritten);
        }
        withBlatantViolationsFiltered.add(Props.TEST_CONSISTENCY_REWRITE ? rewritten : slotFill);
      }
      if (!maybeRewritten.isDefined()) {
        debug("removing slot: " + slotFill.fillToString() + " (impossible slot)");
        checklist.discardInconsistent(slotFill);
      }
    }
    filteredSlots = withBlatantViolationsFiltered;

    // -- Nonlocal consistency
    KBPSlotFill[] sortedSlots = filteredSlots.toArray(new KBPSlotFill[filteredSlots.size()]);
    Arrays.sort(sortedSlots);
    boolean[] slotsActive = new boolean[sortedSlots.length];
    int slotsEnabled = greedyEnableSlotsInPlace(pivot, sortedSlots, slotsActive);
    // (pass 1: greedy)
    debug("[begin gibbs] enabled " + slotsEnabled + " slots (of " + sortedSlots.length + ")");
    assert (isConsistent(pivot, sortedSlots, slotsActive));
    // (pass 2: pairwise hops)
    if (Props.TEST_CONSISTENCY_GIBBSOBJECTIVE != Props.GibbsObjective.TOP) {
      Function<Pair<boolean[], KBPSlotFill[]>, Double> objectiveFn = getObjective(Props.TEST_CONSISTENCY_GIBBSOBJECTIVE);

      // Sample and greedy hill climb
      //   variables
      boolean[] argmax = new boolean[slotsActive.length];
      System.arraycopy(slotsActive, 0, argmax, 0, slotsActive.length);
      double max = objectiveFn.apply(Pair.makePair(slotsActive, sortedSlots));
      Random rand = new Random(42);
      int[] enableOrder = CollectionUtils.seq(slotsActive.length);
      log("initial objective: " + max);
      //   sample
      for (int i = 0; i < Props.TEST_CONSISTENCY_MIXINGTIME; ++i) {
        Arrays.fill(slotsActive, false);
        ArrayMath.shuffle(enableOrder, rand);
        for (int toEnable : enableOrder) {
          if (blockGibbsCanTransition(pivot, sortedSlots, new GibbsState(slotsActive, toEnable, toEnable, toEnable))) {
            slotsActive[toEnable] = true;
          }
        }
        double newObjective = objectiveFn.apply(Pair.makePair(slotsActive, sortedSlots));
        if (newObjective > max) {
          max = newObjective;
          System.arraycopy(slotsActive, 0, argmax, 0, slotsActive.length);
          log("found higher objective: " + max);
        }
      }
      //   save result
      slotsActive = argmax;
      slotsEnabled = 0;
      for (boolean active : slotsActive) { slotsEnabled += active ? 1 : 0; }
      debug("[end gibbs] enabled " + slotsEnabled + " slots (of " + sortedSlots.length + ") with objective " + max);

      /*
      // ^^ Alternative to above ^^
      // Gibbs Hill climbing: while( not converged ): disable one and enable two slots.
      boolean converged = false;
      double objective = objectiveFn.apply(Pair.makePair(slotsActive, sortedSlots));
      while (!converged) {
        log("hill climbing...");
        converged = true;
        for (int off = 0; off < slotsActive.length; ++off) {
          for (int on = 0; on < slotsActive.length; ++on) {
            for (int on2 = 0; on2 < slotsActive.length; ++on2) {
              double candidateObjective = new GibbsState(slotsActive, off, on, on2).restoreAndReturn(objectiveFn.apply(Pair.makePair(slotsActive, sortedSlots)));
              if (candidateObjective > objective &&
                  blockGibbsCanTransition(pivot, sortedSlots, new GibbsState(slotsActive, off, on, on2))) {
                log("replacing " + sortedSlots[off] + " with " + sortedSlots[on] + (on == on2 ? "" : " and " + sortedSlots[on2]));
                slotsActive[off] = false;
                slotsActive[on] = true;
                slotsActive[on2] = true;
                converged = false;
                objective = candidateObjective;
              }
            }
          }
        }
      }
      log("converged.");
      */
    }
    assert (isConsistent(pivot, sortedSlots, slotsActive));
    // (copy list)
    List<KBPSlotFill> withGlobalConsistency = new ArrayList<KBPSlotFill>();
    for (int i = 0; i < sortedSlots.length; ++i) {
      if (slotsActive[i]) {
        withGlobalConsistency.add(sortedSlots[i]);
      } else {
        checklist.discardInconsistent(sortedSlots[i]);
      }
    }
    filteredSlots = withGlobalConsistency;

    // -- Return
    endTrack("Gibbs Filtering");
    return filteredSlots;

  }

  private Function<Pair<boolean[],KBPSlotFill[]>,Double> getObjective(Props.GibbsObjective type) {
    switch (type) {
      // Fill as many of te top slots as possible
      case TOP:
        return new Function<Pair<boolean[], KBPSlotFill[]>, Double>() {
          @Override
          public Double apply(Pair<boolean[], KBPSlotFill[]> in) {
            throw new IllegalStateException("No well defined objective for GibbsObjective.TOP");
          }
        };
      // Optimize for the maximum sum score of the slots filled
      case SUM:
        return new Function<Pair<boolean[], KBPSlotFill[]>, Double>() {
          @Override
          public Double apply(Pair<boolean[], KBPSlotFill[]> in) {
            boolean[] mask = in.first;
            KBPSlotFill[] fills = in.second;
            double sum = 0.0;
            for (int i = 0; i < mask.length; ++i) {
              if (mask[i]) { sum += fills[i].score.getOrElse(0.0); }
            }
            return sum;
          }
        };
      default:
        throw new IllegalArgumentException("Objective type not implemented: " + type);
    }
  }

  private List<KBPSlotFill> generateStep(KBPEntity pivot, List<KBPSlotFill> slotFills, GoldResponseSet checklist) {
    List<KBPSlotFill> filteredSlots = slotFills;

    // -- Direct Entailment
    List<KBPSlotFill> newSlotFills = new ArrayList<KBPSlotFill>();
    Queue<KBPSlotFill> antecedents = new LinkedList<KBPSlotFill>(filteredSlots);
    Set<KBPSlotFill> seenAsAntecedent = new HashSet<KBPSlotFill>();
    // (run inference)
    while(!antecedents.isEmpty()) {
      // (poll)
      KBPSlotFill antecedent = antecedents.poll();
      seenAsAntecedent.add(antecedent);
      newSlotFills.add(antecedent);
      // (add consequents)
      for (Map.Entry<Pair<RelationType, String>, Double> entry : entailsDirectly(pivot, antecedent).entrySet()) {
        KBPSlotFill consequent = KBPNew.from(antecedent.key).slotValue(entry.getKey().second).rel(entry.getKey().first).score(entry.getValue()).KBPSlotFill();
        if (!seenAsAntecedent.contains(consequent)) {
          debug("Adding inferred slot" + consequent.fillToString() + " (inferred from " + antecedent.fillToString() + ")");
          antecedents.add(consequent);
        }
      }
    }
    // (update filtered slots0
    filteredSlots = newSlotFills;

    // -- Return
    return filteredSlots;
  }

  /** @inheritDoc */
  @Override
  public SlotfillPostProcessor and(final SlotfillPostProcessor alsoProcess) {
    if (alsoProcess instanceof HeuristicSlotfillPostProcessor) {
      final HeuristicSlotfillPostProcessor hpp = (HeuristicSlotfillPostProcessor) alsoProcess;
      final HeuristicSlotfillPostProcessor outer = this;
      return new HeuristicSlotfillPostProcessor() {
        @Override
        public Maybe<KBPSlotFill> isValidSlotAndRewrite(KBPEntity pivot, KBPSlotFill candidate) {
//          if ( !hpp.isValidSlotAndRewrite(pivot, candidate) ) {
//            log("Filtered (" + hpp.getClass().getSimpleName() + "): " + candidate);
//          }
          Maybe<KBPSlotFill> outerValid = outer.isValidSlotAndRewrite(pivot, candidate);
          if (outerValid.isDefined()) {
            Maybe<KBPSlotFill> hppValid = hpp.isValidSlotAndRewrite(pivot, outerValid.get());
            if (hppValid.isDefined()) {
              return hppValid;
            } else {
              return Maybe.Nothing();

            }
          } else {
            return Maybe.Nothing();
          }
        }
        @Override
        public boolean pairwiseKeepLowerScoringFill(KBPEntity pivot, KBPSlotFill higherScoring, KBPSlotFill lowerScoring) {
//          if ( !hpp.pairwiseKeepLowerScoringFill(pivot, higherScoring, lowerScoring) ) {
//            log("Filtered (" + hpp.getClass().getSimpleName() + "): " + lowerScoring);
//          }
          return outer.pairwiseKeepLowerScoringFill(pivot, higherScoring, lowerScoring) &&
              hpp.pairwiseKeepLowerScoringFill(pivot, higherScoring, lowerScoring);
        }
        @Override
        public boolean leaveOneOutKeepHeldOutSlot(KBPEntity pivot, IdentityHashSet<KBPSlotFill> others, KBPSlotFill candidate) {
//          if ( !hpp.leaveOneOutKeepHeldOutSlot(pivot, others, candidate) ) {
//            log("Filtered (" + hpp.getClass().getSimpleName() + "): " + candidate);
//          }
          return outer.leaveOneOutKeepHeldOutSlot(pivot, others, candidate) &&
              hpp.leaveOneOutKeepHeldOutSlot(pivot, others, candidate);
        }
        @Override
        public Counter<Pair<RelationType, String>> entailsDirectly(KBPEntity pivot, KBPSlotFill antecedent) {
          Counter<Pair<RelationType, String>> counts = outer.entailsDirectly(pivot, antecedent);
          counts.addAll(hpp.entailsDirectly(pivot, antecedent));
          return counts;
        }
      };
    } else {
      return super.and(alsoProcess);
    }
  }

  /**
   * @inheritDoc
   */
  @Override
  protected List<KBPSlotFill> postProcess(KBPEntity pivot, List<KBPSlotFill> slotFills, GoldResponseSet checklist) {
    List<KBPSlotFill> filter1 = filterStep(pivot, slotFills, checklist);
    List<KBPSlotFill> inference = generateStep(pivot, filter1, checklist);
    return filterStep(pivot, inference, checklist);
  }

  /**
   * Independently of other slot fills, determine if this is even a reasonable slot fill,
   * and if it is, optionally rewrite it.
   * @param pivot The entity we are filling slots for
   * @param candidate The candidate slot fill
   * @return either Maybe.Nothing to ignore the slot, or a rewriten slot.
   */
  public abstract Maybe<KBPSlotFill> isValidSlotAndRewrite(KBPEntity pivot, KBPSlotFill candidate);

  /**
   * Filter a slot if it is directly inconsistent with another slot
   * @param pivot The entity we are filling slots for
   * @param higherScoring The higher scoring slot (this one always survives!)
   * @param lowerScoring The lower scoring slot (this is the one that may die)
   * @return True if the lower scoring slot should also be kept (this is the usual case)
   */
  public abstract boolean pairwiseKeepLowerScoringFill(KBPEntity pivot, KBPSlotFill higherScoring, KBPSlotFill lowerScoring);

  /**
   * Filter a slot if it is directly inconsistent with the set of other slots.
   * @param pivot The entity we are filling slots for
   * @param others The set of slots which are already filled.
   * @param candidate The slot which we are proposing to add
   * @return True if the candidate slot should also be kept (this is the usual case)
   */
  public abstract boolean leaveOneOutKeepHeldOutSlot(KBPEntity pivot, IdentityHashSet<KBPSlotFill> others, KBPSlotFill candidate);

  /**
   * A simple direct inference -- if a slot is filled, another slot may well want to be filled as well.
   * @param pivot The entity we are filling slots for
   * @param antecedent The antecedent of the inference.
   * @return A collection of implied slot fills. The keys to the counter are (relation, slot-fill) pairs;
   *         the counts are the scores of the inferred relations.
   */
  public abstract Counter<Pair<RelationType, String>> entailsDirectly(KBPEntity pivot, KBPSlotFill antecedent);

  /**
   * A default implementation (effectively a NOOP) so that selective methods can be overwritten
   */
  public static class Default extends HeuristicSlotfillPostProcessor {
    @Override
    public Maybe<KBPSlotFill> isValidSlotAndRewrite(KBPEntity pivot, KBPSlotFill candidate) {
      return Maybe.Just(candidate);
    }
    @Override
    public boolean pairwiseKeepLowerScoringFill(KBPEntity pivot, KBPSlotFill higherScoring, KBPSlotFill lowerScoring) {
      return true;
    }
    @Override
    public boolean leaveOneOutKeepHeldOutSlot(KBPEntity pivot, IdentityHashSet<KBPSlotFill> others, KBPSlotFill candidate) {
      return true;
    }
    @Override
    public Counter<Pair<RelationType, String>> entailsDirectly(KBPEntity pivot, KBPSlotFill antecedent) {
      return new ClassicCounter<Pair<RelationType, String>>();
    }
  }

}
