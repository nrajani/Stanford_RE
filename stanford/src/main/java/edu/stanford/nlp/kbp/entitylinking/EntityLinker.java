package edu.stanford.nlp.kbp.entitylinking;

import edu.stanford.nlp.kbp.entitylinking.classify.namematcher.RuleBasedNameMatcher;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.util.Function;
import edu.stanford.nlp.util.Pair;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

/**
 * KBP Slotfilling's view into Entity Linking.
 *
 * @author Gabor Angeli
 */
public abstract class EntityLinker implements Function<Pair<EntityContext, EntityContext>, Boolean> {

  public boolean sameEntity(EntityContext a, EntityContext b) {
    // Try conventional entity linking
    for (String id1 : link(a)) {
      //noinspection LoopStatementThatDoesntLoop
      for (String id2 : link(b)) {
        return id1.equals(id2);
      }
    }

    // Check some hard constraints
    // Enforce same type
    if (a.entity.type != b.entity.type) { return false; }
    // Check for acronym match
    if (RuleBasedNameMatcher.isAcronym(a.tokens(), b.tokens())) { return true; }

    // Backoff to the implementing model
    return sameEntityWithoutLinking(a, b);
  }

  @Override
  public Boolean apply(Pair<EntityContext, EntityContext> in) {
    return sameEntity(in.first, in.second);
  }

  /**
   * Return a unique id for this entity, if one is available.
   * @param context The entity to link, along with all known context.
   * @return The id of the entity, if one could be found.
   */
  public abstract Maybe<String> link(EntityContext context);

  /**
   * If the entity could not be linked, try to determine if two entities are the same anyways.
   * @param entityOne The first entity, with its context.
   * @param entityTwo The second entity, with its context.
   * @return True if the two entities are the same entity in reality.
   */
  protected abstract boolean sameEntityWithoutLinking(EntityContext entityOne, EntityContext entityTwo);




  /**
   * A very dumb entity linker that checks for hard constraints, in addition to anything checked
   * by the {@link EntityLinker} base class.
   */
  public static class HardConstraintsEntityLinker extends EntityLinker {
    @Override
    public Maybe<String> link(EntityContext context) {
      return Maybe.Nothing();
    }
    @Override
    protected boolean sameEntityWithoutLinking(EntityContext entityOne, EntityContext entityTwo) {
      return entityOne.entity.name.equals(entityTwo.entity.name);
    }
  }

  /**
   * The hacky baseline Gabor tuned for KBP2013.
   * Please do better than me!
   */
  public static class GaborsHackyBaseline extends EntityLinker {
    @Override
    public Maybe<String> link(EntityContext context) {
      return Maybe.Nothing();
    }

    @SuppressWarnings("RedundantIfStatement")
    @Override
    protected boolean sameEntityWithoutLinking(EntityContext entityOne, EntityContext entityTwo) {
      NERTag type = entityOne.entity.type;
      double matchScore = approximateEntityMatchScore(entityOne.entity.name, entityTwo.entity.name);

      // Some simple cases
      if( matchScore == 1.0 ) { return true; }
      if( matchScore < 0.34 ) { return false; }
      if (type == NERTag.PERSON && matchScore > 0.49) { return true; }
      if (type == NERTag.ORGANIZATION && matchScore > 0.79) { return true; }

      // See if we can use properties
      for (Collection<KBPSlotFill> fillsOne : entityOne.properties) {
        for (Collection<KBPSlotFill> fillsTwo : entityTwo.properties) {
          Set<Pair<String, String>> propsOne = new HashSet<Pair<String, String>>();
          for (KBPSlotFill fill : fillsOne) { propsOne.add(Pair.makePair(fill.key.relationName, fill.key.slotValue)); }
          Set<Pair<String, String>> propsTwo = new HashSet<Pair<String, String>>();
          for (KBPSlotFill fill : fillsTwo) { propsTwo.add(Pair.makePair(fill.key.relationName, fill.key.slotValue)); }
          int overlap = CollectionUtils.allOverlaps(propsOne, propsTwo).size();
          int minSize = Math.min(propsOne.size(), propsTwo.size());
          if (minSize > 1 && overlap >= minSize / 2) { return true; }
        }
      }

      // Other cases to be careful about length of stuff.
      String higherGloss = entityOne.entity.name;
      String lowerGloss = entityTwo.entity.name;

      // Partial names
      String[] higherToks = higherGloss.split("\\s+");
      String[] lowerToks = lowerGloss.split("\\s+");

      // Set the minimum overlap to consider the slots the same (dependent on the relation)
      if ( Math.min(higherToks.length, lowerToks.length) > 1 ) {
        return true;  // case insensitive token overlap of > cutoff
        // When we have that one of the words has length just 1, make sure
        // that they match at the beginning or end
      } else if ( Math.min(higherToks.length, lowerToks.length) == 1 && Math.abs(higherToks.length - lowerToks.length) <= 2 ) {
        return true;  // case insensitive token overlap of > cutoff
      } else {
        return false;
      }
    }

    private static boolean nearExactEntityMatch( String higherGloss, String lowerGloss ) {
      // case: slots have same relation, and that relation isn't an alternate name
      // Filter case sensitive match
      if (higherGloss.equalsIgnoreCase(lowerGloss)) { return true; }
      // Ignore certain characters
      else if (Utils.noSpecialChars(higherGloss).equalsIgnoreCase(Utils.noSpecialChars(lowerGloss))) { return true; }
      return false;
    }

    /**
     * Approximately check if two entities are equivalent.
     * Taken largely from
     * edu.stanford.nlp.kbp.slotfilling.evaluate,HeuristicSlotfillPostProcessors.NoDuplicatesApproximate;
     */
    private static double approximateEntityMatchScore( String higherGloss, String lowerGloss) {
      if( nearExactEntityMatch(higherGloss, lowerGloss) ) return 1.0;

      // Partial names
      String[] higherToks = higherGloss.split("\\s+");
      String[] lowerToks = lowerGloss.split("\\s+");
      // Case: acronyms of each other
      if (RuleBasedNameMatcher.isAcronym(higherToks, lowerToks)) { return 1.0; }

      int match = 0;
      // Get number of matching tokens between the two slot fills
      for (String higherTok : higherToks) {
        String higherTokNoSpecialChars = Utils.noSpecialChars(higherTok);
        boolean doesMatch = false;
        for (String lowerTok : lowerToks) {
          String lowerTokNoSpecialCars = Utils.noSpecialChars(lowerTok);
          if (higherTokNoSpecialChars.equalsIgnoreCase(lowerTokNoSpecialCars) ||  // equal
              higherTokNoSpecialChars.endsWith(lowerTokNoSpecialCars) || higherTokNoSpecialChars.startsWith(lowerTokNoSpecialCars) ||  // substring
              lowerTokNoSpecialCars.endsWith(higherTokNoSpecialChars) || lowerTokNoSpecialCars.startsWith(higherTokNoSpecialChars) ||  // substring the other way
              (Math.min(lowerTokNoSpecialCars.length(), higherTokNoSpecialChars.length()) > 4 && Utils.levenshteinDistance(lowerTokNoSpecialCars, higherTokNoSpecialChars) <= 1)  // edit distance <= 1
              ) {
            // TODO(gabor) recognize acronyms
            doesMatch = true;  // a loose metric of "same token"
          }
        }
        if (doesMatch) { match += 1; }
      }

      return (double) match / ((double) Math.max(higherToks.length, lowerToks.length));
    }

  }



  /**
   * The hacky baseline Gabor made for mining inferential paths, comparing certain properties of
   * the two entities to link.
   */
  public static class GaborsHighPrecisionBaseline extends EntityLinker {
    @Override
    public Maybe<String> link(EntityContext context) { return Maybe.Nothing(); }

    @Override
    protected boolean sameEntityWithoutLinking(EntityContext entityOne, EntityContext entityTwo) {
      // Check name equality
      if (entityOne.entity.name.equals(entityTwo.entity.name)) { return true; }
      for (Collection<KBPSlotFill> props1 : entityOne.properties) {
        // Check alternate names
        for (KBPSlotFill fill1 : props1) {
          if (fill1.key.hasKBPRelation() && fill1.key.kbpRelation().isAlternateName() && fill1.key.slotValue.equals(entityTwo.entity.name)) { return true; }
          if (fill1.key.slotValue.equals(entityOne.entity.name)) { return false; }  // should not have relation with ones self
        }
        //noinspection LoopStatementThatDoesntLoop
        for (Collection<KBPSlotFill> props2 : entityTwo.properties) {
          // Check alternate names
          for (KBPSlotFill fill2 : props1) {
            if (fill2.key.hasKBPRelation() && fill2.key.kbpRelation().isAlternateName() && fill2.key.slotValue.equals(entityOne.entity.name)) { return true; }
            if (fill2.key.slotValue.equals(entityTwo.entity.name)) { return false; }  // should not have relation with ones self
          }
          // Check relation overlap
          Set<String> overlap = commonProperties(props1, props2);
          return overlap.contains(RelationType.PER_DATE_OF_BIRTH.canonicalName) ||
              overlap.contains(RelationType.PER_DATE_OF_DEATH.canonicalName) ||
              overlap.contains(RelationType.ORG_FOUNDED.canonicalName) ||
              (overlap.contains(RelationType.PER_TITLE.canonicalName) && overlap.size() > 1) ||
              (overlap.contains(RelationType.PER_CITY_OF_BIRTH.canonicalName) && overlap.size() > 1) ||
              (overlap.contains(RelationType.PER_CITY_OF_DEATH.canonicalName) && overlap.size() > 1) ||
              (overlap.contains(RelationType.PER_COUNTRY_OF_BIRTH.canonicalName) && overlap.size() > 1) ||
              (overlap.contains(RelationType.PER_COUNTRY_OF_DEATH.canonicalName) && overlap.size() > 1) ||
              (overlap.contains(RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH.canonicalName) && overlap.size() > 1) ||
              (overlap.contains(RelationType.PER_STATE_OR_PROVINCES_OF_DEATH.canonicalName) && overlap.size() > 1) ||
              (overlap.size() == 2 && (props1.size() == 3 || props2.size() == 3)) ||
              overlap.size() > Math.max(2, Math.max(props1.size(), props2.size()) / 2);
        }
      }
      return false;
    }

    private Set<String> commonProperties(Collection<KBPSlotFill> a, Collection<KBPSlotFill> b) {
      Set<String> overlap = new HashSet<String>();
      for (KBPSlotFill fillA : a) {
        for (KBPSlotFill fillB : b) {
          if (fillA.key.relationName.equals(fillB.key.relationName) &&
              fillA.key.slotValue.trim().equals(fillB.key.slotValue.trim())) {
            overlap.add(fillA.key.relationName);
          }
        }
      }
      return overlap;
    }
  }
}
