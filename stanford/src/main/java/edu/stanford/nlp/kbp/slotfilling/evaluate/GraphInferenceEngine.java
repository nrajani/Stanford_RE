package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.util.Function;
import edu.stanford.nlp.util.Generics;
import edu.stanford.nlp.util.Pair;

import java.io.File;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * Inference rules
 */
public class GraphInferenceEngine {

  final List<Rule> rules;
  private Maybe<KBPIR> irComponent;
  Set<String> validPredicates;

  public GraphInferenceEngine(final List<Rule> rules) {
    this.rules = rules;
    this.irComponent = Maybe.Nothing();
    validPredicates = new HashSet<String>();
    // Add all kbpRelations
    for( RelationType kbpReln : RelationType.values() ) {
      validPredicates.add(kbpReln.canonicalName);
    }
    // Add all other symbols that come up
    for ( Rule rule : rules ) {
      for( Predicate pred : rule.heads )
        validPredicates.add(pred.relation);
      validPredicates.add(rule.tail.relation);
    }
  }
  public GraphInferenceEngine(final KBPIR irComponent, final List<Rule> rules) {
    this(rules);
    this.irComponent = Maybe.Just(irComponent);
  }

  public boolean isUsefulRelation(String reln) {
    return validPredicates.contains(reln);
  }

  public static class Predicate {
    final String relation;
    final String arg1;
    final Maybe<String> arg2; // I may not have a second argument.    14.00 MISC(x_1) found in(x_0,x_1) => org:founded(x_0,x_1)

    public boolean isUnary() { return arg2.isNothing(); }

    static final Pattern unaryPattern = Pattern.compile("([^()\\s]*)\\((x_\\d)\\)");
    static final Pattern binaryPattern = Pattern.compile("([^()\\s]*)\\((x_\\d),(x_\\d)\\)");

    public Predicate(String relation, String arg1, String arg2) {
      this.relation = relation;
      this.arg1 = arg1;
      this.arg2 = Maybe.Just(arg2);
    }
    public Predicate(String relation, String arg1) {
      this.relation = relation;
      this.arg1 = arg1;
      this.arg2 = Maybe.Nothing();
    }

    /**
     * @param str String of the form pred(x_0,x_1)
     * @return Corresponding predicate
     */
    public static Predicate fromString(String str) {
      Matcher matcher = unaryPattern.matcher(str);
      if( matcher.matches() )
        return new Predicate(matcher.group(1), matcher.group(2));
      matcher = binaryPattern.matcher(str);
      if( matcher.matches() )
        return new Predicate(matcher.group(1), matcher.group(2), matcher.group(3));
      else
        throw new IllegalArgumentException("Illformed predicate string: " + str);
    }

    public String toString() {
      if(arg2.isDefined()) {
        return relation + "(" + arg1 + "," + arg2.get() + ")";
      } else {
        return relation + "(" + arg1 + ")";

      }
    }

    public boolean matchUnary(KBPEntity entity, Maybe<NERTag> fillType) {
      if(!isUnary()) {
        throw new IllegalArgumentException();
      }

      return fillType.getOrElse( entity.type ).name.equals(relation);
    }

    public KBPSlotFill apply(Context<String,KBPEntity> context, EntityGraph graph) {
      assert !isUnary();
      assert context.getB(arg1) != null;
      assert context.getB(arg2.get()) != null;

      KBPEntity arg1Entity = context.getB(arg1);
      KBPEntity arg2Entity = context.getB(arg2.get());


      return KBPNew.from(arg1Entity).slotValue(arg2Entity).slotType(graph.guessType(arg2Entity)).rel(relation).KBPSlotFill();
    }


  }
  public static class Rule {
    final double weight;
    final List<Predicate> heads;
    final Predicate tail;

    public Rule(double weight, List<Predicate> heads, Predicate tail) {
      this.weight = weight;
      this.heads = heads;
      this.tail = tail;
    }

    /**
     * @param str String of the form wt pred(x_0,x_1) pred(x_1) pred(x_1,x_4) => pred(x_0,x_1)
     * @return Corresponding predicate
     */
    public static Rule fromString(String str) {
      String[] parts = str.trim().split(" ");

      double weight = 1.0;
      List<Predicate> heads = new ArrayList<Predicate>();
      Predicate tail;

      int partIdx = 0;

      // Try the weight
      try {
        weight = Double.parseDouble(parts[partIdx]);
        partIdx++;
      } catch( NumberFormatException e ) {
        // Try to proceed as if nothing happened.
        weight = 1.0;
      }

      for( ; partIdx < parts.length; partIdx++ ) {
        String part = parts[partIdx];
        // Matched the head
        if( part.trim().equals("=>") ) { partIdx++; break; }
        else heads.add(Predicate.fromString(part));
      }
      if( partIdx != parts.length - 1 ) {
        throw new IllegalArgumentException("Illformed rule string: " + str);
      }
      tail = Predicate.fromString(parts[partIdx]);

      return new Rule(weight, heads, tail);
    }

    public String toString() {
      StringBuilder sb = new StringBuilder();
      sb.append(weight).append(" ");
      for(Predicate head : heads)
        sb.append(head).append(" ");
      sb.append("=> ");
      sb.append(tail);

      return sb.toString();
    }

  }

  /**
   * Constructs a set of rules from a file containing the following lines:
   * [wt] pred(x_0,x_1) pred(x_0) pred(x_0,x_1) => pred(x_0,x_1)
   * 19.00 ORGANIZATION(x_0) org:alternate_names(x_0,x_1) => org:city_of_headquarters(x_0,x_1)
   * The inferred edge is always from x0 to x1.
   */
  public static List<Rule> loadFromFile(File rulesFile, double cutoff) {
    List<Rule> rules = new ArrayList<Rule>();

    // Read each line and parse
    for( String line : IOUtils.readLines(rulesFile) ) {
      if( line.startsWith("#") ) continue; // Comments
      try {
        Rule rule = Rule.fromString(line);
        if( rule.weight >= cutoff )
          rules.add( rule );
      } catch(IllegalArgumentException e ) {
        debug("Couldn't parse rule: " +  line);
      }
    }
    return rules;
  }
  public static List<Rule> loadFromStrings(List<String> strings) {
    List<Rule> rules = new ArrayList<Rule>();

    // Read each line and parse
    for( String line : strings ) {
      if( line.startsWith("#") ) continue; // Comments
      Rule rule = Rule.fromString(line);
      rules.add( rule );
    }
    return rules;
  }

  class Context<A,B> {
    Map<A,B> map = Generics.newHashMap();
    Map<B,A> invMap = Generics.newHashMap();
    public A getA(B key) { return invMap.get(key); }
    public B getB(A key) { return map.get(key); }
    public boolean hasA(A key) { return map.containsKey(key); }
    public boolean hasB(B key) { return invMap.containsKey(key); }
    public void add(A key, B value) { map.put(key,value); invMap.put(value, key); }
    public void removeA(A key) { invMap.remove(map.get(key)); map.remove(key); }
    public void removeB(B key) { map.remove(invMap.get(key)); invMap.remove(key); }
  }

  public boolean match( final EntityGraph graph, final Rule rule, final List<Predicate> originalHeads, final KBPEntity x0, final Maybe<NERTag> type, final Context<String,KBPEntity> context ) {
    // At the very bottom, just verify that this slot is valid
    if( originalHeads.size() == 0 )  {
      List<KBPSlotFill> fill = Collections.singletonList(rule.tail.apply( context, graph ));
      if( irComponent.isDefined() )
        fill = SlotfillPostProcessor.unary(irComponent.get()).postProcess(context.getB("x_0"), fill);
      return fill.size() > 0;
    }

    final List<KBPSlotFill> edges = graph.getOutgoingEdges(graph.findEntity(x0).getOrElse(x0)); // Best effort to find
    final Pair<List<Predicate>,List<Predicate>> thingsToDo = CollectionUtils.split( originalHeads, new Function<Predicate, Boolean>() {
      @Override
      public Boolean apply(Predicate in) {
        return in.arg1.equals(context.getA(x0));
      }
    });

    final List<Predicate> othersWork = thingsToDo.second;

    if(  CollectionUtils.forall( thingsToDo.first, new Function<Predicate, Boolean>() {
      @Override
      public Boolean apply(final Predicate pred) {
        if( pred.isUnary() ) return pred.matchUnary( x0, type );
        // Check if we even have any slots of this type.
        final String arg2 = pred.arg2.get();
        return CollectionUtils.exists(edges, new Function<KBPSlotFill, Boolean>() {
          @Override
          public Boolean apply(final KBPSlotFill in) {
            // Check if I have a relation that matches this.
            if(!in.key.relationName.equals(pred.relation)) return false;

            // Try to match the argument.
            if( context.hasA(arg2) ) {
              return in.key.getSlotEntity().orCrash().name.equals(context.getB(arg2).name);
            } else {
              KBPEntity unificationAttempt = in.key.getSlotEntity().orCrash();
              unificationAttempt = graph.findEntity(x0, unificationAttempt.name).getOrElse(unificationAttempt);

              // Push onto the stack
              context.add( arg2, unificationAttempt );
              if( match(graph, rule, othersWork, unificationAttempt, in.key.slotType, context ) ) {
                return true;
              } else { // Undo
                context.removeA(arg2);
                return false;
              }
            }
          }
        });
      }
    }) ) {
      // TODO(arun): eeks this is terrible, but I don't know what else to do.
      // Copy changes to originalHeads
      originalHeads.clear();
      originalHeads.addAll(othersWork);
      return true;
    } else return false;
  }
  public Maybe<KBPSlotFill> match( EntityGraph graph, Rule rule, KBPEntity headEntity ) {
    Context<String, KBPEntity> context = new Context<String, KBPEntity>();
    context.add("x_0", headEntity);
    try {
      if( match(graph, rule, new ArrayList<GraphInferenceEngine.Predicate>(rule.heads), headEntity, Maybe.Just(graph.guessType(headEntity)), context ) ) {
        // Great, we matched; now use the context to find out what the tail is and appropriately fill in provenance.
        KBPEntity tailEntity = context.getB(rule.tail.arg2.get());
        List<KBPRelationProvenance> justification = new ArrayList<KBPRelationProvenance>();
        double score = 1.0;
        for( final Predicate pred : rule.heads ) {
          if( pred.isUnary() ) continue;
          // Get the canonical entity :-/
          KBPEntity arg1Entity = context.getB(pred.arg1);
          KBPEntity arg2Entity = context.getB(pred.arg2.get());
          // This search must work.
          KBPSlotFill link = CollectionUtils.find( graph.getEdges(arg1Entity, arg2Entity), new Function<KBPSlotFill, Boolean>() {
            @Override
            public Boolean apply(KBPSlotFill in) {
              return in.key.relationName.equals(pred.relation);
            }
          }).get();

          double linkScore = link.score.getOrElse(1.0);
          if (Double.isInfinite(linkScore) || Double.isNaN(linkScore)) { linkScore = 1.0; }
          if (linkScore < 0.0) { linkScore = 0.0; }
          if (linkScore > 1.0) { linkScore = 1.0; }
          score *= linkScore;
          // Add the provenance if I can
          if( link.provenance.isDefined() ) justification.add( link.provenance.get() );
        }

        // TODO(arun): Do a better job of guessing type.
        return Maybe.Just(
            KBPNew.from(headEntity).slotValue(tailEntity.name).slotType(graph.guessType(tailEntity)).rel(rule.tail.relation)
                  .provenance(( justification.size() > 0 ) ? Maybe.Just(justification.get(0))  : Maybe.<KBPRelationProvenance>Nothing())
                  .score(score).KBPSlotFill());
      } else {
        return Maybe.Nothing();
      }
    } catch( Exception e ) {
      err("Error while trying to match rule " + rule + " for entity " + headEntity);
    }
    return Maybe.Nothing();
  }


  /**
   * Apply given rules to the graph
   */
  public EntityGraph apply(EntityGraph graph, KBPEntity entity) {
    startTrack("Graph inference");
    log("Applying " + rules.size() + " rules");
    int addedSlots = 0;
    for(Rule rule : rules) {
      for( KBPSlotFill inferredFill : match(graph, rule, entity) ) {
        addedSlots += 1;
        log( "Matched rule: " + rule );
        log( inferredFill );
        graph.add(inferredFill);
      }
    }
    log("Added " + addedSlots + " additional slots.");
    endTrack("Graph inference");
    return graph;
  }

  /**
   * Apply the rules in GraphInferenceRules
   */
//  public static EntityGraph apply(EntityGraph graph) {
//    forceTrack("Applying inference rules");
//    log( "Rules: " + Props.TEST_GRAPH_INFERENCE_RULES );
//    for( String ruleIdentifier : Props.TEST_GRAPH_INFERENCE_RULES.split(",") ) {
//      forceTrack("Applying rule:" + ruleIdentifier);
//      List<KBPSlotFill> slots = Collections.emptyList();
//      if( ruleIdentifier.trim().equalsIgnoreCase("ParentSpouse") ) {
//        slots = GraphInferenceRules.ParentSpouseRule.apply(graph);
//      } else if( ruleIdentifier.trim().equalsIgnoreCase("ParentParent") ) {
//        slots = GraphInferenceRules.ParentParentRule.apply( graph );
//      } else if( ruleIdentifier.trim().equalsIgnoreCase("RelationAge") ) {
//        slots = GraphInferenceRules.RelationAgeRule.apply(graph);
//      } else if( ruleIdentifier.trim().equalsIgnoreCase("PoliticalTitle") ) {
//        slots = GraphInferenceRules.PoliticalTitleRule.apply(graph);
//      } else if( ruleIdentifier.trim().equalsIgnoreCase("PoliticalResidence") ) {
//        slots = GraphInferenceRules.PoliticalResidenceRule.apply(graph);
//      } else if( ruleIdentifier.trim().equalsIgnoreCase("RelationDetection") ) {
//        slots = GraphInferenceRules.RelationDetectionRule.apply(graph);
//      }
//      log( "Adding " + slots.size() + " relations" );
//      log( slots );
//      graph.addRelations( slots );
//      endTrack("Applying rule:" + ruleIdentifier);
//    }
//
//    endTrack("Applying inference rules");
//    return graph;
//  }

}
