package edu.stanford.nlp.kbp.slotfilling.evaluate;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.sql.Connection;
import java.sql.SQLException;
import java.util.*;

import edu.stanford.nlp.kbp.entitylinking.classify.namematcher.RuleBasedNameMatcher;
//import edu.stanford.nlp.kbp.slotfilling.SlotfillingTasks;
import edu.stanford.nlp.kbp.slotfilling.classify.HeuristicRelationExtractor;
import edu.stanford.nlp.kbp.slotfilling.classify.ModelType;
import edu.stanford.nlp.kbp.slotfilling.classify.RelationClassifier;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.common.CollectionUtils;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.ir.PostIRAnnotator;
import edu.stanford.nlp.kbp.slotfilling.ir.StandardIR;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.kbp.slotfilling.process.*;
import edu.stanford.nlp.kbp.slotfilling.process.KBPProcess.AnnotateMode;
import edu.stanford.nlp.kbp.slotfilling.process.RelationFilter.RelationFilterBuilder;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.util.logging.Redwood;

/**
 * An implementation of a SlotFiller.
 *
 * This class will fill slots given a query entity (complete with provenance).
 * The class depends on an IR component, the components in process to annotate documents, and the classifiers.
 *
 * @author Gabor Angeli
 */
public class SimpleSlotFiller implements SlotFiller {

  protected static final Redwood.RedwoodChannels logger = Redwood.channels("Infer");

  // Defining Instance Variables
  protected final Properties props;
  public final KBPIR irComponent;
  public final KBPProcess process;
  public final RelationClassifier classifyComponent;
  public final Maybe<RelationFilter> relationFilterForFeaturizer;
  public List<SentenceTriple> sentenceRecords = new ArrayList<SentenceTriple>();
  StanfordCoreNLP mypipeline = null;
  List<CoreMap> rawSentences=null;
  HashMap<String,HashMap<String,ArrayList<SentenceDouble>>> sentencesContainer=null;
  
  /**
   * Used to keep track of all the (entity, slot fill candidate) pairs recovered via IR
   */
  protected final GoldResponseSet goldResponses;

  public SimpleSlotFiller(Properties props,
                      KBPIR ir,
                      KBPProcess process,
                      final RelationClassifier classify,
                      GoldResponseSet goldResponses
                      ) {
    this.props = props;
    this.process = process;
    this.classifyComponent = classify;
    
    if(Props.TEST_RELATIONFILTER_DO) {
      RelationFilterBuilder rfBuilder = new RelationFilterBuilder(new Function<Pair<SentenceGroup, Maybe<CoreMap[]>>, Counter<String>>() {
        // Note(gabor): This function is to prevent a dependency between from Process on Classify; if you know of a more elegant solution I'm all for it!
        @Override
        public Counter<String> apply(Pair<SentenceGroup, Maybe<CoreMap[]>> in) {
          return classify.classifyRelationsNoProvenance(in.first, in.second);
        }
      });
      //noinspection unchecked
      for(Class<RelationFilter.FilterComponent> filterComponent : Props.TEST_RELATIONFILTER_COMPONENTS) {
        rfBuilder.addFilterComponent(filterComponent);
      }
      this.relationFilterForFeaturizer = Maybe.Just(rfBuilder.make());
    }
    else {
      this.relationFilterForFeaturizer = Maybe.Nothing();
    }

    this.goldResponses = goldResponses;

    if (Props.TEST_GOLDIR) {
      this.irComponent = new StandardIR(); // override the passed IR component
    } else {
      this.irComponent = ir;
    }
    
    mypipeline = new StanfordCoreNLP(props);

    /*populate sentences container*/
    sentencesContainer=new HashMap<String,HashMap<String,ArrayList<SentenceDouble>>>();
    try {
		BufferedReader br = new BufferedReader(new FileReader("/var/local/vidhoon/thesis/stanford-RE/candidate_sentences.txt"));
		String text;
		String delimiter="\t";
		while((text=br.readLine()) != null){
			//content+=text;
			String[] fields=text.split(delimiter);
			String eid=fields[0];//eid
			String rn=fields[1];//relation name
			String sent=fields[8];//sentence text			
			String prov=fields[3]; //total provenance string
			SentenceDouble sentRecord=new SentenceDouble(sent,prov);
			//System.out.println("Inserting record for "+eid);
			if(sentencesContainer.containsKey(eid)){
				HashMap<String,ArrayList<SentenceDouble>> entitySentMap=sentencesContainer.get(eid);
				if(entitySentMap.containsKey(rn)){
					ArrayList<SentenceDouble> entityRelSents=entitySentMap.get(rn);					
					entityRelSents.add(sentRecord);
					entitySentMap.put(rn, entityRelSents);
					sentencesContainer.put(eid, entitySentMap);
				}
				else{
					//no sentences for that relation
					ArrayList<SentenceDouble> entityRelSents=new ArrayList<SentenceDouble>();
					entityRelSents.add(sentRecord);
					entitySentMap.put(rn, entityRelSents);
					sentencesContainer.put(eid, entitySentMap);
				}
			}
			else{
				//no entry for the entity itself
				HashMap<String,ArrayList<SentenceDouble>> entitySentMap=new HashMap<String,ArrayList<SentenceDouble>>();
				ArrayList<SentenceDouble> entityRelSents=new ArrayList<SentenceDouble>();
				entityRelSents.add(sentRecord);
				entitySentMap.put(rn, entityRelSents);
				sentencesContainer.put(eid, entitySentMap);
			}

		}	
		br.close();

    }
    catch (IOException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}
    
    
  }

  @Override
  public List<KBPSlotFill> fillSlots(final KBPOfficialEntity queryEntity) {
    startTrack("Annotating " + queryEntity);

    // -- Raw Classification
    startTrack("Raw Annotation");
    // Read slot candidates from the index
    // Each of these tuples effectively represents (e_1, e_2, [datums]) where e_1 is the query entity.
    final Pair<? extends List<SentenceGroup>, ? extends Map<KBPair, CoreMap[]>> datumsAndSentences =
        Props.TEST_GOLDSLOTS ? Pair.makePair(new ArrayList<SentenceGroup>(), new HashMap<KBPair, CoreMap[]>())
                             : queryAndProcessSentences(queryEntity, Props.TEST_SENTENCES_PER_ENTITY);
        
    if(datumsAndSentences==null){
    	//no candidate sentences for the query
    	return null;
    }
    // Register datums we've seen
    final List<String> allSlotFills = new LinkedList<String>();
    int totalDatums = 0;
    for (SentenceGroup tuple : datumsAndSentences.first) { 
      //System.out.println(tuple.key.slotType+ "\t " + tuple.key.slotValue);
      allSlotFills.add(tuple.key.slotValue); totalDatums += tuple.size(); 
    }
    logger.log("Found " + datumsAndSentences.first.size() + " sentence groups (ave. " + (((double) totalDatums) / ((double) datumsAndSentences.first.size())) + " sentences per group)");

    forceTrack("P(r | e_1, e_2)");
    startTrack("Classifying Relations");
    List<Counter<KBPSlotFill>> tuplesWithRelation;
    if (Props.TEST_GOLDSLOTS) {
      // Case: simply output the right answer every time. This can be useful to test consistency, and the scoring script
      tuplesWithRelation = new ArrayList<Counter<KBPSlotFill>>();
      for (final KBPSlotFill response : goldResponses.correctFills()) {
        if (response.key.getEntity().equals(queryEntity)) {
          tuplesWithRelation.add(new ClassicCounter<KBPSlotFill>() {{
            setCount(response, response.score.orCrash());
          }});
        }
      }
    } else {
    //	System.out.println("Creating tuplesWithRelation here!!");
      tuplesWithRelation = CollectionUtils.map(datumsAndSentences.first, new Function<SentenceGroup, Counter<KBPSlotFill>>() {
        @Override public Counter<KBPSlotFill> apply(SentenceGroup input) {
        	
          // vvv RUN CLASSIFIER vvv
          Counter<Pair<String, Maybe<KBPRelationProvenance>>> relationsAsStrings = classifyComponent.classifyRelations(input, Maybe.fromNull(datumsAndSentences.second.get(input.key)));
          // ^^^                ^^^
          /*
            if(relationsAsStrings.size()==0){
        	  System.out.println("classification result empty");
          }
          */
          // Convert to Probabilities
          Counter<KBPSlotFill> countsForKBPair = new ClassicCounter<KBPSlotFill>();
          for (Map.Entry<Pair<String, Maybe<KBPRelationProvenance>>, Double> entry : relationsAsStrings.entrySet()) {
            RelationType rel = RelationType.fromString(entry.getKey().first).orCrash();
            Maybe<KBPRelationProvenance> provenance = entry.getKey().second;
            double prob = entry.getValue();
            //System.out.println("prob : "+prob);
            if (Props.TEST_PROBABILITYPRIORS) {
              prob = new Probabilities(irComponent, allSlotFills, entry.getValue())
                  .ofSlotValueGivenRelationAndEntity(input.key.slotValue, rel, queryEntity);
            }
            KBPair key = input.key;
            if (!key.slotType.isDefined()) { logger.warn("slot type is not defined for KBPair: " + key); } // needed for some of the consistency checks
            countsForKBPair.setCount(KBPNew.from(key).rel(rel).provenance(provenance).score(prob).KBPSlotFill(), prob);
          }

          // output
          return countsForKBPair;
        }});
    }
    endTrack("Classifying Relations");
    // Display predictions
    startTrack("Relation Predictions");
    java.text.DecimalFormat df = new java.text.DecimalFormat("0.000");
    for (Counter<KBPSlotFill> fillsByKBPair : tuplesWithRelation) {
      if (fillsByKBPair.size() > 0) {
        List<KBPSlotFill> bestRels = Counters.toSortedList(fillsByKBPair);
        StringBuilder b = new StringBuilder();
        b.append(fillsByKBPair.keySet().iterator().next().key.entityName).append(" | ");
        for (int i = 0; i < Math.min(bestRels.size(), 3); ++i) {
          b.append(bestRels.get(i).key.relationName)
              .append(" [").append(df.format(fillsByKBPair.getCount(bestRels.get(i)))).append("] | ");
        }
        b.append(fillsByKBPair.keySet().iterator().next().key.slotValue);
        logger.log(b);
      }
    }
    endTrack("Relation Predictions");
    endTrack("P(r | e_1, e_2)");

    // -- Convert to KBP data structures
    forceTrack("P(e_2 | e_1, r)");
    // Flatten the list of tuples, as we only care about (e_1, r, e_2) and don't want to group by e_2.
    List<KBPSlotFill> relations = new ArrayList<KBPSlotFill>();
    for (Counter<KBPSlotFill> counter : tuplesWithRelation) {
      for (Map.Entry<KBPSlotFill, Double> entry : counter.entrySet()) {
        relations.add(entry.getKey());
      }
    }
    // Add in rule-based extractions
    if (Props.TEST_RULES_DO) {
      startTrack("Rule based additions");
      List<CoreMap> sentences = new ArrayList<CoreMap>();
      for (Map.Entry<KBPair, CoreMap[]> entry : datumsAndSentences.second.entrySet()) {
        // Get sentences in official index
        for (CoreMap sent : entry.getValue()) {
          if (KBPRelationProvenance.isOfficialIndex(sent.get(KBPAnnotations.SourceIndexAnnotation.class))) { sentences.add(sent); }
        }
        // Get heuristic fills
        for (Map.Entry<Pair<String, Maybe<KBPRelationProvenance>>, Double> fill : HeuristicRelationExtractor.allExtractors.apply(Pair.makePair(entry.getKey(), entry.getValue())).entrySet()) {
          log(entry.getKey().entityName + " | " + fill.getKey().first + " | " + entry.getKey().slotValue);
          relations.add( KBPNew.from(entry.getKey()).rel(fill.getKey().first).provenance(fill.getKey().second).score(fill.getValue()).KBPSlotFill());
        }
      }
      if (Props.TEST_RULES_ALTERNATENAMES_DO) {
        for (KBPSlotFill altName : AlternateNamesExtractor.extractSlots(queryEntity, sentences)) {
          log(altName.key.entityName + " | " + altName.key.relationName + " | " + altName.key.slotValue);
          relations.add(altName);
        }
      }
      endTrack("Rule based additions");
    }
    logger.log("" + relations.size() + " slots extracted");
    for (KBPSlotFill slot : relations) { goldResponses.registerResponse(slot); }
    endTrack("P(e_2 | e_1, r)");
    endTrack("Raw Annotation");

    // -- Run consistency checks
    startTrack("Consistency and Inference");
    // Run consistency pass 1
    List<KBPSlotFill> cleanRelations
        = Props.TEST_CONSISTENCY_DO ? SlotfillPostProcessor.unary(irComponent).postProcess(queryEntity, relations, goldResponses) : relations;
    logger.log("" + cleanRelations.size() + " slot fills remain after consistency (pass 1)");
    // Filter on missing provenance
    List<KBPSlotFill> withProvenance = new ArrayList<KBPSlotFill>(cleanRelations.size());
    for (KBPSlotFill fill : cleanRelations) {     		
     		fill.provenance=findBestProvenance(queryEntity, fill);
     		//System.out.println();
    		  //KBPNew.from(fill).provenance(findBestProvenance(queryEntity, fill)).KBPSlotFill();
      if ((!Props.TEST_PROVENANCE_DO || (fill.provenance.isDefined() /* && augmented.provenance.get().isOfficial()*/) ) && fill.provenance.get()!=null ) {
        withProvenance.add(fill);
      } else {
    	  System.out.println("discaring fill");
        goldResponses.discardNoProvenance(fill);
      }
    }
    /*
    for (KBPSlotFill slot : withProvenance) { goldResponses.registerResponse(slot); } // re-register after provenance
    logger.log("" + withProvenance.size() + " slot fills remain after provenance");
    // Run consistency pass 2
    List<KBPSlotFill> consistentRelations = finalConsistencyAndProvenancePass(queryEntity, withProvenance, goldResponses);
    */
    endTrack("Consistency and Inference");
   
    // -- Print Judgements
    //prettyLog(goldResponses.loggableForEntity(queryEntity, Maybe.Just(irComponent)));

    endTrack("Annotating " + queryEntity);
    
    return withProvenance;
    //return consistentRelations;
   
  }

  //
  // Public Utilities
  //
  
  private List<CoreMap> myquerySentences(KBPOfficialEntity entity,int sentLimit){
	  List<CoreMap> resultSentences = new ArrayList<CoreMap>();
	  System.out.println("querying sentences for "+entity.queryId);
	  int counter=0;
	  if(sentencesContainer.containsKey(entity.queryId.get())){
		  HashSet<String> sentSet = new HashSet<String>();
		  HashMap<String,ArrayList<SentenceDouble>> entitySentMap=sentencesContainer.get(entity.queryId.get());
		  PostIRAnnotator postirAnn=new PostIRAnnotator(entity.name, Maybe.<String>Nothing(), Maybe.<String>Nothing(), Maybe.<String>Nothing(), true);
		  for(String key:entitySentMap.keySet()){
			  ArrayList<SentenceDouble> entityRelSents=entitySentMap.get(key);
			  for(SentenceDouble sd : entityRelSents){
				  if(sentSet.contains(sd.sentence)){
					  continue;
				  }
				  else{
					  sentSet.add(sd.sentence);
				  }
				  if(resultSentences.size()>sentLimit)
					  break;
				  //System.out.println("processing  --  "+sd.sentence);
				  counter++;
				  
				  Annotation document = new Annotation(sd.sentence);
				  mypipeline.annotate(document);				
				  postirAnn.annotate(document);
				  List<CoreMap> tempResults  = document.get(SentencesAnnotation.class);
				 
				  
				  for(CoreMap res : tempResults){
					  //	System.out.println("coremap: "+res);
						resultSentences.add(res);
						sd.provenance.containingSentenceLossy=Maybe.Just(res);
				  }					
			  }
		  }
		  
		  return resultSentences;
	  }
	  else{
		  System.out.println("returning null for "+entity.queryId);
		  return null;
	  }
	
/*	  
	  try {
		BufferedReader br = new BufferedReader(new FileReader("/var/local/vidhoon/ra/stan-process-temp/output_short.txt"));
		String text;
		//String content= new String();
		String delimiter="\t";
		List<CoreMap> resultSentences = new ArrayList<CoreMap>();
		while((text=br.readLine()) != null){
			//content+=text;
			String[] fields=text.split(delimiter);
			String rn=fields[1];//relation name
			String sent=fields[8];//sentence text
			
			String prov=fields[3]; //total provenance string
			SentenceTriple st=new SentenceTriple(rn,sent,prov);
			sentenceRecords.add(st);
			Annotation document = new Annotation(sent);
			//System.out.println("doing pipeline annotate!!!!!");
			mypipeline.annotate(document);
			//System.out.println("pipeline annotate complete!!!!!");
			new PostIRAnnotator(entity.name, Maybe.<String>Nothing(), Maybe.<String>Nothing(), Maybe.<String>Nothing(), true).annotate(document);
			//System.out.println("postIR annotate complete!!!!!");
			List<CoreMap> tempResults  = document.get(SentencesAnnotation.class);
			for(CoreMap res : tempResults){
				resultSentences.add(res);
			}
		}
		
		
	
		br.close();
		 System.out.println("returning sentences!!!!!");
		return resultSentences; 
	} catch (IOException e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}

	  
	  return null;
	  */
	  
  }

  /**
   * Runs a final pass for consistency and get any provenances that haven't been retrieved yet.
   *
   * @param queryEntity The entity we we are checking slots for
   * @param slotFills The candidate slot fills to filter for consistency
   * @param responseChecklist The response checklist to register fills that have been added or removed
   * @return A list of slot fills, guaranteed to be consistent and with provenance
   */
  protected List<KBPSlotFill> finalConsistencyAndProvenancePass(KBPOfficialEntity queryEntity, List<KBPSlotFill> slotFills, GoldResponseSet responseChecklist) {
    // Run consistency pass 2
    List<KBPSlotFill> consistentRelations
      = Props.TEST_CONSISTENCY_DO ? SlotfillPostProcessor.global(irComponent).postProcess(queryEntity, slotFills, responseChecklist) : slotFills;
    logger.log("" + consistentRelations.size() + " slot fills remain after consistency (pass 2)");
    // Run provenance pass 2
    List<KBPSlotFill> finalRelations = new ArrayList<KBPSlotFill>();
    for (KBPSlotFill fill : consistentRelations) {
      assert fill != null;
      KBPSlotFill augmented = KBPNew.from(fill).provenance(findBestProvenance(queryEntity, fill)).KBPSlotFill();
      if (!Props.TEST_PROVENANCE_DO || (augmented.provenance.isDefined() && augmented.provenance.get().isOfficial())) {
        finalRelations.add(augmented);
      } else {
        responseChecklist.discardNoProvenance(fill);
      }
    }
    // (last pass to make sure we have provenance)
    finalRelations = CollectionUtils.filter(finalRelations, new Function<KBPSlotFill, Boolean>() { public Boolean apply(KBPSlotFill in) { return !Props.TEST_PROVENANCE_DO || (in.provenance.isDefined() && in.provenance.get().isOfficial()); } });
    logger.log("" + consistentRelations.size() + " slot fills remain after final provenance check");

   // if (!Props.TEST_GOLDSLOTS && Props.TRAIN_MODEL != ModelType.GOLD && !Props.TEST_GOLDIR) {
    //  for (KBPSlotFill fill : finalRelations) { if (!fill.provenance.isDefined() || !fill.provenance.get().isOfficial()) { throw new IllegalStateException("Invalid provenance for " + fill); } }
   // }
    return finalRelations;
  }

  /**
   * Query and annotate a KBPOfficialEntity to get a featurized and annotated KBPTuple.
   *
   * @param entity The entity to query. This is a fancy way of saying "Obama"
   * @return A list of featurized datums, and a collection of raw sentences (for rule-based annotators)
   */
  private Pair<List<SentenceGroup>, Map<KBPair, CoreMap[]>> queryAndProcessSentences(KBPOfficialEntity entity, int sentencesPerEntity) {
    startTrack("Processing " + entity + " [" + sentencesPerEntity + " sentences max]");

    // -- IR
    // Get supporting sentences
    rawSentences = myquerySentences(entity, sentencesPerEntity);
    if(rawSentences==null){
    	return null;
    }
    
	//List<CoreMap> supportingSentences=document.get(SentencesAnnotation.class);
    /*
    List<CoreMap> rawSentences = irComponent.querySentences(entity.name, entity.type,
        entity.representativeDocument.isDefined() ? new HashSet<String>(Arrays.asList(entity.queryId.get())) : new HashSet<String>(),
        sentencesPerEntity);
        */
    // Get datums from sentences.
    Redwood.startTrack("Annotating " + rawSentences.size() + " sentences...");
    //System.out.println("Annotating " + rawSentences.size() + " sentences...");
    List<CoreMap> supportingSentences = process.annotateSentenceFeatures(entity, rawSentences, AnnotateMode.ALL_PAIRS);
    Redwood.endTrack("Annotating " + rawSentences.size() + " sentences...");

    // -- Process
    Annotation annotation = new Annotation("");
    annotation.set(CoreAnnotations.SentencesAnnotation.class, rawSentences);
    Redwood.forceTrack("Featurizing " + annotation.get(CoreAnnotations.SentencesAnnotation.class).size() + " sentences...");
    Map<KBPair, Pair<SentenceGroup, List<CoreMap>>> datums = process.featurizeWithSentences(annotation, relationFilterForFeaturizer);
    Redwood.endTrack("Featurizing " + annotation.get(CoreAnnotations.SentencesAnnotation.class).size() + " sentences...");
    // Register this as a datum we've seen
    logger.log("registering slot fills [" + datums.size() + " KBPairs]...");
    endTrack("Processing " + entity + " [" + sentencesPerEntity + " sentences max]");

    // -- Return
    List<SentenceGroup> groups = new ArrayList<SentenceGroup>();
    Map<KBPair, CoreMap[]> sentences = new HashMap<KBPair, CoreMap[]>();
    for (Map.Entry<KBPair, Pair<SentenceGroup, List<CoreMap>>> datum : datums.entrySet()) {
      if (datum.getKey().getEntity().equals(entity)) {
        groups.add( Props.HACKS_DISALLOW_DUPLICATE_DATUMS ? datum.getValue().first.removeDuplicateDatums() : datum.getValue().first );
        sentences.put(datum.getKey(), datum.getValue().second.toArray(new CoreMap[datum.getValue().second.size()]));
      }
    }
    return Pair.makePair(groups, sentences);
  }


  protected KBPRelationProvenance getProvenance(String qid, String sentence, KBTriple key){
//	  System.out.println( "get provenance for: "+sentence + " relation name: "+key.relationName);
//	  System.out.println("querying sentences for "+qid);
	  if(sentencesContainer.containsKey(qid)){
		  HashMap<String,ArrayList<SentenceDouble>> entitySentMap=sentencesContainer.get(qid);
		  if(entitySentMap.containsKey(key.relationName)){
			  ArrayList<SentenceDouble> entityRelSents=entitySentMap.get(key.relationName);
			  for(SentenceDouble sd : entityRelSents){
				 if(sd.sentence.equals(sentence)){
					 return sd.provenance;
				 }
			  }
			  /*
			  //sentence not found in that relation - try all other relations to see if the sentence is present and return provenance
			  for(String k : entitySentMap.keySet()){
				  ArrayList<SentenceDouble> erSents=entitySentMap.get(key.relationName);
				  for(SentenceDouble sd : erSents){
					 if(sd.sentence.equals(sentence)){
						 return sd.provenance;
					 }
				  }
			  }
			  */
			  
			  //none of the relations contained the sentence - something is horribly wrong
			 // System.out.println("Could nt find sentence against any relation - classifier has identified a new relation for a sentence");
			  return null;
		  }	  
		  else if(key.relationName.equals("per:member_of") || key.relationName.equals("per:employee_of")){
			  String rn= new String("per:employee_or_member_of");
			  if(entitySentMap.containsKey(rn)){
				  ArrayList<SentenceDouble> entityRelSents=entitySentMap.get(rn);
				  for(SentenceDouble sd : entityRelSents){
					 if(sd.sentence.equals(sentence)){
						 return sd.provenance;
					 }
				  }
				  return null;
			  }
			  else{
				//  System.out.println("relation not found!!!! for provenance");
				  return null;
			  }
			  
		  }
		  else{
			 // System.out.println("relation not found!!!! for provenance");
			  return null;
		  }
		 
		  
	  }
	  else{
	//	  System.out.println("returning null for "+qid);
		  return null;
	  }

	  
/*	  
	  for(SentenceTriple sentRecord : sentenceRecords){
		 
		  String sent=sentRecord.sentence;
		  String rel=sentRecord.relationType;
		//  System.out.println( "sentence record " +sent);
		  if(sent.equals(sentence) && rel.equals(key.relationName)){
			  System.out.println( "sentence record found");
			// System.out.println( sentRecord.provenance.toString());
			 return sentRecord.provenance;
		  }
		  else if(sent.equals(sentence) && rel.equals("per:employee_or_member_of") && ( key.relationName.equals("per:employee") || key.relationName.equals("per:member_of"))){
			  System.out.println( "sentence record found - hack");
				 //System.out.println( sentRecord.provenance.toString());
				 return sentRecord.provenance;
		  }
	  }
*/	  
	 
  }
  protected Maybe<KBPRelationProvenance> findBestProvenance(final KBPOfficialEntity entity, final KBPSlotFill fill) {
    if (!Props.TEST_PROVENANCE_DO) { return fill.provenance.orElse(Maybe.Just(new KBPRelationProvenance("unk_id", "/unk/index"))); }
    startTrack("Provenance For " + fill);
    final Pointer<KBPRelationProvenance> bestProvenance = new Pointer<KBPRelationProvenance>();
    double bestProvenanceProbability = -0.01;
    KBPRelationProvenance prov=null;
    // Try the cache
    if (Props.CACHE_PROVENANCE_DO) {
      PostgresUtils.withKeyProvenanceTable(Props.DB_TABLE_PROVENANCE_CACHE, new PostgresUtils.KeyProvenanceCallback() {
        @Override
        public void apply(Connection psql) throws SQLException {
          for (KBPRelationProvenance prov : get(psql, Props.DB_TABLE_PROVENANCE_CACHE, keyToString(fill.key))) {
            if (!prov.sentenceIndex.isDefined()) {
              warn("retreived provenance that didn't have a sentence index -- re-computing");
            } else {
              bestProvenance.set(prov);
            }
          }
        }
      });
    }
    if (!bestProvenance.dereference().isDefined()) {
      logger.debug("provenance cache miss!");
      // Try to use provenance from original classifier
      if (!bestProvenance.dereference().isDefined() && fill.provenance.isDefined() && fill.provenance.get().sentenceIndex.isDefined()) {
        logger.debug("using provenance from classifier");
        bestProvenance.set(fill.provenance.get());
        if (bestProvenance.dereference().isDefined() && Utils.assertionsEnabled()) { assert bestProvenance.dereference().get().sentenceIndex.isDefined(); }
      }

      // Query
      if (!bestProvenance.dereference().isDefined()) {
    	//  System.out.println("abc!!!!!!!");
        KBTriple key = fill.key;
        // Get String forms of entity and slot value
        String entityName = entity.name;
        if (fill.provenance.isDefined() && fill.provenance.get().entityMentionInSentence.isDefined() && fill.provenance.get().containingSentenceLossy.isDefined()) {
          entityName = CoreMapUtils.sentenceSpanString(fill.provenance.get().containingSentenceLossy.get(), fill.provenance.get().entityMentionInSentence.get());
        }
        String slotValue = key.slotValue;
        if (fill.provenance.isDefined() && fill.provenance.get().slotValueMentionInSentence.isDefined() && fill.provenance.get().containingSentenceLossy.isDefined()) {
          slotValue = CoreMapUtils.sentenceSpanString(fill.provenance.get().containingSentenceLossy.get(), fill.provenance.get().slotValueMentionInSentence.get());
        }
//        List<CoreMap> potentialProvenances = this.irComponent.querySentences(entity.name, key.slotValue, key.relationName, 25, true);

        // List<CoreMap> potentialProvenances = myquerySentences(entity);
        List<CoreMap> potentialProvenances = rawSentences;
        
//        if (!key.slotValue.equals(slotValue)) { potentialProvenances.addAll(this.irComponent.querySentences(entity.name, key.slotValue, key.relationName, 25, true)); }
//        if (!entity.name.equals(entityName)) { potentialProvenances.addAll(this.irComponent.querySentences(entityName, key.slotValue, key.relationName, 25, true)); }
        for(CoreMap sentence : potentialProvenances) {
          if (sentence.get(CoreAnnotations.TokensAnnotation.class).size() > 150) { continue; }
          // Error check
         /* if (!sentence.containsKey(KBPAnnotations.SourceIndexAnnotation.class) && sentence.get(KBPAnnotations.SourceIndexAnnotation.class).toLowerCase().endsWith(Props.INDEX_OFFICIAL.getName().toLowerCase())) {
            warn("Queried a document which is purportedly not from the official index!");
            continue;
          }*/


          // Try to classify provenance
          Annotation ann = new Annotation("");
          List<CoreMap> sentences = Arrays.asList(sentence);
          sentences = this.process.annotateSentenceFeatures(key.getEntity(), sentences);
          ann.set(CoreAnnotations.SentencesAnnotation.class, sentences);
          Map<KBPair, SentenceGroup> datums = this.process.featurize(ann);
          // Get the best key to match to
          KBPair pair = KBPNew.from(key).KBPair();  // default
          if (!datums.containsKey(pair)) {  // try to find a close match
            for (KBPair candidate : datums.keySet()) {
              if ((candidate.getEntity().equals(key.getEntity()) || candidate.getEntity().name.equals(entityName)) &&  // the entities must match
                 (candidate.slotValue.toLowerCase().contains(slotValue.toLowerCase()) ||  // try with the inferred "original" slot value
                     slotValue.toLowerCase().contains(candidate.slotValue.toLowerCase()) ||
                     RuleBasedNameMatcher.isAcronym(candidate.slotValue, slotValue.split("\\s+")) ||
                     RuleBasedNameMatcher.isAcronym(slotValue, candidate.slotValue.split("\\s+")) ||
                     candidate.slotValue.toLowerCase().contains(key.slotValue.toLowerCase()) ||  // try with the defined slot value
                     key.slotValue.toLowerCase().contains(candidate.slotValue.toLowerCase()) ||
                     RuleBasedNameMatcher.isAcronym(candidate.slotValue, key.slotValue.split("\\s+")) ||
                     RuleBasedNameMatcher.isAcronym(key.slotValue, candidate.slotValue.split("\\s+"))  )  ) {
                logger.debug("using key: " + candidate);
                pair = candidate;
                break;
              }
            }
          }
          // Classify
          if (datums.containsKey(pair)) {
            Pair<Double, Maybe<KBPRelationProvenance>> candidate = this.classifyComponent.classifyRelation(datums.get(pair), RelationType.fromString(key.relationName).orCrash(), Maybe.<CoreMap[]>Nothing());
            if (!bestProvenance.dereference().isDefined() || candidate.first > bestProvenanceProbability) {
              // Candidate provenance found
              boolean updated = false;
              if (candidate.second.isDefined() && candidate.second.get().sentenceIndex.isDefined()) {
                // Case: can use the provenance from classification (preferred)
                bestProvenance.set(candidate.second.get());
                updated = true;
                if (bestProvenance.dereference().isDefined() && Utils.assertionsEnabled()) { assert bestProvenance.dereference().get().sentenceIndex.isDefined(); }
              } else {
                // Case: can compute a provenance from the sentence
            	  /*
                for (KBPRelationProvenance provenance : KBPRelationProvenance.compute(sentence, fill.key)) {
                	System.out.println("setting provenance "+fill.key.relationName); bestProvenance.set(provenance); updated = true; 
                	}
                	*/
            	  prov=getProvenance(entity.queryId.get(),sentence.toString(),fill.key);
            	  //System.out.println("setting provenance "+fill.key.relationName); 
            	  if(prov==null){
            		//  System.out.println("received null provenance"); 
            		  if(!bestProvenance.dereference().isNothing()){
            			  prov=bestProvenance.dereference().get();
            		  }  
        		  
            	  }
            	  else{
            		  bestProvenance.set(prov); 
            	  }
            	  
            	  updated = true; 
            	  
                if (bestProvenance.dereference().isDefined() && Utils.assertionsEnabled()) { assert bestProvenance.dereference().get().sentenceIndex.isDefined(); }
              }
              // Perform some updating if a provenance was set
              if (updated) {
                if (candidate.first > bestProvenanceProbability) { bestProvenanceProbability = candidate.first; }
                logger.debug("using: " + CoreMapUtils.sentenceToMinimalString(sentence));
              }
            }
          }
        }

        // Backup (pick first IR result)
        for(CoreMap sentence : potentialProvenances) {
          if (sentence.get(CoreAnnotations.TokensAnnotation.class).size() > 75) { continue; }
          if (!bestProvenance.dereference().isDefined()) {
        	//  System.out.println("Again setting best provenance");
            logger.warn("using first IR result: " + CoreMapUtils.sentenceToMinimalString(sentence));
            for (KBPRelationProvenance provenance : KBPRelationProvenance.compute(sentence, fill.key)) { bestProvenance.set(provenance); }
            if (bestProvenance.dereference().isDefined() && Utils.assertionsEnabled()) { assert bestProvenance.dereference().get().sentenceIndex.isDefined(); }
            break;
          }
        }
      }
      /*
      // Do Cache
      if (Props.CACHE_PROVENANCE_DO && bestProvenance.dereference().isDefined()) {
        PostgresUtils.withKeyProvenanceTable(Props.DB_TABLE_PROVENANCE_CACHE, new PostgresUtils.KeyProvenanceCallback() {
          @Override
          public void apply(Connection psql) throws SQLException {
            put(psql, Props.DB_TABLE_PROVENANCE_CACHE, keyToString(fill.key), bestProvenance.dereference().orCrash());
            if (Props.KBP_EVALUATE && !psql.getAutoCommit()) { psql.commit(); }  // slower, but allows for stopping a run halfway through
          }
        });
      }
      */
    }

    // Return
    if (bestProvenance.dereference().isDefined() && Utils.assertionsEnabled()) { assert bestProvenance.dereference().get().sentenceIndex.isDefined(); }
    debug(bestProvenance.dereference().isDefined() ? "found provenance" : "no provenance!");
   // System.out.println(bestProvenance.dereference().isDefined() ? "found provenance" : "no provenance!");
    endTrack("Provenance For " + fill);
    Maybe<KBPRelationProvenance> res=Maybe.Just(prov);
    return res;
  }

}


