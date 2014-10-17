package edu.stanford.nlp.kbp.slotfilling.process;

import static edu.stanford.nlp.util.logging.Redwood.Util.err;

import java.sql.Connection;
import java.sql.SQLException;
import java.util.*;

import edu.stanford.nlp.ie.machinereading.structure.*;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.kbp.slotfilling.common.KBPAnnotations.SourceIndexAnnotation;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.kbp.slotfilling.ir.PostIRAnnotator;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.ling.CoreAnnotations.DocIDAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.AnnotationPipeline;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.logging.Redwood;

/**
 * <p>The entry point for processing and featurizing a sentence or group of sentences.
 * In particular, before being fed to the classifier either at training or evaluation time,
 * a sentence should be passed through the {@link KBPProcess#annotateSentenceFeatures(KBPEntity, List)}
 * function, as well as the {@link KBPProcess#featurizeSentence(CoreMap, Maybe)}
 * (or variations of this function, e.g., the buik {@link KBPProcess#featurize(Annotation)}).</p>
 *
 * <p>At a high level, this function is a mapping from {@link CoreMap}s representing sentences to
 * {@link SentenceGroup}s representing datums (more precisely, a collection of datums for a single entity pair).
 * The input should have already passed through {@link edu.stanford.nlp.kbp.slotfilling.ir.PostIRAnnotator}; the
 * output is ready to be passed into the classifier.</p>
 *
 * <p>This is also the class where sentence gloss caching is managed. That is, every datum carries with itself a
 * hashed "sentence gloss key" which alleviates the need to carry around the raw sentence, but can be used to retrieve
 * that sentence if it is needed -- primarily, for Active Learning. See {@link KBPProcess#saveSentenceGloss(String, CoreMap, Maybe, Maybe)}
 * and {@link KBPProcess#recoverSentenceGloss(String)}.</p>
 *
 * @author Kevin Reschke
 * @author Jean Wu (sentence gloss infrastructure)
 * @author Gabor Angeli (managing hooks into Mention Featurizers; some hooks into sentence gloss caching)
 */
public class KBPProcess extends Featurizer implements DocumentAnnotator {

  protected static final Redwood.RedwoodChannels logger = Redwood.channels("Process");

  private final FeatureFactory rff;

  private final Properties props;  // needed to create a StanfordCoreNLP down the line
  private final KBPIR querier;

  public enum AnnotateMode { 
                              NORMAL,   // do normal relation annotation for main entity
                              ALL_PAIRS  // do relation annotation for all entity pairs
                           }
  
  public KBPProcess(Properties props, KBPIR querier) {
	
    this.props = props;
    this.querier = querier;
    // Setup feature factory
    rff = new FeatureFactory(Props.TRAIN_FEATURES);
    rff.setDoNotLexicalizeFirstArgument(true);
  }

  public Maybe<Datum<String,String>> featurize( RelationMention rel ) {
    try {
      Datum<String, String> d = rff.createDatum(rel);

      return Maybe.Just(d);
    } catch (RuntimeException e) {
      err(e);
      return Maybe.Nothing();
    }
  }

  // TODO(gabor) In principle, there's no reason this shouldn't multithread
  // TODO (gabor) maybe this does mulithtread alsready ???
  /**
   * Featurize |sentence| with respect to |entity|, with optional |filter|.
   * 
   * That is, build a datum (singleton sentence group) for each relation that
   * is headed by |entity|, subject to filtering.
   *  
   * @param sentence
   *          CoreMap with relation mention annotations. 
   * @param filter
   *          Optional relation filter for within-sentence filtering
   * @return List of singleton sentence groups.  Each singleton represents a
   *          datum for one of the relations found in this sentence.
   */
  @Override
  public synchronized List<SentenceGroup> featurizeSentence(CoreMap sentence, Maybe<RelationFilter> filter) {
    List<RelationMention> relationMentionsForEntity = sentence.get(MachineReadingAnnotations.RelationMentionsAnnotation.class);
    List<RelationMention> relationMentionsForAllPairs = sentence.get(MachineReadingAnnotations.AllRelationMentionsAnnotation.class);

    List<SentenceGroup> datumsForEntity = featurizeRelations(relationMentionsForEntity,sentence);
    
    if(filter.isDefined()) {
      List<SentenceGroup> datumsForAllPairs = featurizeRelations(relationMentionsForAllPairs,sentence);
      datumsForEntity = filter.get().apply(datumsForEntity, datumsForAllPairs, sentence);
    }
    
    return datumsForEntity;
  }

  //Construct datums for these relation mentions.
  //Output is list of singleton sentence groups
  private List<SentenceGroup> featurizeRelations(List<RelationMention> relationMentions, CoreMap sentence) {
    List<SentenceGroup> datums = new ArrayList<SentenceGroup>();
    if (relationMentions == null) { return datums; }

    for (RelationMention rel : relationMentions) {
      assert rel instanceof NormalizedRelationMention;
      NormalizedRelationMention normRel = (NormalizedRelationMention) rel;
      assert normRel.getEntityMentionArgs().get(0).getSyntacticHeadTokenPosition() >= 0;
      assert normRel.getEntityMentionArgs().get(1).getSyntacticHeadTokenPosition() >= 0;
      for (Datum<String, String> d : featurize(rel)) {


        // Pull out the arguments to construct the entity pair this
        // datum will express.
        List<EntityMention> args = rel.getEntityMentionArgs();
        EntityMention leftArg = args.get(0);
        EntityMention rightArg = args.get(1);
        KBPEntity entity = normRel.getNormalizedEntity();
        String slotValue = normRel.getNormalizedSlotValue();

        // Create key
        KBPair key = KBPNew
            .entName(entity != null ? entity.name : (leftArg.getNormalizedName() != null ? leftArg.getNormalizedName() : leftArg.getFullValue()))
            .entType(entity != null ? entity.type : Utils.getNERTag(leftArg).orCrash())
            .entId(entity != null
                ? (entity instanceof KBPOfficialEntity ? ((KBPOfficialEntity) entity).id : Maybe.<String>Nothing()).orElse(Utils.getKbpId(leftArg))
                : Utils.getKbpId(leftArg))
            .slotValue(slotValue)
            .slotType(Utils.getNERTag(rightArg)).KBPair();
        logger.debug("featurized datum key: " + key);

        // Also track the provenance information
        String indexName = leftArg.getSentence().get(SourceIndexAnnotation.class);
        String docId = leftArg.getSentence().get(DocIDAnnotation.class);
        Integer sentenceIndex = leftArg.getSentence().get(CoreAnnotations.SentenceIndexAnnotation.class);
        Span entitySpan = leftArg.getExtent();
        Span slotFillSpan = rightArg.getExtent();
        KBPRelationProvenance provenance =
            sentenceIndex == null ? new KBPRelationProvenance(docId, indexName)
                : new KBPRelationProvenance(docId, indexName, sentenceIndex, entitySpan, slotFillSpan, sentence);

        // Handle Sentence Gloss Caching
        String hexKey = CoreMapUtils.getSentenceGlossKey(sentence.get(CoreAnnotations.TokensAnnotation.class), key.entityName, key.slotValue);
        saveSentenceGloss(hexKey, sentence, Maybe.Just(entitySpan), Maybe.Just(slotFillSpan));

        // Construct singleton sentence group; group by slotValue entity
        SentenceGroup sg = new SentenceGroup(key, d, provenance, hexKey);
        datums.add(sg);

        // TODO(arun): Handle counters
        // labelStats.incrementCount(d.label());
      }
    }
    return datums;
  }

  public synchronized List<CoreMap> annotateSentenceFeatures (KBPEntity entity, List<CoreMap> sentences) {
    return annotateSentenceFeatures(entity,sentences,AnnotateMode.NORMAL);
  }
  
  // TODO(gabor) In principle, there's no reason this shouldn't multithread. Particularly in the new framework
  public synchronized List<CoreMap> annotateSentenceFeatures( KBPEntity entity,
                                                              List<CoreMap> sentences, AnnotateMode annotateMode) {
    // Check if PostIR was run
    for (CoreMap sentence : sentences) {
      if (!sentence.containsKey(KBPAnnotations.AllAntecedentsAnnotation.class) && !Props.JUNIT) {
        throw new IllegalStateException("Must pass sentence through PostIRAnnotator before calling AnnotateSentenceFeatures");
      }
    }

    // Create the mention annotation pipeline
    AnnotationPipeline pipeline = new AnnotationPipeline();
    PostIRAnnotator postirAnn=new PostIRAnnotator(entity.name, Maybe.<String>Nothing(), Maybe.<String>Nothing(), Maybe.<String>Nothing(), true);

    pipeline.addAnnotator(new EntityMentionAnnotator(entity));
    pipeline.addAnnotator(new SlotMentionAnnotator());
    pipeline.addAnnotator(new RelationMentionAnnotator(entity, querier.getKnownSlotFillsForEntity(entity), annotateMode));
    pipeline.addAnnotator(new PreFeaturizerAnnotator(props));
    //pipeline.addAnnotator(postirAnn);
    // Annotate
    Annotation ann = new Annotation(sentences);
    pipeline.annotate(ann);
    // Sanity checks
    for (CoreMap sentence : ann.get(SentencesAnnotation.class)) {
      for (RelationMention rm : sentence.get(MachineReadingAnnotations.RelationMentionsAnnotation.class)) {
        assert rm.getArg(0) instanceof EntityMention;
        assert rm.getArg(1) instanceof EntityMention;
        assert ((EntityMention) rm.getArg(0)).getSyntacticHeadTokenPosition() >= 0;
        assert ((EntityMention) rm.getArg(1)).getSyntacticHeadTokenPosition() >= 0;
      }
    }
    // Return valid sentences
    return ann.get(SentencesAnnotation.class);
  }

  public void saveSentenceGloss(final String hexKey, CoreMap sentence, Maybe<Span> entitySpanMaybe, Maybe<Span> slotFillSpanMaybe) {
    // TODO(gabor) do nothing in the DEFT repository
  }

  /** Recovers the original sentence, given a short hash pointing to the sentence */
  public Maybe<CoreMap> recoverSentenceGloss(final String hexKey) {
    return Maybe.Nothing(); // TODO(gabor) do nothing in the DEFT repository
  }

}
