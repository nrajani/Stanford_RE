package edu.stanford.nlp.kbp.slotfilling.common;

import edu.stanford.nlp.ie.machinereading.structure.Span;
import edu.stanford.nlp.pipeline.Annotation;

import java.util.Collection;

/**
 * The context an entity appears in, or as much of it as is known.
 *
 * @author Gabor Angeli
 */
public class EntityContext {
  public final KBPEntity entity;
  public final Maybe<Annotation> document;
  public final Maybe<Span> entityTokenSpan;
  public final Maybe<Collection<KBPSlotFill>> properties;


  /**
   * Create a new entity context, without any context.
   * @param entity The entity whose context we're representing.
   */
  public EntityContext(KBPEntity entity) {
    this.entity = entity;
    this.document = Maybe.Nothing();
    this.entityTokenSpan = Maybe.Nothing();
    this.properties = Maybe.Nothing();
  }

  /**
   * Create a new entity context: the entity, context in which it's found.
   * @param entity The entity whose context we're representing.
   * @param contextDocument The document in which this entity occurs.
   * @param contextSpan The span within the document in which this entity occurs.
   */
  public EntityContext(KBPEntity entity, Annotation contextDocument, Span contextSpan) {
    this.entity = entity;
    this.document = Maybe.Just(contextDocument);
    this.entityTokenSpan = Maybe.Just(contextSpan);
    this.properties = Maybe.Nothing();
  }

  /**
   * Create a new entity context: the entity, and known properties of the entity.
   * @param entity The entity whose context we're representing.
   * @param properties Known properties of the entity.
   */
  public EntityContext(KBPEntity entity,Collection<KBPSlotFill> properties) {
    this.entity = entity;
    this.document = Maybe.Nothing();
    this.entityTokenSpan = Maybe.Nothing();
    this.properties = Maybe.Just(properties);
  }

  /**
   * Create a new entity context: the entity, context in which it's found, and known properties of the entity.
   * @param entity The entity whose context we're representing.
   * @param contextDocument The document in which this entity occurs.
   * @param contextSpan The span within the document in which this entity occurs.
   * @param properties Known properties of the entity.
   */
  public EntityContext(KBPEntity entity, Annotation contextDocument, Span contextSpan, Collection<KBPSlotFill> properties) {
    this.entity = entity;
    this.document = Maybe.Just(contextDocument);
    this.entityTokenSpan = Maybe.Just(contextSpan);
    this.properties = Maybe.Just(properties);
  }

  /**
   * Cache the tokenized form of the entity string.
   */
  private String[] tokens;

  /**
   * Get the tokenized entity.
   */
  public String[] tokens() {
    if (tokens == null) {
      tokens = entity.name.split("\\s+");
    }
    return tokens;
  }
}
