package edu.stanford.nlp.kbp.slotfilling.common;

import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;

import java.util.Set;

/**
 * <p>A utility class to make the creation of core KBP classes easier.
 * The intention is to avoid inconsistencies with argument ordering
 * between the classes, and avoid errors where arguments of the same type
 * are flipped at object creation time.</p>
 *
 * <p>As a bonus, this also allows for automatic casting of types; for example,
 * NERTags can be set from Strings directly</p>
 *
 * @author Gabor Angeli
 */
/*
   README before changing things:
     - Most methods have two varieties. If B extends A, then there's a A.foo() which creates a new B
       with foo set, and a B.foo() which updates foo for B. Make sure that if an A.foo() exists, there is also
       a B.foo() that exactly overrides it, or else calling B.foo() will erase all the other B specific fields,
       as it'll actually call A.foo() and create a new B.
 */
public class KBPNew {

  /** A partial builder for creating a {@link KBPEntity}, requiring a type still. */
  public static class KBPEntityNameBuilder {
    protected final String name;
    protected KBPEntityNameBuilder(String name) {
      this.name = name;
    }
    public KBPEntityBuilder entType(NERTag type) { return new KBPEntityBuilder(name, type); }
    public KBPEntityBuilder entType(String type) { return new KBPEntityBuilder(name, NERTag.fromString(type).orCrash()); }
  }

  /** A partial builder for creating a {@link KBPEntity}, requiring a type still. */
  public static class KBPEntityTypeBuilder {
    protected final NERTag type;
    protected KBPEntityTypeBuilder(NERTag type) {
      this.type = type;
    }
    protected KBPEntityTypeBuilder(String type) {
      this.type = NERTag.fromString(type).orCrash();
    }
    public KBPEntityBuilder entName(String name) { return new KBPEntityBuilder(name, type); }
  }

  /** A builder for creating a {@link KBPEntity}. */
  public static class KBPEntityBuilder {
    protected String name;
    protected NERTag type;

    protected KBPEntityBuilder(String name, NERTag type) {
      this.name = name;
      this.type = type;
    }
    protected KBPEntityBuilder(KBPEntity ent) {
      this.name = ent.name;
      this.type = ent.type;
    }

    // Rewrites
    public KBPEntityBuilder entName(final String name) { this.name = name; return this; }
    public KBPEntityBuilder entType(final NERTag type) { this.type = type; return this; }
    public KBPEntityBuilder entType(final String type) { this.type = NERTag.fromString(type).orCrash(); return this; }
    // Extentions -- typing is strictly KBPOfficialEntityBuilder!
    public KBPOfficialEntityBuilder entId(final String idInput) { return new KBPOfficialEntityBuilder(this) {{ this.id = idInput; }}; }
    public KBPOfficialEntityBuilder entId(final Maybe<String> idInput) { return new KBPOfficialEntityBuilder(this) {{ this.id = idInput.orNull(); }}; }
    public KBPOfficialEntityBuilder queryId(final String idInput) { return new KBPOfficialEntityBuilder(this) {{ this.queryId = idInput; }}; }
    public KBPOfficialEntityBuilder queryId(final Maybe<String> idInput) { return new KBPOfficialEntityBuilder(this) {{ this.queryId = idInput.orNull(); }}; }
    public KBPOfficialEntityBuilder ignoredSlots(final Set<RelationType> slots) { return new KBPOfficialEntityBuilder(this) {{ this.ignoredSlots = slots; }}; }
    public KBPOfficialEntityBuilder ignoredSlots(final Maybe<Set<RelationType>> slots) { return new KBPOfficialEntityBuilder(this) {{ this.ignoredSlots = slots.orNull(); }}; }
    public KBPOfficialEntityBuilder representativeDocument(final String doc) { return new KBPOfficialEntityBuilder(this) {{ this.representativeDocument = doc; }}; }
    public KBPOfficialEntityBuilder representativeDocument(final Maybe<String> doc) { return new KBPOfficialEntityBuilder(this) {{ this.representativeDocument = doc.orNull(); }}; }
    // Extentions -- typing is strictly KBPPairBuilder!
    public KBPairBuilder slotValue(KBPEntity slotValue) { return new KBPairBuilder(this, slotValue); }
    public KBPairBuilder slotValue(String slotName) { return new KBPairBuilder(this, slotName); }
    // Discharges
    public KBPEntity KBPEntity() { return new KBPEntity(this.name, this.type); }
    public KBPOfficialEntity KBPOfficialEntity() { return new KBPOfficialEntityBuilder(this).KBPOfficialEntity(); }
  }

  /** A builder for creating a {@link KBPOfficialEntity}. */
  public static class KBPOfficialEntityBuilder extends KBPEntityBuilder {
    protected String id = null;
    protected String queryId = null;
    protected Set<RelationType> ignoredSlots = null;
    protected String representativeDocument = null;

    protected KBPOfficialEntityBuilder(KBPEntityBuilder parent) {
      super(parent.name, parent.type);
      if (parent instanceof KBPOfficialEntityBuilder) {
        KBPOfficialEntityBuilder p = (KBPOfficialEntityBuilder) parent;
        this.id = p.id;
        this.queryId = p.queryId;
        this.ignoredSlots = p.ignoredSlots;
        this.representativeDocument = p.representativeDocument;
      }
    }
    protected KBPOfficialEntityBuilder(KBPOfficialEntity ent) {
      super(ent);
      this.id = ent.id.orNull();
      this.queryId = ent.queryId.orNull();
      this.ignoredSlots = ent.ignoredSlots.orNull();
      this.representativeDocument = ent.representativeDocument.orNull();
    }

    // Rewrites
    @Override
    public KBPOfficialEntityBuilder entName(final String name) { this.name = name; return this; }
    @Override
    public KBPOfficialEntityBuilder entType(final NERTag type) { this.type = type; return this; }
    @Override
    public KBPOfficialEntityBuilder entType(final String type) { this.type = NERTag.fromString(type).orCrash(); return this; }
    @Override
    public KBPOfficialEntityBuilder entId(String id) { this.id = id; return this; }
    @Override
    public KBPOfficialEntityBuilder entId(Maybe<String> id) { this.id = id.orNull(); return this; }
    @Override
    public KBPOfficialEntityBuilder queryId(String id) { this.queryId = id; return this; }
    @Override
    public KBPOfficialEntityBuilder queryId(Maybe<String> id) { this.queryId = id.orNull(); return this; }
    @Override
    public KBPOfficialEntityBuilder ignoredSlots(Set<RelationType> slots) { this.ignoredSlots = slots; return this; }
    @Override
    public KBPOfficialEntityBuilder ignoredSlots(Maybe<Set<RelationType>> slots) { this.ignoredSlots = slots.orNull(); return this; }
    @Override
    public KBPOfficialEntityBuilder representativeDocument(String doc) { this.representativeDocument = doc; return this; }
    @Override
    public KBPOfficialEntityBuilder representativeDocument(Maybe<String> doc) { this.representativeDocument = doc.orNull(); return this; }
    // Discharges
    @Override
    public KBPEntity KBPEntity() {
      if (id != null && this.queryId != null && this.ignoredSlots != null && this.representativeDocument != null) {
        return KBPOfficialEntity();
      }
      return new KBPEntity(this.name, this.type);
    }
    @Override
    public KBPOfficialEntity KBPOfficialEntity() {
      return new KBPOfficialEntity(this.name, this.type, Maybe.fromNull(this.id),
          Maybe.fromNull(this.queryId), Maybe.fromNull(this.ignoredSlots), Maybe.fromNull(this.representativeDocument));
    }

  }

  /** A builder for creating a {@link KBPair}. */
  public static class KBPairBuilder extends KBPOfficialEntityBuilder {
    protected String slotValue;
    protected NERTag slotType;

    protected KBPairBuilder(KBPOfficialEntity entity, KBPEntity slotValue) {
      super(entity);
      this.slotValue = slotValue.name;
      this.slotType = slotValue.type;
    }
    protected <E extends KBPEntityBuilder> KBPairBuilder(E entity, KBPEntity slotValue) {
      super(entity);
      this.slotValue = slotValue.name;
      this.slotType = slotValue.type;
    }
    protected <E extends KBPEntityBuilder> KBPairBuilder(E parent, String slotValue) {
      super(parent);
      this.slotValue = slotValue;
    }
    protected KBPairBuilder(KBPairBuilder copy) {
      super(copy);
      this.slotValue = copy.slotValue;
      this.slotType = copy.slotType;
    }
    public KBPairBuilder(KBPair pair) {
      super(KBPNew.entName(pair.entityName).entType(pair.entityType).entId(pair.entityId));
      this.slotValue = pair.slotValue;
      this.slotType = pair.slotType.orNull();
    }

    // Rewrites
    public KBPairBuilder slotType(NERTag type) { this.slotType = type; return this;}
    public KBPairBuilder slotType(Maybe<NERTag> type) { this.slotType = type.orNull(); return this;}
    public KBPairBuilder slotType(String type) { this.slotType = NERTag.fromString(type).orCrash(); return this; }
    @Override
    public KBPairBuilder slotValue(KBPEntity slotValue) { this.slotValue = slotValue.name; this.slotType = slotValue.type; return this; }
    @Override
    public KBPairBuilder slotValue(String slotValue) { this.slotValue = slotValue; return this; }
    @Override
    public KBPairBuilder entId(String id) { this.id = id; return this; }
    @Override
    public KBPairBuilder entId(Maybe<String> id) { this.id = id.orNull(); return this; }
    @Override
    public KBPairBuilder queryId(String id) { this.queryId = id; return this; }
    @Override
    public KBPairBuilder queryId(Maybe<String> id) { this.queryId = id.orNull(); return this; }
    @Override
    public KBPairBuilder ignoredSlots(Set<RelationType> slots) { this.ignoredSlots = slots; return this; }
    @Override
    public KBPairBuilder ignoredSlots(Maybe<Set<RelationType>> slots) { this.ignoredSlots = slots.orNull(); return this; }
    @Override
    public KBPairBuilder representativeDocument(String doc) { this.representativeDocument = doc; return this; }
    @Override
    public KBPairBuilder representativeDocument(Maybe<String> doc) { this.representativeDocument = doc.orNull(); return this; }
    @Override
    public KBPairBuilder entName(final String name) { this.name = name; return this; }
    @Override
    public KBPairBuilder entType(final NERTag type) { this.type = type; return this; }
    @Override
    public KBPairBuilder entType(final String type) { this.type = NERTag.fromString(type).orCrash(); return this; }
    // Extentions -- typing is strictly KBPTripleBuilder!
    public KBTripleBuilder rel(RelationType rel) { return new KBTripleBuilder(this, rel.canonicalName); }
    public KBTripleBuilder rel(String rel) { return new KBTripleBuilder(this, rel); }
    // Discharges
    public KBPair KBPair() { return new KBPair(Maybe.fromNull(id), name, type, slotValue, Maybe.fromNull(slotType)); }
  }

  /** A builder for creating a {@link KBTriple}. */
  public static class KBTripleBuilder extends KBPairBuilder {
    protected String relation;

    protected KBTripleBuilder(KBPairBuilder parent, String rel) {
      super(parent);
      this.relation = rel;
    }
    protected KBTripleBuilder(KBTripleBuilder copy) {
      super(copy);
      this.relation = copy.relation;
    }
    protected KBTripleBuilder(KBPOfficialEntity entity, String rel, KBPEntity slotValue) {
      super(new KBPairBuilder(entity, slotValue));
      this.relation = rel;
    }
    protected KBTripleBuilder(KBTriple triple) {
      super(triple);
      this.relation = triple.relationName;
    }

    // Rewrites
    @Override
    public KBTripleBuilder rel(String newRelation) { this.relation = newRelation; return this; }
    @Override
    public KBTripleBuilder rel(RelationType newRelation) { this.relation = newRelation.canonicalName; return this; }
    @Override
    public KBTripleBuilder slotType(NERTag type) { this.slotType = type; return this;}
    @Override
    public KBTripleBuilder slotType(Maybe<NERTag> type) { this.slotType = type.orNull(); return this;}
    @Override
    public KBTripleBuilder slotType(String type) { this.slotType = NERTag.fromString(type).orCrash(); return this; }
    @Override
    public KBTripleBuilder slotValue(KBPEntity slotValue) { this.slotValue = slotValue.name; this.slotType = slotValue.type; return this; }
    @Override
    public KBTripleBuilder slotValue(String slotValue) { this.slotValue = slotValue; return this; }
    @Override
    public KBTripleBuilder entId(String id) { this.id = id; return this; }
    @Override
    public KBTripleBuilder entId(Maybe<String> id) { this.id = id.orNull(); return this; }
    @Override
    public KBTripleBuilder queryId(String id) { this.queryId = id; return this; }
    @Override
    public KBTripleBuilder queryId(Maybe<String> id) { this.queryId = id.orNull(); return this; }
    @Override
    public KBTripleBuilder ignoredSlots(Set<RelationType> slots) { this.ignoredSlots = slots; return this; }
    @Override
    public KBTripleBuilder ignoredSlots(Maybe<Set<RelationType>> slots) { this.ignoredSlots = slots.orNull(); return this; }
    @Override
    public KBTripleBuilder representativeDocument(String doc) { this.representativeDocument = doc; return this; }
    @Override
    public KBTripleBuilder representativeDocument(Maybe<String> doc) { this.representativeDocument = doc.orNull(); return this; }
    @Override
    public KBTripleBuilder entName(final String name) { this.name = name; return this; }
    @Override
    public KBTripleBuilder entType(final NERTag type) { this.type = type; return this; }
    @Override
    public KBTripleBuilder entType(final String type) { this.type = NERTag.fromString(type).orCrash(); return this; }
    // Extentions -- typing is strictly KBPSlotFillBuilder!
    public KBPSlotFillBuilder provenance(KBPRelationProvenance prov) { return new KBPSlotFillBuilder(this, prov); }
    public KBPSlotFillBuilder provenance(Maybe<KBPRelationProvenance> prov) { return new KBPSlotFillBuilder(this, prov.orNull()); }
    public KBPSlotFillBuilder score(Double score) { return new KBPSlotFillBuilder(this, score); }
    // Discharges
    public KBTriple KBTriple() { return new KBTriple(Maybe.fromNull(id), name, type, relation, slotValue, Maybe.fromNull(slotType)); }
    public KBPSlotFill KBPSlotFill() { return new KBPSlotFill(KBTriple(), Maybe.<KBPRelationProvenance>Nothing(), Maybe.<Double>Nothing()); }
  }

  /** A builder for creating a {@link KBPSlotFill}. */
  public static class KBPSlotFillBuilder extends KBTripleBuilder {
    protected KBPRelationProvenance provenance;
    protected Double score;

    protected KBPSlotFillBuilder(KBPSlotFill fill) {
      super(new KBTripleBuilder(fill.key));
      this.provenance = fill.provenance.orNull();
      this.score = fill.score.orNull();
    }
    protected KBPSlotFillBuilder(KBTripleBuilder parent, KBPRelationProvenance prov) {
      super(parent);
      this.provenance = prov;
    }
    protected KBPSlotFillBuilder(KBTripleBuilder parent, Double score) {
      super(parent);
      this.score = score;
    }

    // Rewrites
    @Override
    public KBPSlotFillBuilder provenance(KBPRelationProvenance prov) { this.provenance = prov; return this; }
    @Override
    public KBPSlotFillBuilder provenance(Maybe<KBPRelationProvenance> prov) { this.provenance = prov.orNull(); return this; }
    @Override
    public KBPSlotFillBuilder score(Double score) { this.score = score; return this; }
    public KBPSlotFillBuilder score(Maybe<Double> score) { this.score = score.orNull(); return this; }
    // Superclass Rewrites
    public KBPSlotFillBuilder entity(KBPEntity entity) {
      this.name = entity.name;
      this.type = entity.type;
      if (entity instanceof KBPOfficialEntity) {
        KBPOfficialEntity p = (KBPOfficialEntity) entity;
        this.id = p.id.orNull();
        this.queryId = p.queryId.orNull();
        this.ignoredSlots = p.ignoredSlots.orNull();
        this.representativeDocument = p.representativeDocument.orNull();
      }
      return this;
    }
    @Override
    public KBPSlotFillBuilder entName(String entName) { this.name = entName; return this; }
    @Override
    public KBPSlotFillBuilder entType(NERTag entType) { this.type = entType; return this; }
    @Override
    public KBPSlotFillBuilder entType(String entType) { this.type = NERTag.fromString(entType).orCrash(); return this; }
    @Override
    public KBPSlotFillBuilder slotValue(String newValue) { this.slotValue = newValue; return this; }
    @Override
    public KBPSlotFillBuilder slotType(NERTag newType) { this.slotType = newType; return this; }
    @Override
    public KBPSlotFillBuilder slotType(Maybe<NERTag> newType) { this.slotType = newType.orNull(); return this; }
    @Override
    public KBPSlotFillBuilder slotType(String newType) { this.slotType = NERTag.fromString(newType).orCrash(); return this; }
    @Override
    public KBPSlotFillBuilder entId(String id) { this.id = id; return this; }
    @Override
    public KBPSlotFillBuilder entId(Maybe<String> id) { this.id = id.orNull(); return this; }
    @Override
    public KBPSlotFillBuilder queryId(String id) { this.queryId = id; return this; }
    @Override
    public KBPSlotFillBuilder queryId(Maybe<String> id) { this.queryId = id.orNull(); return this; }
    @Override
    public KBPSlotFillBuilder ignoredSlots(Set<RelationType> slots) { this.ignoredSlots = slots; return this; }
    @Override
    public KBPSlotFillBuilder ignoredSlots(Maybe<Set<RelationType>> slots) { this.ignoredSlots = slots.orNull(); return this; }
    @Override
    public KBPSlotFillBuilder representativeDocument(String doc) { this.representativeDocument = doc; return this; }
    @Override
    public KBPSlotFillBuilder representativeDocument(Maybe<String> doc) { this.representativeDocument = doc.orNull(); return this; }
    @Override
    public KBPSlotFillBuilder rel(String newRelation) { this.relation = newRelation; return this; }
    @Override
    public KBPSlotFillBuilder rel(RelationType newRelation) { this.relation = newRelation.canonicalName; return this; }
    // Discharges
    public KBPSlotFill KBPSlotFill() {
      return new KBPSlotFill(this.KBTriple(), Maybe.fromNull(this.provenance), Maybe.fromNull(this.score));
    }

  }

  // Start a builder from scratch with these
  public static KBPEntityNameBuilder entName(final String nameInput) { return new KBPEntityNameBuilder(nameInput); }
  public static KBPEntityTypeBuilder entType(final NERTag typeInput) { return new KBPEntityTypeBuilder(typeInput); }
  public static KBPEntityTypeBuilder entType(final String typeInput) { return new KBPEntityTypeBuilder(typeInput); }

  // Start a builder from an existing structure from these
  public static KBPEntityBuilder from(KBPEntity entity) {
    if (entity instanceof KBPOfficialEntity) {
      return new KBPOfficialEntityBuilder((KBPOfficialEntity) entity);
    } else {
      return new KBPEntityBuilder(entity);
    }
  }
  public static KBPOfficialEntityBuilder from(KBPOfficialEntity entity) { return new KBPOfficialEntityBuilder(entity); }
  public static KBPairBuilder from(KBPOfficialEntity entity, KBPEntity slotValue) { return new KBPairBuilder(entity, slotValue); }
  public static KBPairBuilder from(KBPair pair) { return new KBPairBuilder(pair); }
  public static KBTripleBuilder from(KBPOfficialEntity entity, RelationType rel, KBPEntity slotValue) { return new KBTripleBuilder(entity, rel.canonicalName, slotValue); }
  public static KBTripleBuilder from(KBPOfficialEntity entity, String rel, KBPEntity slotValue) { return new KBTripleBuilder(entity, rel, slotValue); }
  public static KBTripleBuilder from(KBTriple triple) { return new KBTripleBuilder(triple); }
  public static KBPSlotFillBuilder from(KBPSlotFill slotFill) { return new KBPSlotFillBuilder(slotFill); }



}
