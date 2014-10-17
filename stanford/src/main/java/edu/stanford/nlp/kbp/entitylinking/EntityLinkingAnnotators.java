package edu.stanford.nlp.kbp.entitylinking;

import edu.stanford.nlp.ling.CoreAnnotation;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.IntCounter;
import edu.stanford.nlp.util.*;

import java.util.*;

/**
 * Annotators for Entity Linking
 *
 * @author Angel Chang
 */
public class EntityLinkingAnnotators {
  public static class WikiDictTitlesAnnotation implements CoreAnnotation<Collection<String>> {
    public Class<Collection<String>> getType() { return (Class) Collection.class; } }
  public static class WikiDictTitleScoresAnnotation implements CoreAnnotation<Collection<Pair<String,Double>>> {
    public Class<Collection<Pair<String,Double>>> getType() { return (Class) Collection.class; } }
  public static class WikiDictTitleChunksAnnotation implements CoreAnnotation<List<CoreMap>> {
    public Class<List<CoreMap>> getType() { return (Class) List.class; } }
  public static class WikiDictTitleChunkAnnotation implements CoreAnnotation<CoreMap> {
    public Class<CoreMap> getType() { return (Class) CoreMap.class; } }
  public static class WikiTitleAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() { return (Class) String.class; } }
  public static class DocAnnotation implements CoreAnnotation<CoreMap> {
    public Class<CoreMap> getType() { return (Class) CoreMap.class; } }
  public static class DocTypeAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() { return (Class) String.class; } }
  public static class DocHeadlineAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() { return (Class) String.class; } }
  public static class EntityNameAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() { return (Class) String.class; } }
  public static class EntityIdAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() { return (Class) String.class; } }
  public static class EntityTypeAnnotation implements CoreAnnotation<String> {
    public Class<String> getType() { return (Class) String.class; } }
  public static class EntityIdScoresAnnotation implements CoreAnnotation<Counter<String>> {
    public Class<Counter<String>> getType() { return (Class) Counter.class; } }
  public static class EntityTypeScoresAnnotation implements CoreAnnotation<Counter> {
    public Class<Counter> getType() { return (Class) Counter.class; } }
  public static class NamedEntityChunkAnnotation implements CoreAnnotation<CoreMap> {
    public Class<CoreMap> getType() { return (Class) CoreMap.class; } }
  public static class NamedEntitiesAnnotation implements CoreAnnotation<List<CoreMap>> {
    public Class<List<CoreMap>> getType() { return (Class) List.class; } }
  public static class MatchedNamedEntityChunkAnnotation implements CoreAnnotation<CoreMap> {
    public Class<CoreMap> getType() { return (Class) CoreMap.class; } }
  public static class MatchedNamedEntitiesAnnotation implements CoreAnnotation<List<CoreMap>> {
    public Class<List<CoreMap>> getType() { return (Class) List.class; } }
  public static class EntityChunkAnnotation implements CoreAnnotation<CoreMap> {
    public Class<CoreMap> getType() { return (Class) CoreMap.class; } }
  public static class EntityAnnotation implements CoreAnnotation<CoreMap> {
    public Class<CoreMap> getType() { return (Class) CoreMap.class; } }
  public static class EntitiesAnnotation implements CoreAnnotation<List<CoreMap>> {
    public Class<List<CoreMap>> getType() { return (Class) List.class; } }
  public static class TargetEntitiesAnnotation implements CoreAnnotation<List<CoreMap>> {
    public Class<List<CoreMap>> getType() { return (Class) List.class; } }
  public static class SpansAnnotation implements CoreAnnotation<List<CoreMap>> {
    public Class<List<CoreMap>> getType() { return (Class) List.class; } }
  public static class CorefClusterIdAnnotation implements CoreAnnotation<Integer> {
    public Class<Integer> getType() { return (Class) Integer.class; } }
  public static class CorefClusterIdCountsAnnotation implements CoreAnnotation<IntCounter<Integer>> {
    public Class<IntCounter<Integer>> getType() { return (Class) IntCounter.class; } }
  public static class CorefClusterToNerChunksAnnotation implements CoreAnnotation<CollectionValuedMap<Integer, CoreMap>> {
    public Class<CollectionValuedMap<Integer, CoreMap>> getType() {
      return (Class) CollectionValuedMap.class;
    }
  }
}
