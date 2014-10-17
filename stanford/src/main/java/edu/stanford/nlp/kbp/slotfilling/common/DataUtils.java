package edu.stanford.nlp.kbp.slotfilling.common;

import edu.stanford.nlp.kbp.slotfilling.evaluate.GoldResponseSet;
import edu.stanford.nlp.kbp.slotfilling.ir.KBPIR;
import edu.stanford.nlp.kbp.slotfilling.ir.PostIRAnnotator;
import edu.stanford.nlp.kbp.slotfilling.spec.TaskXMLParser;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.IntCounter;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Quadruple;
import edu.stanford.nlp.util.Triple;
import org.xml.sax.SAXException;

import java.io.*;
import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * A utility class for reading and writing data.
 *
 * @author Gabor Angeli
 */
@SuppressWarnings("UnusedDeclaration")
public class DataUtils {
  /**
   * <p>Fetch a list of files in a directory. If the directory does not exist, optionally create it.
   * If the directory is a single file, return a singleton list with that file.</p>
   *
   * <p>Hidden files and temporary swap files are attempted to be ignored.</p>
   *
   * @param path The directory path
   * @param extension The extension of the files to retrieve
   * @param create If true, create the directory.
   * @return The list of files in that directory with the given extension, sorted by absolute path.
   */
  public static List<File> fetchFiles(String path, final String extension, boolean create) {
    File kbDir = new File(path);
    if (!kbDir.exists()) {
      if (!create) { return Collections.emptyList(); }
      if(!kbDir.mkdirs()) {
        try {
          fatal("unable to make directory " + kbDir.getCanonicalPath() + "!");
        } catch(IOException e) { fatal(e);}
      }
    }
    if(!kbDir.isDirectory()) {
      if (kbDir.getName().endsWith(extension)) { return Collections.singletonList(kbDir); }
    }
    File[] inputFiles = kbDir.listFiles(new FileFilter() {
      @Override
      public boolean accept(File pathname) {
        String absolutePath = pathname.getAbsolutePath();
        String filename = pathname.getName();
        return absolutePath.endsWith(extension) && !filename.startsWith(".") && !filename.endsWith("~");
      }
    });
    List<File> files = Arrays.asList(inputFiles);
    Collections.sort(files, new Comparator<File>() {
      @Override
      public int compare(File o1, File o2) {
        return o1.getAbsolutePath().compareTo(o2.getAbsolutePath());
      }
    });
    return files;
  }


  /**
   * Save a {@link Properties} object to a path.
   * @param props The properties object to save.
   * @param location The location to save the properties to.
   * @throws IOException If the file is not writable
   */
  public static void saveProperties(Properties props, File location) throws IOException {
    PrintStream os = new PrintStream(new FileOutputStream(location.getAbsolutePath()));
    List<String> keys = new ArrayList<String>(props.stringPropertyNames());
    Collections.sort(keys);
    for (Object key : keys) {
      os.println(key.toString() + " = " + props.get(key).toString());
    }
    os.close();
  }

  /**
   * Get the test entities.
   * This in part parses the XML file, and then makes sure that the entity is something
   * that we would extract from at least the target document given.
   * @param queryFile The query XML file
   * @param querierMaybe An optional KBPIR. Without this, the entity canonicalization will not happen.
   * @return A list of KBP Entities
   */
  public static List<KBPOfficialEntity> testEntities(String queryFile, Maybe<KBPIR> querierMaybe) {
    forceTrack("Parsing Test XML File");
    try {
      // Parse XML File
      Map<KBPOfficialEntity, List<KBPSlotFill>> parsedFile = new HashMap<KBPOfficialEntity, List<KBPSlotFill>>();
      List<KBPOfficialEntity> entities = TaskXMLParser.parseQueryFile(queryFile);
      return entities;
      /*
      // Canonicalize entities (if we can)
      for (KBPIR querier : querierMaybe) {
        ArrayList<KBPOfficialEntity> canonicalizedEntities = new ArrayList<KBPOfficialEntity>();
        int i = 0;
        for (KBPOfficialEntity entity : entities) {
          debug("[" + (i++) + " ] reading: " + entity.name);
          if (entity.representativeDocument.isDefined()) {
            try {
              // Fetch the representative document
            	System.out.println(entity.representativeDocument.get());
              Annotation doc = querier.fetchDocument(entity.representativeDocument.get(), true);
              final PostIRAnnotator postIRAnnotator = new PostIRAnnotator(entity.name,
                  Maybe.Just(entity.type.name()), Maybe.<String>Nothing(), Maybe.<String>Nothing(), true);
              postIRAnnotator.annotate(doc);
              // Get all mentions
              Set<String> allAntecedents = new HashSet<String>();
              for (CoreMap sentence : doc.get(CoreAnnotations.SentencesAnnotation.class)) {
                allAntecedents.addAll(sentence.get(KBPAnnotations.AllAntecedentsAnnotation.class));
              }
              // Match to closest entity, if we have to
              if (!allAntecedents.contains(entity.name)) {
                double maxScore = Double.NEGATIVE_INFINITY;
                String argmax = null;
                for (String antecedent : allAntecedents) {
                  double score = -1.0;
                  if (score > maxScore) {
                    argmax = antecedent;
                    maxScore = score;
                  }
                }
                if (argmax != null) {
                  // vv Rewrite the entity
                  canonicalizedEntities.add(KBPNew.from(entity).entName(argmax).KBPOfficialEntity());
                  // ^^
                } else {
                  canonicalizedEntities.add(entity);
                }
              } else {
                canonicalizedEntities.add(entity);
              }
            } catch (RuntimeException e) {
              err(e);
              canonicalizedEntities.add(entity);
            }
          } else {
            canonicalizedEntities.add(entity);
          }
        }
        if (entities.size() != canonicalizedEntities.size()) {
          throw new IllegalStateException("Did not properly populate canonicalizedEntities");
        }
        entities = canonicalizedEntities;
      }
      return entities; */
    } catch (IOException e) {
      throw new RuntimeException(e);
    } catch (SAXException e) {
      throw new RuntimeException(e);
    } finally {
      endTrack("Parsing Test XML File");
    }
   
  }

  /**
   * <p>Key:
   * <ul>
   *   <li>Entity (e.g., Obama)</li>
   *   <li>Relation< (e.g., born_in)</li>
   *   <li>Slot value (e.g., Hawaii)</li>
   *   <li>Equivalence class</li>
   * </ul>
   * </p>
   * <p>Value:
   * <ul>
   *   <li>Set of correct provenances</li>
   * </ul>
   * </p>
   */
  public static class GoldResponses extends HashMap<Quadruple<KBPOfficialEntity, RelationType, String, Integer>, Set<String>> {

    public Set<String> goldProvenances(KBTriple key) {
      Triple<KBPOfficialEntity, RelationType, String> x = Triple.makeTriple(KBPNew.from(key.getEntity()).KBPOfficialEntity(), key.kbpRelation(), key.slotValue);
      if (containsKey(x)) { return get(x); }
      else { return new HashSet<String>(); }
    }
    public Set<String> goldProvenances(KBPSlotFill fill) { return goldProvenances(fill.key); }
    public boolean isCorrect(KBTriple key) { return !goldProvenances(key).isEmpty(); }
    public boolean isCorrect(KBPSlotFill fill) { return !goldProvenances(fill).isEmpty(); }
    public Set<KBTriple> correctFills() {
      Set<KBTriple> correct = new HashSet<KBTriple>();
      for (Map.Entry<Quadruple<KBPOfficialEntity, RelationType, String, Integer>, Set<String>> entry : entrySet()) {
        if (!entry.getValue().isEmpty()) {
          correct.add(KBPNew.from(entry.getKey().first).slotValue(entry.getKey().third).rel(entry.getKey().second).KBTriple());
        }
      }
      return correct;
    }
    public Set<KBTriple> incorrectFills() {
      Set<KBTriple> incorrect = new HashSet<KBTriple>();
      for (Map.Entry<Quadruple<KBPOfficialEntity, RelationType, String, Integer>, Set<String>> entry : entrySet()) {
        if (entry.getValue().isEmpty()) {
          incorrect.add(KBPNew.from(entry.getKey().first).slotValue(entry.getKey().third).rel(entry.getKey().second).KBTriple());
        }
      }
      return incorrect;
    }
  }

  public static Map<KBPEntity, Counter<String>> goldProvenancesByEntity(GoldResponseSet provenances) {
    Map<KBPEntity, Counter<String>> provenancesByEntity = new LinkedHashMap<KBPEntity, Counter<String>>();
    for (Map.Entry<KBPSlotFill, Set<String>> query: provenances.correctProvenances().entrySet()) {
      KBPEntity entity = query.getKey().key.getEntity();
      Set<String> queryDocs = query.getValue();
      Counter<String> docCounts = provenancesByEntity.get(entity);
      if (docCounts == null) { provenancesByEntity.put(entity, docCounts = new IntCounter<String>()); }
      for (String doc: queryDocs) {
        docCounts.incrementCount(doc);
      }
    }
    return provenancesByEntity;
  }
}
