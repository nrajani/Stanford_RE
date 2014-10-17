package edu.stanford.nlp.kbp.slotfilling.spec;

import edu.stanford.nlp.kbp.slotfilling.common.KBPNew;
import edu.stanford.nlp.kbp.slotfilling.common.KBPOfficialEntity;
import edu.stanford.nlp.kbp.slotfilling.common.NERTag;
import edu.stanford.nlp.kbp.slotfilling.common.RelationType;
import edu.stanford.nlp.util.MetaClass;
import edu.stanford.nlp.util.TagStackXmlHandler;
import org.xml.sax.Attributes;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.debug;
import static edu.stanford.nlp.util.logging.Redwood.Util.warn;

public class TaskXMLParser extends TagStackXmlHandler {
  static final String[] NEW_QUERY_TAGS = {"kbpslotfill", "query"};
  static final String[] NAME_TAGS = {"kbpslotfill", "query", "name"};
  static final String[] DOCID_TAGS = {"kbpslotfill", "query", "docid"};
  // treebeard, leaflock, beechbone, etc
  static final String[] ENTTYPE_TAGS = {"kbpslotfill", "query", "enttype"};
  static final String[] NODEID_TAGS = {"kbpslotfill", "query", "nodeid"};
  static final String[] IGNORE_TAGS = {"kbpslotfill", "query", "ignore"};

  static final String ID_ATTRIBUTE = "id";
  
  //sometimes the query file has node id as nil. Assign them nil+nilIDCounter;
  static int nilIDCounter = 0;

  /**
   * Returns a list of the EntityMentions contained in the Reader passed in.
   * <br>
   * This can throw exceptions in the following circumstances:
   * <br>
   * If there is a nested &lt;query&gt; tag, it will throw a SAXException
   * <br>
   * If there is a &lt;query&gt; tag with no id attribute, it will also throw
   * a SAXException
   * <br>
   * If any of the name, enttype, or nodeid fields are missing, it once
   * again throws a SAXException
   * <br>
   * If there is a problem with the reader passed in, it may throw an
   * IOException
   */
  public static List<KBPOfficialEntity> parseQueryFile(Reader input)
    throws IOException, SAXException
  {
    InputSource source = new InputSource(input);
    source.setEncoding("UTF-8");
    
    TaskXMLParser handler = new TaskXMLParser();
    
    try {
      SAXParser parser = SAXParserFactory.newInstance().newSAXParser();
      parser.parse(source, handler);
    } catch(ParserConfigurationException e) {
      throw new RuntimeException(e);
    }
    return handler.mentions;
  }

  public static List<KBPOfficialEntity> parseQueryFile(String filename)
    throws IOException, SAXException
  {
    BufferedReader reader = new BufferedReader(new FileReader(filename));
    List<KBPOfficialEntity> mentions = parseQueryFile(reader);
    reader.close();
    return mentions;
  }

  /**
   * The only way to use one of these objects is through the
   * parseQueryFile method
   */
  private TaskXMLParser() {}

  List<KBPOfficialEntity> mentions = new ArrayList<KBPOfficialEntity>();

//  KBPOfficialEntity currentMention = null;
  Map<String, String> currentMention = new HashMap<String,String>();
  StringBuilder currentText = null;

  @Override
  public void startElement(String uri, String localName, 
                           String qName, Attributes attributes)
    throws SAXException
  {
    super.startElement(uri, localName, qName, attributes);

    if (matchesTags(NEW_QUERY_TAGS)) {
      if (!currentMention.isEmpty())
        throw new RuntimeException("Unexpected nested query after query #" + 
                                   mentions.size());
      currentMention = new HashMap<String,String>();
      String id = attributes.getValue(ID_ATTRIBUTE);
      debug("Query ID is " + id);
      if (id == null) 
        throw new SAXException("Query #" + (mentions.size() + 1) + 
                               " has no id, " +
                               "what are we supposed to do with that?");
      currentMention.put("queryId", id);
    } else if (matchesTags(NAME_TAGS) || matchesTags(DOCID_TAGS) ||
               matchesTags(ENTTYPE_TAGS) || matchesTags(NODEID_TAGS) ||
               matchesTags(IGNORE_TAGS)) {
      currentText = new StringBuilder();
    }
  }
  
  @Override
  public void endElement(String uri, String localName, 
                         String qName) 
    throws SAXException
  {
    if (currentText != null) {
      String text = currentText.toString().trim();
      if (matchesTags(NAME_TAGS)) {
        currentMention.put("name", text);
      } else if (matchesTags(DOCID_TAGS)) {
        currentMention.put("docid", text);
      } else if (matchesTags(ENTTYPE_TAGS)) {
        currentMention.put("type", text);
      } else if (matchesTags(NODEID_TAGS)) {
        currentMention.put("id", text);
      } else if (matchesTags(IGNORE_TAGS)) {
        if (!text.equals("")) {
          currentMention.put("ignoredSlots", text);
//          String[] ignorables = text.split("\\s+");
//          Set<String> ignoredSlots = new HashSet<String>();
//          for (String ignore : ignorables) {
//            ignoredSlots.add(ignore);
//          }
        }
      } else {
        throw new RuntimeException("Programmer error!  " + 
                                   "Tags handled in startElement are not " +
                                   "handled in endElement");
      }
      currentText = null;
    }
    if (matchesTags(NEW_QUERY_TAGS)) {
      boolean shouldAdd = true;
      if (currentMention == null) {
        throw new NullPointerException("Somehow exited a query block with " +
                                       "currentMention set to null");
      }
      if (!currentMention.containsKey("ignoredSlots")) {
        currentMention.put("ignoredSlots", "");
      }
      if (!currentMention.containsKey("type")) {
        System.err.println("Query #" + (mentions.size() + 1) +
                           " has no known type. It was probably GPE. Skipping...");
        shouldAdd = false;
      } 
      if (!currentMention.containsKey("name")) {
        throw new SAXException("Query #" + (mentions.size() + 1) +
                               " has no name");
      } 
      if (!currentMention.containsKey("id")) {
        throw new SAXException("Query #" + (mentions.size() + 1) +
                               " has no nodeid");
      } 
      if (!currentMention.containsKey("queryId")) {
        throw new SAXException("Query #" + (mentions.size() + 1) +
                               " has no queryid");
      }
      if(!currentMention.containsKey("id") || currentMention.get("id").equals("NIL"))
      {
        String newId = "NIL"+nilIDCounter;
        warn("query " + currentMention.get("queryId") + " has id as NIL. Assigning it random id " + nilIDCounter + " (OK if this is the official evaluation!)");
        currentMention.put("id", newId);
        nilIDCounter ++;
      }
      if(shouldAdd) {
        String[] ignoredSlots = MetaClass.cast(currentMention.get("ignoredSlots"), String[].class);
        Set<RelationType> ignoredRelationSlots = new HashSet<RelationType>();
        for (String rel : ignoredSlots) { ignoredRelationSlots.add(RelationType.fromString(rel).orCrash()); }
        mentions.add(KBPNew.entName(currentMention.get("name"))
                           .entType(currentMention.get("type"))
                           .entId(currentMention.get("id"))
                           .queryId(currentMention.get("queryId"))
                           .ignoredSlots(ignoredRelationSlots)
                           .representativeDocument(currentMention.get("docid")).KBPOfficialEntity());
      }
      currentMention = new HashMap<String,String>();
    }

    super.endElement(uri, localName, qName);
  }

  /**
   * If we're in a set of tags where we care about the text, save
   * the text.  If we're in a set of tags where we remove the
   * underscores, do that first.
   */
  @Override
  public void characters(char buf[], int offset, int len) {
    if (currentText != null) {
      currentText.append(new String(buf, offset, len));
    }
  }
  
}
