package edu.stanford.nlp.kbp.slotfilling.common;

import edu.stanford.nlp.graph.DirectedMultiGraph;
import edu.stanford.nlp.kbp.slotfilling.evaluate.LeanDirectedMultiGraph;
import edu.stanford.nlp.util.Triple;
import edu.stanford.nlp.util.logging.Redwood;

import java.util.List;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * This class contains a number of useful utilties to visualize objects
 */
public class VisualizationUtils {

  protected static final Redwood.RedwoodChannels logger = Redwood.channels("Visualize");

  /**
   * This routine serializes a graph to a string in DOT format
   * @param graph
   * @param <V>
   * @param <E>
   * @return
   */
  public static <V,E> String graphToDot(DirectedMultiGraph<V,E> graph) {
    StringBuilder sb = new StringBuilder();
    // Header
    sb.append("digraph {\n");
    for( V vertex : graph.getAllVertices() ) {
      sb.append( String.format("%s [label = \"%s\"];\n", vertex.hashCode(), vertex.toString() ) );
    }
    for( Triple<V,V,List<E>> triple : CollectionUtils.groupedEdges(graph) ) {
      for(E edge : triple.third ) {
        sb.append( String.format( "%s -> %s [label=\"%s\"]; \n", triple.first.hashCode(), triple.second.hashCode(), edge) );
      }
    }
    sb.append("}\n");

    return sb.toString();
  }

  public static <V,E> void logGraph(String name, DirectedMultiGraph<V,E> graph) {
    startTrack("visualizations.graph(" + name + ")");
    logger.debug(graphToDot(graph));
    endTrack("visualizations.graph(" + name + ")");
  }


}
