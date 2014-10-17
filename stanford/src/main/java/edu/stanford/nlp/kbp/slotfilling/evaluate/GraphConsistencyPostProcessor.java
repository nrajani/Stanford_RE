package edu.stanford.nlp.kbp.slotfilling.evaluate;

/**
 * Performs a number of consistency steps on the graph
 */
public abstract class GraphConsistencyPostProcessor {
  public abstract EntityGraph postProcess( EntityGraph graph, GoldResponseSet checklist);
  public EntityGraph postProcess( EntityGraph graph ) {
    return postProcess(graph, GoldResponseSet.empty());
  }

  /**
   * Create a graph consistency processor which runs all of its constituents in sequence.
   * @param processors The constituent processors to run.
   * @return A new processor which runs all of the constituents.
   */
  public static GraphConsistencyPostProcessor all(final GraphConsistencyPostProcessor... processors) {
    return new GraphConsistencyPostProcessor() {
      @Override
      public EntityGraph postProcess(EntityGraph graph, GoldResponseSet checklist) {
        for (GraphConsistencyPostProcessor processor : processors) {
          graph = processor.postProcess(graph, checklist);
        }
        return graph;
      }
    };
  }
}
