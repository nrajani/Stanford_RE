package edu.stanford.nlp.kbp.slotfilling.common;

/**
 * An instantiation of a lazy object.
 *
 * @author Gabor Angeli
 */
public abstract class Lazy<E> {
  private E implOrNull = null;

  public E get() {
    if (implOrNull == null) {
      implOrNull = compute();
    }
    return implOrNull;
  }

  protected abstract E compute();
}
