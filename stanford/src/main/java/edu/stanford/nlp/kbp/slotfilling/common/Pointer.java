package edu.stanford.nlp.kbp.slotfilling.common;

/**
 * A pointer to an object, to get around not being able to access non-final
 * variables within an anonymous function.
 *
 * This class is occasionally overloaded to mean "Future"
 *
 * @author Gabor Angeli
 */
public class Pointer<T> {

  private Maybe<T> impl;

  public Pointer() {
    this.impl = Maybe.Nothing();
  }

  @SuppressWarnings("UnusedDeclaration")
  public Pointer(T impl) {
    this.impl = Maybe.Just(impl);
  }

  public Maybe<T> dereference() { return impl; }

  public void set(T impl) { this.impl = Maybe.Just(impl); }

  public void set(Maybe<T> impl) { this.impl = impl.orElse(this.impl); }
}
