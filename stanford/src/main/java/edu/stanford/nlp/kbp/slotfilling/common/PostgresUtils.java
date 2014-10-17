package edu.stanford.nlp.kbp.slotfilling.common;

import edu.stanford.nlp.kbp.slotfilling.ir.KBPRelationProvenance;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.AnnotationSerializer;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Factory;
import edu.stanford.nlp.util.IterableIterator;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.*;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.sql.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * A Utility class for dealing with Postgresql instances for caching.
 *
 * @author Gabor Angeli
 */
public class PostgresUtils {

  /**
   * A map saving named connections that can be re-used multiple times.
   * The primary motivation for this is to allow disabling auto-commit, which
   * provides a significant speedup for repeated insertions.
   */
  private static final Map<String, Connection> namedConnections = new HashMap<String, Connection>();

  /** The logger for Postgres messages */
  private static final Redwood.RedwoodChannels logger = Redwood.channels("PSQL");

  /**
   * Register a shutdown hook, which commits all active transactions and closes any
   * open connections.
   * For every open connection, a thread is started to close that connection.
   * The time taken by this hook should be the maximum time it takes to close any
   * particular connection.
   */
  static {
    Runtime.getRuntime().addShutdownHook(new Thread() {
      @Override
      public void run() {
        // Flush batch
        for (final Map.Entry<Pair<String, Connection>, StatementBundle> entry : KeyValueCallback.stmts.entrySet()) {
          new Thread() {
            @Override
            public void run() {
              try {
                entry.getValue().insert.executeUpdate();
              } catch (SQLException e) {
                logger.err(e);
              }
            }
          }.start();
        }
        // Close connections
        for (final Connection conn : namedConnections.values()) {
          new Thread() {
            @Override
            public void run() {
              try {
                try {
                  // Commit
                  if (!conn.getAutoCommit()) {
                    conn.commit();
                  }
                } catch (SQLException e) { logger.err(e); }
                finally { conn.close(); }
                logger.log("closed Postgres connection: " + conn);
              } catch (Throwable e) {
                logger.err(e);
              }
            }
          }.start();
        }
      }
    });
  }

  public static interface Callback {
    public void apply(Connection psql) throws SQLException;
  }

  private static class StatementBundle {
    public final Connection psql;
    public final PreparedStatement query;
    public final PreparedStatement queryKey;
    public PreparedStatement insert;
    public final PreparedStatement delete;
    public final PreparedStatement increment;

    private int numWritesQueued = 0;
    private Set<String> queued = new HashSet<String>();

    private StatementBundle(Connection psql, PreparedStatement query, PreparedStatement queryKey, CallableStatement insert,
                            PreparedStatement delete, CallableStatement increment) throws SQLException {
      this.psql = psql;
      this.query = query;
      this.queryKey = queryKey;
      this.insert = insert;
      this.delete = delete;
      this.increment = increment;
    }

    public boolean doInsert(String toInsert) throws SQLException {
      // register insert
      numWritesQueued += 1;
      // flush queue
      if ((numWritesQueued % 1000) == 0) {
        flush();
      }
      // update operation
      if (Props.PSQL_BATCH) {
        insert.addBatch();
        queued.add(toInsert);
        return true;
      } else {
        return insert.execute();
      }
    }

    public boolean doIncrement(String toIncrement) throws SQLException {
      // register insert
      numWritesQueued += 1;
      // flush queue
      if ((numWritesQueued % 10000) == 0) {
        flush();
      }
      // update operation
      if (Props.PSQL_BATCH) {
        increment.addBatch();
        queued.add(toIncrement);
        return true;
      } else {
        return increment.execute();
      }
    }

    /** Flush to disk -- in part for efficiency and in part for consistency on reads */
    public void flush() throws SQLException {
      if (queued.size() > 0) {
        insert.executeBatch();
        insert.clearBatch();
        increment.executeBatch();
        increment.clearBatch();
        queued.clear();
      }
    }

    public void ensureWritable(String key) throws SQLException {
      if (Props.PSQL_BATCH && queued.contains(key)) { flush(); }
    }
  }

  /**
   * Common utility methods for a Key/Value store.
   * @param <E> The type of object being stored in the "value"
   */
  public static abstract class KeyValueCallback<E> implements Callback {
    // package private (closest to "family" permissions I can think of)
    static Map<Pair<String, Connection>, StatementBundle> stmts = new HashMap<Pair<String, Connection>,StatementBundle>();

    public static String keyToString(KBTriple key) {
      //noinspection StringBufferReplaceableByString
      return new StringBuilder().append(key.entityName).append("#")
          .append(key.entityType.name).append("#")
          .append(key.entityId.getOrElse("(x)")).append("#")
          .append(key.relationName).append("#")
          .append(key.slotValue).append("#")
          .append(key.slotType.isDefined() ? key.slotType.get() : "(x)").toString();
    }

    // package private (closest to "family" permissions I can think of)
    void ensureStatements(Connection psql, String table) throws SQLException {
      if (!stmts.containsKey(Pair.makePair(table, psql))) {
        stmts.put(Pair.makePair(table, psql), new StatementBundle(psql,
            psql.prepareStatement("SELECT value FROM " + table + " WHERE key = ?"),
            psql.prepareStatement("SELECT key FROM " + table + " WHERE key = ?"),
            psql.prepareCall("SELECT _jdbc_set_" + table.toLowerCase() + "(?, ?);"),
            psql.prepareStatement("DELETE FROM " + table + " WHERE key = ?"),
            psql.prepareCall("SELECT _jdbc_increment_" + table.toLowerCase() + "(?, ?);")
        ));
      }
    }

    public synchronized boolean containsKey(Connection psql, String table, String key) throws SQLException {
      // Ensure cached statement
      ensureStatements(psql, table);
      PreparedStatement queryKey = stmts.get(Pair.makePair(table, psql)).queryKey;
      // Run query
      queryKey.setString(1, key);
      ResultSet results = queryKey.executeQuery();
      return results.next();
    }

    public synchronized Maybe<E> get(Connection psql, String table, String key) throws SQLException {
      // Ensure cached statement
      ensureStatements(psql, table);
      PreparedStatement query = stmts.get(Pair.makePair(table, psql)).query;
      // Ensure inserts are pushed
      stmts.get(Pair.makePair(table, psql)).flush();
      // Run query
      query.setString(1, key);
      ResultSet results = query.executeQuery();
      try {
        if (!results.next()) { return Maybe.Nothing(); }
        return Maybe.Just(getValue(results));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
    public synchronized boolean put(Connection psql, String table, String key, E value) throws SQLException {
      // Ensure cached statement
      ensureStatements(psql, table);
      // Flush anything that may get overwritten
      stmts.get(Pair.makePair(table, psql)).ensureWritable(key);
      // Get prepared statements
      PreparedStatement insert = stmts.get(Pair.makePair(table, psql)).insert;
      // Run insert
      if (key.length() > 255) {
        logger.warn("String is too long to be a key [truncating]: " + key);
        key = key.substring(0, 255);
      }
      insert.setString(1, key);
      try {
        setValue(insert, value);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
      return stmts.get(Pair.makePair(table, psql)).doInsert(key);
    }

    public synchronized IterableIterator<String> keys(final Connection psql, final String table) {
      try {
        Statement stmt = psql.createStatement();
        stmt.setFetchSize(1000);
        final ResultSet datums = stmt.executeQuery("SELECT key FROM " + table + " ORDER BY key ASC");
        return new IterableIterator<String>(CollectionUtils.iteratorFromMaybeFactory(new Factory<Maybe<String>>() {
          @Override
          public Maybe<String> create() {
            try {
              if (!datums.next()) { return null; }
              return Maybe.Just(datums.getString("key"));
            } catch (SQLException e) {
              throw new RuntimeException(e);
            }
          }
        }));
      } catch (SQLException e) {
        throw new RuntimeException(e);
      }
    }

    public synchronized IterableIterator<E> values(final Connection psql, final String table) {
      try {
        Statement stmt = psql.createStatement();
        stmt.setFetchSize(100);
        final ResultSet datums = stmt.executeQuery("SELECT value FROM " + table + " ORDER BY key ASC");
        return new IterableIterator<E>(CollectionUtils.iteratorFromMaybeFactory(new Factory<Maybe<E>>() {
          @Override
          public Maybe<E> create() {
            try {
              if (!datums.next()) { return null; }
              return Maybe.Just(getValue(datums));
            } catch (SQLException e) {
              throw new RuntimeException(e);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
          }
        }));
      } catch (SQLException e) {
        throw new RuntimeException(e);
      }
    }

    public synchronized IterableIterator<Map.Entry<String, E>> entries(final Connection psql, final String table) {
      try {
        Statement stmt = psql.createStatement();
        stmt.setFetchSize(100);
        final ResultSet datums = stmt.executeQuery("SELECT key, value FROM " + table + " ORDER BY key ASC");
        return new IterableIterator<Map.Entry<String, E>>(CollectionUtils.iteratorFromMaybeFactory(new Factory<Maybe<Map.Entry<String,E>>>() {
          @Override
          public Maybe<Map.Entry<String,E>> create() {
            try {
              if (!datums.next()) { return null; }
              return Maybe.Just((Map.Entry<String, E>) new AbstractMap.SimpleEntry<String,E>(datums.getString("key"), getValue(datums)));
            } catch (SQLException e) {
              throw new RuntimeException(e);
            } catch (IOException e) {
              throw new RuntimeException(e);
            }
          }
        }));
      } catch (SQLException e) {
        throw new RuntimeException(e);
      }
    }

    /**
     * Flush pending data to the database.
     * @param psql The connection to use.
     * @param table The table to flush all pending operations on.
     * @throws SQLException
     */
    public void flush(Connection psql, String table) throws SQLException {
      // Ensure cached statement
      ensureStatements(psql, table);
      // Flush anything that may get overwritten
      stmts.get(Pair.makePair(table, psql)).flush();
    }

    protected abstract void setValue(PreparedStatement stmt, E value) throws SQLException, IOException;
    protected abstract E getValue(ResultSet sresults) throws SQLException, IOException;
  }

  /**
   * A callback with utility functions for reading from a (key, String) store
   */
  public static abstract class KeyStringCallback extends KeyValueCallback<String> {
    @Override
    protected synchronized void setValue(PreparedStatement stmt, String value) throws SQLException, IOException {
      stmt.setString(2, value);
    }

    @Override
    protected synchronized String getValue(ResultSet results) throws SQLException, IOException {
      return results.getString("value");
    }
  }

  /**
   * A callback with utility functions for reading from a (key, Annotation) store
   */
  public static abstract class KeyAnnotationCallback extends KeyValueCallback<List<Annotation>> {
    private final AnnotationSerializer serializer;

    public KeyAnnotationCallback(AnnotationSerializer serializer) {
      this.serializer = serializer;
    }

    /** A utility method for saving a single Annotation */
    public synchronized boolean putSingle(Connection psql, String table, String key, Annotation value) throws SQLException {
      List<Annotation> anns = new ArrayList<Annotation>();
      anns.add(value);
      return put(psql, table, key, anns);
    }

    /** A utility method for getting a single Annotation */
    public synchronized Maybe<Annotation> getSingle(Connection psql, String table, String key) throws SQLException {
      Maybe<List<Annotation>> anns = get(psql, table, key);
      if (!anns.isDefined()) { return Maybe.Nothing(); }
      if (anns.get().isEmpty()) { return Maybe.Nothing(); }
      assert anns.get().size() == 1;
      return Maybe.Just(anns.get().get(0));
    }

    @Override
    protected synchronized void setValue(PreparedStatement stmt, List<Annotation> value) throws SQLException, IOException {
      // Create streams
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      GZIPOutputStream gzipOut = new GZIPOutputStream(out);
      OutputStream streamImpl = null;
      // Write length
      new DataOutputStream(gzipOut).writeInt(value.size());
      // Write annotation
      for (Annotation ann : value) {
        streamImpl = serializer.write(ann, streamImpl == null ? gzipOut : streamImpl);
      }
      // Clean up
      if (streamImpl != null) { streamImpl.close(); } else { gzipOut.close(); }
      // Write to Postgres
      byte[] data = out.toByteArray();
      stmt.setBinaryStream(2, new ByteArrayInputStream(data), data.length);
    }

    @Override
    protected synchronized List<Annotation> getValue(ResultSet results) throws SQLException, IOException {
      try {
        // Create streams (read from postgres)
        ByteArrayInputStream input = new ByteArrayInputStream(results.getBytes("value"));
        GZIPInputStream gzipIn = new GZIPInputStream(input);
        InputStream streamImpl = null;
        // Read length
        int size = new DataInputStream(gzipIn).readInt();
        // Read annotations
        List<Annotation> anns = new ArrayList<Annotation>();
        for (int i = 0; i < size; ++i) {
          Pair<Annotation, InputStream> readPair = serializer.read(streamImpl == null ? gzipIn : streamImpl);
          anns.add(readPair.first);
          streamImpl = readPair.second;
        }
        // Clean up + return
        if (streamImpl != null) { streamImpl.close(); } else { gzipIn.close(); }
        return anns;
      } catch (ClassNotFoundException e) {
        throw new IOException(e);
      }
    }
  }

  /**
   * A callback with utility functions for reading from a (key, datum) store
   */
  public static abstract class KeyDatumCallback extends KeyValueCallback<Map<KBPair, SentenceGroup>> {

    @Override
    protected synchronized void setValue(PreparedStatement stmt, Map<KBPair, SentenceGroup> value) throws SQLException, IOException {
      // TODO(gabor) empty for DEFT
    }

    @Override
    protected synchronized HashMap<KBPair, SentenceGroup> getValue(ResultSet results) throws SQLException, IOException {
      return new HashMap<KBPair, SentenceGroup>();  // TODO(gabor) empty for DEFT
    }
  }


  /**
   * A callback with utility functions for reading from a (key, provenance) store
   */
  public static abstract class KeyProvenanceCallback extends KeyValueCallback<KBPRelationProvenance> {

    @Override
    protected synchronized void setValue(PreparedStatement stmt, KBPRelationProvenance value) throws SQLException, IOException {
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      ObjectOutputStream oos = new ObjectOutputStream(new GZIPOutputStream(out));
      oos.writeObject(value);
      oos.close();
      byte[] data = out.toByteArray();
      stmt.setBinaryStream(2, new ByteArrayInputStream(data), data.length);
    }

    @Override
    protected synchronized KBPRelationProvenance getValue(ResultSet results) throws SQLException, IOException {
      try {
        ObjectInputStream ois = new ObjectInputStream(new GZIPInputStream(new ByteArrayInputStream(results.getBytes("value"))));
        KBPRelationProvenance rtn = (KBPRelationProvenance) ois.readObject();
        ois.close();
        return rtn;
      } catch (ClassNotFoundException e) {
        throw new IOException(e);
      }
    }
  }

  /**
   * An abstract callback implementing a Counter stored in Postgres.
   * This is, effectively, a key/value store but one whose values are constrained to be doubles and thus can
   * implement an {@link CounterCallback#incrementCount(java.sql.Connection, String, Object, double)} operation.
   *
   * @param <KEY> The type of key to store in this counter. This type must be convertable to a String by the callback.
   */
  public static abstract class CounterCallback<KEY> extends KeyValueCallback<Double> {
    /**
     * Convert the key to a string representation, to be saved in a Key/Value table.
     * @param key The key to serialize.
     * @return A string representation of a key, such that a == b iff key2string(a) == key2string(b)
     */
    protected abstract String key2string(KEY key);

    /**
     * Increment the count of the given key by the given value.
     * If the key does not yet exist, it is incremented from 0 by the value.
     *
     * @param psql The postgres connection to use.
     * @param table The table which implements the Counter
     * @param keyAsObject The key, as an object not yet serialized to its string form
     * @param value The value to increment the counter by.
     * @throws SQLException
     */
    protected void incrementCount(Connection psql, String table, KEY keyAsObject, double value) throws SQLException {
      String key = key2string(keyAsObject);
      // Ensure cached statement
      ensureStatements(psql, table);
      // Flush anything that may get overwritten
      stmts.get(Pair.makePair(table, psql)).ensureWritable(key);
      // Get prepared statements
      PreparedStatement increment = stmts.get(Pair.makePair(table, psql)).increment;
      // Run insert
      increment.setString(1, key);
      increment.setDouble(2, value);
      stmts.get(Pair.makePair(table, psql)).doIncrement(key);
    }

    /**
     * Add every element of a counter to this Postgres table.
     */
    protected void addAll(Connection psql, String table, Counter<KEY> counts) throws SQLException {
      for (Map.Entry<KEY, Double> entry : counts.entrySet()) {
        incrementCount(psql, table, entry.getKey(), entry.getValue());
      }
    }

    /**
     * Get teh count of a given key, or 0 if the key does not exist.
     */
    protected Double getCount(Connection psql, String tableName, KEY clauses) throws SQLException {
      return get(psql, tableName, key2string(clauses)).getOrElse(0.0);
    }

    /**
     * Set the count of a given key to a given count, overwriting the existing count if one exists.
     */
    protected boolean setCount(Connection psql, String tableName, KEY clauses, double count) throws SQLException {
      return put(psql, tableName, key2string(clauses), count);
    }

    @Override
    protected void setValue(PreparedStatement stmt, Double value) throws SQLException, IOException {
      stmt.setDouble(2, value);
    }

    @Override
    protected Double getValue(ResultSet results) throws SQLException, IOException {
      return results.getDouble("value");
    }
  }

  /**
   * <p>
   *   An implementation of a counter over Conjunctive Normal Form formulas.
   *   Each clause of the formula is represented as a KBTriple, enforcing the types
   *   of both the entity and the slot value, and the values of the entity and slot value.
   * </p>
   *
   * <p>
   *   Note that the values are not abstracted into variables here. This allows for literals
   *   to be included in the CNF formula, but on the other hand punts on the conversion from an
   *   instance of a formula to its abstract form to the caller. To illustrate,
   *   born_in(Obama, Hawaii) will not be converted to born_in(x, y), even though this is often
   *   the form in which you'd like to store it.
   * </p>
   */
  public static abstract class CNFFormulaCounterCallback extends CounterCallback<Set<KBTriple>> {
    /**
     * The regular expression for a clause of a CNF formula. Used by {@link CNFFormulaCounterCallback#string2key(String)}
     */
    public static final Pattern CLAUSE_REGEXP = Pattern.compile("([^\\(∧]+)\\(([^:]+):([^,]+),([^:]+):([^\\)]+)\\)");

    /**
     * Serialize a CNF formula to a relatively compact String representation.
     * This serialization is guaranteed to be deterministic, and with the exception of some corner cases dealing with
     * special symbols (to ensure it can be deserialized unambiguously) it is guaranteed that deserializing with
     * with {@link CNFFormulaCounterCallback#string2key(String)} will yield the exact input passed to this function.
     *
     * @param clauses The clauses to serialize
     * @return A string representation of the clauses, compacted but hopefully still somewhat human readable.
     */
    @Override
    protected String key2string(Set<KBTriple> clauses) {
      // Serialize clauses into Strings
      List<String> clausesAsList = new ArrayList<String>(clauses.size());
      for (KBTriple clause : clauses) {
        String slotType = "NIL";
        if (clause.slotType.isDefined()) { slotType = clause.slotType.get().shortName; }
        StringBuilder clauseAsString = new StringBuilder();
        clauseAsString.append(clause.relationName.replaceAll("\\(", "").replaceAll("\\s+", " "))
            .append("(")
            .append(clause.entityName.replaceAll(":", ""))
            .append(":").append(clause.entityType.shortName)
            .append(",").append(clause.slotValue.replaceAll(":", ""))
            .append(":").append(slotType)
            .append(")");
        clausesAsList.add(clauseAsString.toString());
      }
      // Sort the clauses to maintain set semantics
      Collections.sort(clausesAsList);
      // Joint the clauses into a single String
      return StringUtils.join(clausesAsList, "∧");
    }

    /**
     * Deserialize a string representation of a CNF formula into the formula.
     *
     * @param str The string representation of the formula, as created by {@link CNFFormulaCounterCallback#key2string(Object)}.
     * @return A set of KBTriples, corresponding to the clauses of the CNF formula.
     */
    protected Set<KBTriple> string2key(String str) {
      Matcher matcher = CLAUSE_REGEXP.matcher(str);
      Set<KBTriple> conjunctions = new HashSet<KBTriple>();
      while (matcher.find()) {
        conjunctions.add(KBPNew.entName(matcher.group(2)).entType(NERTag.fromShortName(matcher.group((3))).orCrash())
            .slotValue(matcher.group(4)).slotType(NERTag.fromShortName(matcher.group(5)))
            .rel(matcher.group(1)).KBTriple());
      }
      return conjunctions;
    }
  }

  /**
   * A callback with utility functions for reading from a (key, String) store
   */
  public static abstract class SetCallback extends KeyValueCallback<Boolean> {
    @Override
    protected synchronized void setValue(PreparedStatement stmt, Boolean value) throws SQLException, IOException {
      stmt.setBoolean(2, value);
    }
    @Override
    protected synchronized Boolean getValue(ResultSet results) throws SQLException, IOException {
      return results.getBoolean("value");
    }
    protected boolean contains(Connection psql, String table, String key) throws SQLException { return get(psql, table, key).getOrElse(false); }
    public synchronized boolean add(Connection psql, String table, String key) throws SQLException { return put(psql, table, key, true); }
  }



  /** Constructs the URI for the postgres instance, reading relevant fields from {@link edu.stanford.nlp.kbp.slotfilling.common.Props} */
  public static String uri() {
    String host = Props.PSQL_HOST;
    try {
      for (String tunnel : Props.PSQL_HOST_TUNNELS) {
        if (tunnel.equals(InetAddress.getLocalHost().getHostName().intern())) {
          host = "localhost";
        }
      }
    } catch (UnknownHostException e) {
      throw new RuntimeException(e);
    }
    return "jdbc:postgresql://" + host + ":" + Props.PSQL_PORT + "/" + Props.PSQL_DB;
  }

  /** Manages a postgres connection, calling the passed Callback as a callback with the active connection */
  public static void withConnection(Callback callback) {
    try {
      Connection psql = DriverManager.getConnection(PostgresUtils.uri(), Props.PSQL_USERNAME, Props.PSQL_PASSWORD);
      psql.setAutoCommit(false);
      try {
        callback.apply(psql);
      } catch (SQLException e) {
        throw new RuntimeException(e);
      } finally {
        if (!psql.getAutoCommit()) { psql.commit(); }
        psql.close();
      }
    } catch (SQLException e) {
      throw new RuntimeException(e);
    }
  }

  public static void withConnection(String connectionName, Callback callback) {
    try {
      // Get (or create) connection
      Connection psql;
      if (!namedConnections.containsKey(connectionName)) {
        psql = DriverManager.getConnection(PostgresUtils.uri(), Props.PSQL_USERNAME, Props.PSQL_PASSWORD);
        psql.setAutoCommit(true);
        namedConnections.put(connectionName, psql);
      } else {
        psql = namedConnections.get(connectionName);
      }
      // Run callback
      //noinspection SynchronizationOnLocalVariableOrMethodParameter
      synchronized (psql) {
        callback.apply(psql);
      }
    } catch (SQLException e) {
      if (e instanceof BatchUpdateException) {
        BatchUpdateException be = (BatchUpdateException) e;
        Exception cause;
        if ( (cause = be.getNextException()) != null) {
          logger.log(cause);
        }
      }
      throw new RuntimeException(e);
    }
  }


  public static void exec(Connection psql, String sql) throws SQLException {
    Statement statement = psql.createStatement();
    statement.execute(sql);
    statement.close();
  }

  /** Returns whether a specified table exists in the database */
  public static boolean haveTable(final String tableName) {
    final Pointer<Boolean> haveTable = new Pointer<Boolean>(false);
    withConnection(tableName, new Callback() {
      @Override
      public void apply(Connection psql) throws SQLException {
        // Check if table exists
        DatabaseMetaData metadata = psql.getMetaData();
        ResultSet tablesResultSet = metadata.getTables(null, null, tableName, new String[] {"TABLE"});
        if (tablesResultSet.next()) { haveTable.set(true); }
      }
    });
    return haveTable.dereference().orCrash();
  }

  /**
   * Runs a specified callback ensuring that the associated table exists
   * @param tableName The table to ensure
   * @param callback The callback to run. The connection is automatically closed once the callback finishes
   * @param createStatement The statement to use to create the table, if it doesn't already exist
   */
  public static void withTable(String tableName, final Callback callback, final Maybe<String> createStatement) {
    final boolean doCreate;

    // Ensure that the table exists
    if (!haveTable(tableName)) {
      if (!createStatement.isDefined()) {
        throw new IllegalArgumentException("Table not found and no create statement specified: " + tableName);
      } else {
        doCreate = true;
      }
    } else {
      doCreate = false;
    }

    // Run callback
    withConnection(tableName + "_" + createStatement.getOrElse(""), new Callback() {  // cache on table and create statement
      @Override
      public void apply(Connection psql) throws SQLException {
        // Create table
        if (doCreate) {
          assert createStatement.isDefined();
          exec(psql, createStatement.get());
          if (!psql.getAutoCommit()) {
            psql.commit();
          }
        }
        // Run callback
        callback.apply(psql);
      }
    });
  }

  /** @see edu.stanford.nlp.kbp.slotfilling.common.PostgresUtils#withTable(String, edu.stanford.nlp.kbp.slotfilling.common.PostgresUtils.Callback, Maybe) */
  public static void withTable(String tableName, final Callback callback, String createStatement) { withTable(tableName, callback, Maybe.Just(createStatement)); }

  public static void withKeyValueTable(String tableName, final Callback callback, String keyType, String valueType) {
    withTable(tableName, callback, Maybe.Just(
        "CREATE TABLE IF NOT EXISTS \"" + tableName + "\"( key " + keyType + " PRIMARY KEY, value " + valueType +" );" +
        "DROP FUNCTION IF EXISTS \"_jdbc_set_" + tableName.toLowerCase() + "\"(" + keyType + ", " + valueType + ");" +
        "CREATE FUNCTION \"_jdbc_set_" + tableName.toLowerCase() + "\"(k " + keyType + ", v " + valueType + ") RETURNS VOID AS\n" +
        "$$\n" +
        "BEGIN\n" +
        "    LOOP\n" +
        "        -- first try to update the key\n" +
        "        UPDATE \"" + tableName + "\" SET value = v WHERE key = k;\n" +
        "        IF found THEN\n" +
        "            RETURN;\n" +
        "        END IF;\n" +
        "        -- not there, so try to insert the key\n" +
        "        -- if someone else inserts the same key concurrently,\n" +
        "        -- we could get a unique-key failure\n" +
        "        BEGIN\n" +
        "            INSERT INTO \"" + tableName + "\"(key, value) VALUES (k, v);\n" +
        "            RETURN;\n" +
        "        EXCEPTION WHEN unique_violation THEN\n" +
        "            -- Do nothing, and loop to try the UPDATE again.\n" +
        "        END;\n" +
        "    END LOOP;\n" +
        "END;\n" +
        "$$\n" +
        "LANGUAGE plpgsql;" +
        "DROP FUNCTION IF EXISTS \"_jdbc_increment_" + tableName.toLowerCase() + "\"(" + keyType + ", " + valueType + ");" +
            "CREATE FUNCTION \"_jdbc_increment_" + tableName.toLowerCase() + "\"(k " + keyType + ", v " + valueType + ") RETURNS VOID AS\n" +
            "$$\n" +
            "BEGIN\n" +
            "    LOOP\n" +
            "        -- first try to update the key\n" +
            "        UPDATE \"" + tableName + "\" SET value = value+v WHERE key = k;\n" +
            "        IF found THEN\n" +
            "            RETURN;\n" +
            "        END IF;\n" +
            "        -- not there, so try to insert the key\n" +
            "        -- if someone else inserts the same key concurrently,\n" +
            "        -- we could get a unique-key failure\n" +
            "        BEGIN\n" +
            "            INSERT INTO \"" + tableName + "\"(key, value) VALUES (k, v);\n" +
            "            RETURN;\n" +
            "        EXCEPTION WHEN unique_violation THEN\n" +
            "            -- Do nothing, and loop to try the UPDATE again.\n" +
            "        END;\n" +
            "    END LOOP;\n" +
            "END;\n" +
            "$$\n" +
            "LANGUAGE plpgsql;"
    ));
  }

  /** Do a series of operations with a (key[string], annotation) table */
  public static void withKeyAnnotationTable(String tableName, final KeyAnnotationCallback callback) {
    withKeyValueTable(tableName, callback, "TEXT", "bytea");
  }
  /** Do a series of operations with a (key[string], string) table */
  public static void withKeyStringTable(String tableName, final KeyStringCallback callback) {
    withKeyValueTable(tableName, callback, "TEXT", "TEXT");
  }
  /** Do a series of operations with a (key[string], datum) table */
  public static void withKeyDatumTable(String tableName, final KeyDatumCallback callback) {
    withKeyValueTable(tableName, callback, "TEXT", "bytea");
  }
  /** Do a series of operations with a (key[string], provenance) table */
  public static void withKeyProvenanceTable(String tableName, final KeyProvenanceCallback callback) {
    withKeyValueTable(tableName, callback, "TEXT", "bytea");
  }
  /** Do a series of operations with Counter implemented on Postgres */
  public static void withCounter(String tableName, final CounterCallback callback) {
    withKeyValueTable(tableName, callback, "TEXT", "DOUBLE PRECISION");
  }
  /** Do a series of operations with a set -- i.e., a (key, boolean) table */
  public static void withSet(String tableName, final SetCallback callback) {
    withKeyValueTable(tableName, callback, "TEXT", "BOOLEAN");
  }

  /** Drops a table from the database. USE WITH CARE (this is mostly just for tests)! */
  public static boolean dropTable(final String tableName) {
    if (!haveTable(tableName)) { return false; }
    withConnection(tableName, new Callback() {
      @Override
      public void apply(Connection psql) throws SQLException {
        // Create table
        exec(psql, "DROP TABLE " + tableName + ";");
        if (!psql.getAutoCommit()) {
          psql.commit();
        }
      }
    });
    return true;
  }
}
