package edu.stanford.nlp.kbp.slotfilling.evaluate;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.kbp.slotfilling.common.*;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Function;
import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.Interner;
import edu.stanford.nlp.util.Pair;

import java.io.*;
import java.util.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

/**
 * A class for encoding some real world inferences and consistency checks.
 * Among these (update this list as more are implemented):
 *   - Geographic Consistency
 *
 * @author Gabor Angeli
 */
public class WorldKnowledgePostProcessor extends HeuristicSlotfillPostProcessor.Default {

  /** The directory the relevant data files are stored in */
  public final File directory;

  /** A mapping from a city to the list of regions that city may be in */
  private final Map<String, Set<String>> city2region = new HashMap<String, Set<String>>();
  /** A mapping from a city to its population (that is, the population of the largest city of that name */
  private final HashMap<String, Integer> city2population = new HashMap<String, Integer>();
  /** A mapping from a city to the *most likely* region that city is in. This is determined by the size of the city */
  private final Map<String, String> city2PrimaryRegion = new HashMap<String, String>();
  /** A mapping from a city abbreviation to the city it refers to */
  private final Map<String, String> abbrv2city = new HashMap<String, String>();
  /** A mapping from region codes to their corresponding region */
  private Map<Pair<String,String>, String> code2region = new HashMap<Pair<String,String>, String>();
  /** A mapping from a region to a set of countries that region could be in. Really, this should be a singleton set. */
  private final Map<String, Set<String>> region2country = new HashMap<String, Set<String>>();
  /**  A mapping from country codes (upper case) to the country */
  private final Map<String, String> code2country = new HashMap<String, String>();
  /**  A mapping from country codes (upper case) to nationalities */
  private final Map<String, List<String>> code2nationalities = new HashMap<String, List<String>>();
  /**  A mapping from nationality (lower case) to country codes (upper case) */
  private final Map<String, String> nationality2countrycode = new HashMap<String, String>();
  /**  A mapping from country name to country code */
  private final Map<String, String> country2code = new HashMap<String, String>();
  /**  A set of valid countries */
  private final Set<String> countries = new HashSet<String>();
  /**  A mapping from alternate country names (case sensitive!) to their canonical form */
  private final Map<String, String> alternateName2country = new HashMap<String, String>();
//  /** The FAUST Gazetteer */
//  public final Gazetteer faustGazetteer;

  private static final Map<RelationType,RelationType> cityRewrite = new HashMap<RelationType,RelationType>();
  private static final Map<RelationType,RelationType> regionRewrite = new HashMap<RelationType,RelationType>();
  private static final Map<RelationType,RelationType> countryRewrite = new HashMap<RelationType,RelationType>();
  static {
    { // City stuff
      cityRewrite.put(RelationType.ORG_CITY_OF_HEADQUARTERS, RelationType.ORG_CITY_OF_HEADQUARTERS);
      cityRewrite.put(RelationType.ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS, RelationType.ORG_CITY_OF_HEADQUARTERS);
      cityRewrite.put(RelationType.ORG_COUNTRY_OF_HEADQUARTERS, RelationType.ORG_CITY_OF_HEADQUARTERS);

      cityRewrite.put(RelationType.PER_CITIES_OF_RESIDENCE, RelationType.PER_CITIES_OF_RESIDENCE);
      cityRewrite.put(RelationType.PER_STATE_OR_PROVINCES_OF_RESIDENCE, RelationType.PER_CITIES_OF_RESIDENCE);
      cityRewrite.put(RelationType.PER_COUNTRIES_OF_RESIDENCE, RelationType.PER_CITIES_OF_RESIDENCE);

      cityRewrite.put(RelationType.PER_CITY_OF_BIRTH, RelationType.PER_CITY_OF_BIRTH);
      cityRewrite.put(RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH, RelationType.PER_CITY_OF_BIRTH);
      cityRewrite.put(RelationType.PER_COUNTRY_OF_BIRTH, RelationType.PER_CITY_OF_BIRTH);

      cityRewrite.put(RelationType.PER_CITY_OF_DEATH, RelationType.PER_CITY_OF_DEATH);
      cityRewrite.put(RelationType.PER_STATE_OR_PROVINCES_OF_DEATH, RelationType.PER_CITY_OF_DEATH);
      cityRewrite.put(RelationType.PER_COUNTRY_OF_DEATH, RelationType.PER_CITY_OF_DEATH);
    }
    { // STATE stuff
      regionRewrite.put(RelationType.ORG_CITY_OF_HEADQUARTERS, RelationType.ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS);
      regionRewrite.put(RelationType.ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS, RelationType.ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS);
      regionRewrite.put(RelationType.ORG_COUNTRY_OF_HEADQUARTERS, RelationType.ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS);

      regionRewrite.put(RelationType.PER_CITIES_OF_RESIDENCE, RelationType.PER_STATE_OR_PROVINCES_OF_RESIDENCE);
      regionRewrite.put(RelationType.PER_STATE_OR_PROVINCES_OF_RESIDENCE, RelationType.PER_STATE_OR_PROVINCES_OF_RESIDENCE);
      regionRewrite.put(RelationType.PER_COUNTRIES_OF_RESIDENCE, RelationType.PER_STATE_OR_PROVINCES_OF_RESIDENCE);

      regionRewrite.put(RelationType.PER_CITY_OF_BIRTH, RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH);
      regionRewrite.put(RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH, RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH);
      regionRewrite.put(RelationType.PER_COUNTRY_OF_BIRTH, RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH);

      regionRewrite.put(RelationType.PER_CITY_OF_DEATH, RelationType.PER_STATE_OR_PROVINCES_OF_DEATH);
      regionRewrite.put(RelationType.PER_STATE_OR_PROVINCES_OF_DEATH, RelationType.PER_STATE_OR_PROVINCES_OF_DEATH);
      regionRewrite.put(RelationType.PER_COUNTRY_OF_DEATH, RelationType.PER_STATE_OR_PROVINCES_OF_DEATH);
    }
    { // City stuff
      countryRewrite.put(RelationType.ORG_CITY_OF_HEADQUARTERS, RelationType.ORG_COUNTRY_OF_HEADQUARTERS);
      countryRewrite.put(RelationType.ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS, RelationType.ORG_COUNTRY_OF_HEADQUARTERS);
      countryRewrite.put(RelationType.ORG_COUNTRY_OF_HEADQUARTERS, RelationType.ORG_COUNTRY_OF_HEADQUARTERS);

      countryRewrite.put(RelationType.PER_CITIES_OF_RESIDENCE, RelationType.PER_COUNTRIES_OF_RESIDENCE);
      countryRewrite.put(RelationType.PER_STATE_OR_PROVINCES_OF_RESIDENCE, RelationType.PER_COUNTRIES_OF_RESIDENCE);
      countryRewrite.put(RelationType.PER_COUNTRIES_OF_RESIDENCE, RelationType.PER_COUNTRIES_OF_RESIDENCE);

      countryRewrite.put(RelationType.PER_CITY_OF_BIRTH, RelationType.PER_COUNTRY_OF_BIRTH);
      countryRewrite.put(RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH, RelationType.PER_COUNTRY_OF_BIRTH);
      countryRewrite.put(RelationType.PER_COUNTRY_OF_BIRTH, RelationType.PER_COUNTRY_OF_BIRTH);

      countryRewrite.put(RelationType.PER_CITY_OF_DEATH, RelationType.PER_COUNTRY_OF_DEATH);
      countryRewrite.put(RelationType.PER_STATE_OR_PROVINCES_OF_DEATH, RelationType.PER_COUNTRY_OF_DEATH);
      countryRewrite.put(RelationType.PER_COUNTRY_OF_DEATH, RelationType.PER_COUNTRY_OF_DEATH);
    }
  }

  // Configuration
  private final boolean doSingleSlotConsistency;
  private final boolean doPairwiseConsistency;
  private final boolean doHoldOneOutConsistency;
  private final boolean doSuggestSlots;

  /** Create a new processor using the data files in the given directory */
  public WorldKnowledgePostProcessor(File dataDir) {
//    this.faustGazetteer = FaustGazetteer.getDefaultGazetteer();
    if (!dataDir.isDirectory() || !dataDir.canRead()) {
      throw new IllegalArgumentException("Could not read " + dataDir + " (does it exist and do you have permissions?)");
    }
    this.directory = dataDir;
    Interner<String> stringInterner = new Interner<String>();
    String line;
    try {
      // Countries
      BufferedReader countryReader = IOUtils.readerFromString(dataDir.getPath() + File.separator + "kbp_code2country.tab");
      while ((line = countryReader.readLine()) != null) {
        String[] fields = line.split("\t");
        String country = stringInterner.intern(fields[1]);
        code2country.put(stringInterner.intern(fields[0].toUpperCase()), country);
        country2code.put(country, stringInterner.intern(fields[0].toUpperCase()));
        countries.add(country);
      }
      // Regions
      BufferedReader regionReader = IOUtils.readerFromString(dataDir.getPath() + File.separator + "kbp_code2region.tab");
      while ((line = regionReader.readLine()) != null) {
        String[] fields = line.split("\t");
        code2region.put(Pair.makePair(stringInterner.intern(fields[0].toUpperCase()),
                                      stringInterner.intern(fields[1].toUpperCase())),
                        stringInterner.intern(fields[2]));
      }
      // Cities
      BufferedReader cityReader = IOUtils.readerFromString(dataDir.getPath() + File.separator + "kbp_cities.tab");
      int citiesWithoutRegions = 0;
      while ((line = cityReader.readLine()) != null) {
        // Parse line
        String[] fields = line.split("\t");
        String city = stringInterner.intern(fields[0]);
        String regionCode = stringInterner.intern(fields[1]);
        String countryCode = stringInterner.intern(fields[2]);
        String region = code2region.get(Pair.makePair(countryCode.toUpperCase(), regionCode.toUpperCase()));
        int population = Integer.parseInt(fields[3]);
        // Fill consistency fields
        Set<String> regions = city2region.get(city);
        if (regions == null) { regions = new HashSet<String>(); city2region.put(city, regions); }
        if (region != null) {
          regions.add(region);
          Set<String> countries = region2country.get(region);
          if (countries == null) { countries = new HashSet<String>(); region2country.put(region, countries); }
          String country = code2country.get(countryCode.toUpperCase());
          if (country != null) {
            countries.add(country);
          } else {
            warn("Could not find country for region: " + region + " -> " + countryCode);
          }
        } else {
          citiesWithoutRegions += 1;
        }
        // Fill inference fields
        if (!city2PrimaryRegion.containsKey(city) || city2population.get(city) < population) {
            city2PrimaryRegion.put(city, region);
            city2population.put(city, population);
        }
      }
      if (citiesWithoutRegions > 0) {
        log("could not find region for " + citiesWithoutRegions + " cities");
      }
      // Alternate Country Names
      BufferedReader alternateCountryNamesReader = IOUtils.readerFromString(dataDir.getPath() + File.separator + "kbp_alternate_country_names.tab");
      while ((line = alternateCountryNamesReader.readLine()) != null) {
        String[] fields = line.split("\t");
        String canonicalName = stringInterner.intern(fields[0]);
        if (countries.contains(canonicalName)) {
          for (int i=1; i<fields.length; ++i) {
            alternateName2country.put(stringInterner.intern(fields[i].length() > 4 ? fields[i].toLowerCase().trim() : fields[i].trim()), canonicalName);
          }
        } else {
          warn("Could not find country: '" + canonicalName + "'");
        }
      }
      // City abbreviations
      BufferedReader alternateCityNamesReader = IOUtils.readerFromString(dataDir.getPath() + File.separator + "kbp_abbreviation2city.tab");
      while ((line = alternateCityNamesReader.readLine()) != null) {
        String[] fields = line.split("\t");
        String canonicalName = stringInterner.intern(fields[1].toLowerCase().trim());
        if (city2region.containsKey(canonicalName)) {
          abbrv2city.put(stringInterner.intern(fields[0].toUpperCase()), canonicalName);
        } else {
          warn("Could not find city: '" + canonicalName + "'");
        }
      }
      // Country to nationality
      BufferedReader countryCode2NationalityReader = IOUtils.readerFromString(dataDir.getPath() + File.separator + "kbp_countrycode2nationality.tab");
      while ((line = countryCode2NationalityReader.readLine()) != null) {
        String[] fields = line.split("\t");
        String code = stringInterner.intern(fields[0].toUpperCase());
        if (!code2nationalities.containsKey(code)) {
          code2nationalities.put(code, new ArrayList<String>());
        }
        for (String nationality : fields[1].split("\\|")) {
          code2nationalities.get(code).add(nationality.trim());
          if (!nationality2countrycode.containsKey(nationality.toLowerCase())) { nationality2countrycode.put(nationality.toLowerCase().trim(), code); }
        }
      }

      // Set configuration
      doSingleSlotConsistency = true;
      doPairwiseConsistency = true;
      doHoldOneOutConsistency = true;
      doSuggestSlots = true;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private WorldKnowledgePostProcessor(
      File directory,
      Map<String, Set<String>> city2region,
      HashMap<String, Integer> city2population,
      Map<String, String> city2PrimaryRegion,
      Map<String, String> abbrv2city,
      Map<Pair<String, String>, String> code2region,
      Map<String, Set<String>> region2country,
      Map<String, String> code2country,
      Map<String, List<String>> code2nationalities,
      Map<String, String> nationality2countrycode,
      Map<String, String> country2code,
      Set<String> countries,
      Map<String, String> alternateName2country,
      boolean doSingleSlotConsistency, boolean doPairwiseConsistency,
      boolean doHoldOneOutConsistency, boolean doSuggestSlots) {
    this.directory = directory;
    this.city2region.putAll(city2region);
    this.city2population.putAll(city2population);
    this.city2PrimaryRegion.putAll(city2PrimaryRegion);
    this.abbrv2city.putAll(abbrv2city);
    this.code2region.putAll(code2region);
    this.region2country.putAll(region2country);
    this.code2country.putAll(code2country);
    this.code2nationalities.putAll(code2nationalities);
    this.nationality2countrycode.putAll(nationality2countrycode);
    this.country2code.putAll(country2code);
    this.countries.addAll(countries);
    this.alternateName2country.putAll(alternateName2country);
    this.doSingleSlotConsistency = doSingleSlotConsistency;
    this.doPairwiseConsistency = doPairwiseConsistency;
    this.doHoldOneOutConsistency = doHoldOneOutConsistency;
    this.doSuggestSlots = doSuggestSlots;
  }


  private String canonicalizeCity(String city) {
    String normalizedName = city.trim().toLowerCase();
    if (city2region.containsKey(normalizedName)) { return normalizedName; }
    else if (abbrv2city.containsKey(city.replaceAll("\\.", ""))) { return abbrv2city.get(city.replaceAll("\\.", "")); }
    else {
      normalizedName = normalizedName.replaceAll("st", "saint").replaceAll("st.", "saint")
                                     .replaceAll("mt", "mount").replaceAll("mt.", "mount");
      if (city2region.containsKey(normalizedName)) { return normalizedName; }
      normalizedName = normalizedName.replaceAll("d\\.?c\\.?", "").trim();
      if (city2region.containsKey(normalizedName)) { return normalizedName; }
      return normalizedName;
    }
  }
  private String canonicalizeRegion(String region, Maybe<String> countryCode) {
    String normalizedName = region.trim().toLowerCase();
    if (region2country.containsKey(normalizedName)) { return normalizedName; }
    else if (this.code2region.containsKey(Pair.makePair(countryCode.getOrElse("UNK").toUpperCase(), region))) {
      return code2region.get(Pair.makePair(countryCode.getOrElse("UNK").toUpperCase(), region));
    } else if (this.code2region.containsKey(Pair.makePair(countryCode.getOrElse("UNK").toUpperCase(), region.replaceAll("\\.", "")))) {
      return code2region.get(Pair.makePair(countryCode.getOrElse("UNK").toUpperCase(), region.replaceAll("\\.", "")));
    }
    else { return normalizedName; }
  }
  private String canonicalizeCountry(String country) {
    String normalizedName = country.trim().toLowerCase();
    if (countries.contains(normalizedName)) { return normalizedName; }
    else if (country.length() > 4 && alternateName2country.containsKey(normalizedName)) { return alternateName2country.get(normalizedName); }
    else if (country.length() <= 4 && alternateName2country.containsKey(country.trim())) { return alternateName2country.get(country.trim()); }
    else if (this.code2country.containsKey(country.trim())) { return code2country.get(country.trim()) ; }
    else return normalizedName;
  }
  private String canonicalizeNationality(String nationality) {
    return nationality.trim().toLowerCase();
  }

  public boolean isValidCountry(String country) {
    return countries.contains(canonicalizeCountry(country));
  }

  @SuppressWarnings("UnusedDeclaration")
  public boolean isValidRegion(String region, String country) {
    return region2country.containsKey(canonicalizeRegion(region, Maybe.Just(country)));
  }

  public boolean isValidRegion(String region) {
    return region2country.containsKey(canonicalizeRegion(region, Maybe.<String>Nothing())) ||
           region2country.containsKey(canonicalizeRegion(region, Maybe.Just("US"))) ||
           region2country.containsKey(canonicalizeRegion(region, Maybe.Just("CA")));
  }

  public boolean isValidCity(String city) {
    return city2region.containsKey(canonicalizeCity(city));
  }

  public boolean consistentCityRegion(String city, String region) {
    city = canonicalizeCity(city);
    region = canonicalizeRegion(region, Maybe.<String>Nothing());
    return (city2region.containsKey(city) && city2region.get(city).contains(region)) ||
        (region.equalsIgnoreCase("washington") && city2region.get(city).contains("district of columbia"));
  }

  public boolean consistentCityCountry(String city, String country) {
    city = canonicalizeCity(city);
    country = canonicalizeCountry(country);
    if (city2region.containsKey(city)) {
      for (String region : city2region.get(city)) {
        if (region2country.get(region).contains(country)) { return true; }
      }
    }
    return false;
  }

  public boolean consistentRegionCountry(String region, String country) {
    country = canonicalizeCountry(country);
    region = canonicalizeRegion(region, Maybe.fromNull(country2code.get(country)));
    return region2country.containsKey(region) && region2country.get(region).contains(country);
  }

  public boolean consistentGeography(Maybe<String> city, Maybe<String> region, Maybe<String> country) {
    for(String c : city) { for (String r : region) { if (!consistentCityRegion(c, r)) return false; }}
    for(String c : city) { for (String y : country) { if (!consistentCityCountry(c, y)) return false; }}
    for(String r : region) { for (String y : country) { if (!consistentRegionCountry(r, y)) return false; }}
    return true;
  }

  public Maybe<Integer> cityPopulation(String city) {
    return Maybe.fromNull(city2population.get(canonicalizeCity(city)));
  }

  public Maybe<String> regionForCity(String city) {
    return Maybe.fromNull(city2PrimaryRegion.get(canonicalizeCity(city)));
  }

  public Maybe<String> countryForRegion(String region) {
    region = canonicalizeRegion(region, Maybe.<String>Nothing());
    Set<String> possibleCountries = region2country.get(region);
    if (possibleCountries != null && !possibleCountries.isEmpty()) {
      return Maybe.Just(possibleCountries.iterator().next());
    }
    return Maybe.Nothing();
  }

  public Maybe<String> countryForCity(String city) {
    return regionForCity(city).flatMap(new Function<String, Maybe<String>>() {
      @Override
      public Maybe<String> apply(String in) { return countryForRegion(in); }
    });
  }

  public boolean consistentNationalityCountry(String nationality, String country) {
    country = canonicalizeCountry(country);
    nationality = canonicalizeNationality(nationality);
    String code = country2code.get(country);
    return code == null || (code2nationalities.containsKey(code) && code2nationalities.get(code).contains(nationality));
  }

  public Maybe<String> nationalityForCountry(String country) {
    country = canonicalizeCountry(country);
    String code = country2code.get(country);
    if (code == null || !code2nationalities.containsKey(code)) {
      return Maybe.Nothing();
    } else {
      return Maybe.fromNull(capitalize(code2nationalities.get(code).get(0)));
    }
  }

  public Maybe<String> countryForNationality(String nationality) {
    if (nationality2countrycode.containsKey(nationality.toLowerCase())) {
      return Maybe.fromNull(capitalize(code2country.get(nationality2countrycode.get(nationality.toLowerCase()))));
    } else {
      return Maybe.Nothing();
    }
  }

  /**
   * Configure what this consistency checker should do in terms of filtering and suggesting
   * @param singleSlot Filter invalid slots
   * @param pairwise Filter pairwise consistency (e.g., that a city is in the state it should be)
   * @param suggest Suggest new slot fills (e.g., the region and country for a city)
   * @return This same processor, but with the filters applied. THIS IS NOT A FUNCTIONAL CALL.
   */
  public WorldKnowledgePostProcessor configure(final boolean singleSlot, final boolean pairwise, final boolean holdOneOut, final boolean suggest) {
    return new WorldKnowledgePostProcessor(
        directory, city2region, city2population, city2PrimaryRegion, abbrv2city, code2region, region2country, code2country, code2nationalities, nationality2countrycode,
        country2code, countries, alternateName2country,
        singleSlot, pairwise, holdOneOut, suggest);
  }


  /** check if a slot fill is valid according to our world knowledge database */
  // TODO(gabor) This should be replaced by hard constraints in the new index
  private boolean isValidSlot(@SuppressWarnings("UnusedParameters") KBPEntity pivot, KBPSlotFill candidate) {
    if (!doSingleSlotConsistency) return true;
    for (NERTag type : Utils.inferFillType(candidate.key.kbpRelation())) {
      if (type == NERTag.CITY) { return isValidCity(candidate.key.slotValue); }
      if (type == NERTag.STATE_OR_PROVINCE) { return isValidRegion(candidate.key.slotValue); }
      if (type == NERTag.COUNTRY) { return isValidCountry(candidate.key.slotValue); }
    }
    return true;
  }

  @Override
  /** check if a slot fill is valid according to our world knowledge database */
  public Maybe<KBPSlotFill> isValidSlotAndRewrite(KBPEntity pivot, KBPSlotFill candidate) {
    if (isValidSlot(pivot, candidate)) {
      return Maybe.Just(candidate);
    }
    if( RelationType.fromString(candidate.key.relationName).isDefined() &&
        RelationType.fromString(candidate.key.relationName).get().isCityRegionCountryRelation() ) {
      // Check if the slot is _actually_ a city/region/country and rewrite appropriately
      if( isValidCity(candidate.key.slotValue) )
        return Maybe.Just(KBPNew.from(candidate).rel(cityRewrite.get(candidate.key.kbpRelation())).KBPSlotFill());
      if( isValidRegion(candidate.key.slotValue) )
        return Maybe.Just(KBPNew.from(candidate).rel(regionRewrite.get(candidate.key.kbpRelation())).KBPSlotFill());
      if( isValidCountry(candidate.key.slotValue) )
        return Maybe.Just(KBPNew.from(candidate).rel(countryRewrite.get(candidate.key.kbpRelation())).KBPSlotFill());
    }

    debug("WorldKnowledge", "Invalid fill: " + candidate);
    return HeuristicSlotfillPostProcessors.singletonFailure(candidate, this.getClass());
  }

  private boolean shouldBeGeoConsistent(String a, String b) {
    for (RelationType relationA : RelationType.fromString(a)) {
      //noinspection LoopStatementThatDoesntLoop
      for (RelationType relationB : RelationType.fromString(b)) {
        switch (relationA) {
          case PER_CITY_OF_BIRTH: return relationB == RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH || relationB == RelationType.PER_COUNTRY_OF_BIRTH;
          case PER_STATE_OR_PROVINCES_OF_BIRTH: return relationB == RelationType.PER_CITY_OF_BIRTH || relationB == RelationType.PER_COUNTRY_OF_BIRTH;
          case PER_COUNTRY_OF_BIRTH: return relationB == RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH || relationB == RelationType.PER_CITY_OF_BIRTH;
          case PER_CITY_OF_DEATH: return relationB == RelationType.PER_STATE_OR_PROVINCES_OF_DEATH || relationB == RelationType.PER_COUNTRY_OF_DEATH;
          case PER_STATE_OR_PROVINCES_OF_DEATH: return relationB == RelationType.PER_CITY_OF_DEATH || relationB == RelationType.PER_COUNTRY_OF_DEATH;
          case PER_COUNTRY_OF_DEATH: return relationB == RelationType.PER_CITY_OF_DEATH || relationB == RelationType.PER_STATE_OR_PROVINCES_OF_DEATH;
          case PER_CITIES_OF_RESIDENCE: return relationB == RelationType.PER_STATE_OR_PROVINCES_OF_RESIDENCE || relationB == RelationType.PER_COUNTRIES_OF_RESIDENCE;
          case PER_STATE_OR_PROVINCES_OF_RESIDENCE: return relationB == RelationType.PER_CITIES_OF_RESIDENCE || relationB == RelationType.PER_COUNTRIES_OF_RESIDENCE;
          case PER_COUNTRIES_OF_RESIDENCE: return relationB == RelationType.PER_STATE_OR_PROVINCES_OF_RESIDENCE || relationB == RelationType.PER_CITIES_OF_RESIDENCE;
          case ORG_CITY_OF_HEADQUARTERS: return relationB == RelationType.ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS || relationB == RelationType.ORG_COUNTRY_OF_HEADQUARTERS;
          case ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS: return relationB == RelationType.ORG_CITY_OF_HEADQUARTERS || relationB == RelationType.ORG_COUNTRY_OF_HEADQUARTERS;
          case ORG_COUNTRY_OF_HEADQUARTERS: return relationB == RelationType.ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS || relationB == RelationType.ORG_CITY_OF_HEADQUARTERS;
          default: return false;
        }
      }
    }
    return true;
  }

  /** Check to see if two slot fills are contradictory. E.g., "San Francisco, Ohio" */
  @Override
  public boolean pairwiseKeepLowerScoringFill(KBPEntity pivot, KBPSlotFill higherScoring, KBPSlotFill lowerScoring) {
    if (!doPairwiseConsistency) return true;
    for (NERTag higher : Utils.inferFillType(higherScoring.key.kbpRelation())) {
      for (NERTag lower : Utils.inferFillType(lowerScoring.key.kbpRelation())) {
        if (shouldBeGeoConsistent(higherScoring.key.relationName, lowerScoring.key.relationName)) {
          // Geography containment
          Maybe<String> city = Maybe.Nothing();
          Maybe<String> region = Maybe.Nothing();
          Maybe<String> country = Maybe.Nothing();
          if (higher == NERTag.CITY) { city = Maybe.Just(higherScoring.key.slotValue); }
          if (higher == NERTag.STATE_OR_PROVINCE) { region = Maybe.Just(higherScoring.key.slotValue); }
          if (higher == NERTag.COUNTRY) { country = Maybe.Just(higherScoring.key.slotValue); }
          if (lower == NERTag.CITY) { city = Maybe.Just(lowerScoring.key.slotValue); }
          if (lower == NERTag.STATE_OR_PROVINCE) { region = Maybe.Just(lowerScoring.key.slotValue); }
          if (lower == NERTag.COUNTRY) { country = Maybe.Just(lowerScoring.key.slotValue); }
          if (!consistentGeography(city, region, country)) { return HeuristicSlotfillPostProcessors.nonlocalFailure(lowerScoring, this.getClass()); }
        }

        // Geography equality
        if (higherScoring.key.kbpRelation() == lowerScoring.key.kbpRelation() && higher == lower) {
          if (higher == NERTag.COUNTRY && canonicalizeCountry(higherScoring.key.slotValue).equals(canonicalizeCountry(lowerScoring.key.slotValue))) {
            // case: alternate names for the same country
            return HeuristicSlotfillPostProcessors.nonlocalFailure(lowerScoring, this.getClass());
          } else if (higher == NERTag.STATE_OR_PROVINCE && canonicalizeRegion(higherScoring.key.slotValue, Maybe.<String>Nothing()).equals(canonicalizeRegion(lowerScoring.key.slotValue, Maybe.<String>Nothing()))) {
            // case: alternate names for the same state or province (region)
            return HeuristicSlotfillPostProcessors.nonlocalFailure(lowerScoring, this.getClass());
          } else if (higher == NERTag.CITY && canonicalizeCity(higherScoring.key.slotValue).equals(canonicalizeCity(lowerScoring.key.slotValue))) {
            // case: alternate names for the same city (e.g., accronyms)
            return HeuristicSlotfillPostProcessors.nonlocalFailure(lowerScoring, this.getClass());
          }

        }
      }
    }
    return true;
  }


  private static final double entailmentWeight = HeuristicSlotfillPostProcessors.FilterVeryLowProbabilitySlots.threshold + 1e-5;

  @SuppressWarnings("UnusedParameters")
  private void geoEntailments(Counter<Pair<RelationType, String>> entailments,
                              Maybe<String> city, Maybe<String> region, Maybe<String> country,
                              RelationType cityRel, RelationType regionRel, RelationType countryRel) {
    for (String c : city) {
      for (String r : regionForCity(c)) { entailments.setCount(Pair.makePair(regionRel, r), entailmentWeight); }
      for (String y : countryForCity(c)) { entailments.setCount(Pair.makePair(countryRel, y), entailmentWeight); }
    }
    for (String r : region) {
      for (String y : countryForRegion(r)) { entailments.setCount(Pair.makePair(countryRel, y), entailmentWeight); }
    }
  }

  @Override
  public boolean leaveOneOutKeepHeldOutSlot(KBPEntity pivot, IdentityHashSet<KBPSlotFill> others, KBPSlotFill candidate) {
    if (doHoldOneOutConsistency && candidate.key.kbpRelation() == RelationType.PER_ORIGIN) {
      assert candidate.key.kbpRelation().canonicalName.equals(RelationType.PER_ORIGIN.canonicalName);
      // Origin is consistent with country of birth
      boolean seenCountryOfBirth = false;
      for (KBPSlotFill fill : others) {
        if (fill.key.kbpRelation() == RelationType.PER_COUNTRY_OF_BIRTH) {
          if (this.consistentNationalityCountry(candidate.key.slotValue, fill.key.slotValue)) {
            return true;
          } else {
            seenCountryOfBirth = true;
          }
        }
      }
      // Is one of top 2 origins
      int numHigherRankingOrigins = 0;
      int maxNumOriginsReturned = 2;
      for (KBPSlotFill fill : others) {
        if (fill.key.kbpRelation() == RelationType.PER_ORIGIN) {
          if (fill.score.getOrElse(0.0) > candidate.score.getOrElse(0.0)) { numHigherRankingOrigins += 1; }
          if (numHigherRankingOrigins > (seenCountryOfBirth ? maxNumOriginsReturned - 2 : maxNumOriginsReturned - 1)) {
            return HeuristicSlotfillPostProcessors.nonlocalFailure(candidate, this.getClass());
          }
        }
      }
      return true;
    } else {
      return true;
    }
  }

  /** Fill in some slot fills we're pretty sure are true. For example, the state for a city */
  @Override
  public Counter<Pair<RelationType, String>> entailsDirectly(KBPEntity pivot, KBPSlotFill antecedent) {
    Counter<Pair<RelationType, String>> entailments =  new ClassicCounter<Pair<RelationType, String>>();
    if (!doSuggestSlots) return entailments;
    for (RelationType relation : RelationType.fromString(antecedent.key.relationName)) {
      switch (relation) {
        // (cities)
        case PER_CITY_OF_BIRTH:
          geoEntailments(entailments, Maybe.Just(antecedent.key.slotValue), Maybe.<String>Nothing(), Maybe.<String>Nothing(),
              RelationType.PER_CITY_OF_BIRTH, RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH, RelationType.PER_COUNTRY_OF_BIRTH);
          break;
        case PER_CITY_OF_DEATH:
          geoEntailments(entailments, Maybe.Just(antecedent.key.slotValue), Maybe.<String>Nothing(), Maybe.<String>Nothing(),
              RelationType.PER_CITY_OF_DEATH, RelationType.PER_STATE_OR_PROVINCES_OF_DEATH, RelationType.PER_COUNTRY_OF_DEATH);
          break;
        case PER_CITIES_OF_RESIDENCE:
          geoEntailments(entailments, Maybe.Just(antecedent.key.slotValue), Maybe.<String>Nothing(), Maybe.<String>Nothing(),
              RelationType.PER_CITIES_OF_RESIDENCE, RelationType.PER_STATE_OR_PROVINCES_OF_RESIDENCE, RelationType.PER_COUNTRIES_OF_RESIDENCE);
          break;
        case ORG_CITY_OF_HEADQUARTERS:
          geoEntailments(entailments, Maybe.Just(antecedent.key.slotValue), Maybe.<String>Nothing(), Maybe.<String>Nothing(),
              RelationType.ORG_CITY_OF_HEADQUARTERS, RelationType.ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS, RelationType.ORG_COUNTRY_OF_HEADQUARTERS);
          break;
        // (regions)
        case PER_STATE_OR_PROVINCES_OF_BIRTH:
          geoEntailments(entailments, Maybe.<String>Nothing(), Maybe.Just(antecedent.key.slotValue), Maybe.<String>Nothing(),
              RelationType.PER_CITY_OF_BIRTH, RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH, RelationType.PER_COUNTRY_OF_BIRTH);
          break;
        case PER_STATE_OR_PROVINCES_OF_DEATH:
          geoEntailments(entailments, Maybe.<String>Nothing(), Maybe.Just(antecedent.key.slotValue), Maybe.<String>Nothing(),
              RelationType.PER_CITY_OF_DEATH, RelationType.PER_STATE_OR_PROVINCES_OF_DEATH, RelationType.PER_COUNTRY_OF_DEATH);
          break;
        case PER_STATE_OR_PROVINCES_OF_RESIDENCE:
          geoEntailments(entailments, Maybe.<String>Nothing(), Maybe.Just(antecedent.key.slotValue), Maybe.<String>Nothing(),
              RelationType.PER_CITIES_OF_RESIDENCE, RelationType.PER_STATE_OR_PROVINCES_OF_RESIDENCE, RelationType.PER_COUNTRIES_OF_RESIDENCE);
          break;
        case ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS:
          geoEntailments(entailments, Maybe.<String>Nothing(), Maybe.Just(antecedent.key.slotValue), Maybe.<String>Nothing(),
              RelationType.ORG_CITY_OF_HEADQUARTERS, RelationType.ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS, RelationType.ORG_COUNTRY_OF_HEADQUARTERS);
          break;
        // (countries)
        case PER_COUNTRY_OF_BIRTH:
          geoEntailments(entailments, Maybe.<String>Nothing(), Maybe.<String>Nothing(), Maybe.Just(antecedent.key.slotValue),
              RelationType.PER_CITY_OF_BIRTH, RelationType.PER_STATE_OR_PROVINCES_OF_BIRTH, RelationType.PER_COUNTRY_OF_BIRTH);
          for (String nationality : this.nationalityForCountry(antecedent.key.slotValue)) {
            entailments.setCount(Pair.makePair(RelationType.PER_ORIGIN, nationality), entailmentWeight);
          }
          break;
        case PER_COUNTRY_OF_DEATH:
          geoEntailments(entailments, Maybe.<String>Nothing(), Maybe.<String>Nothing(), Maybe.Just(antecedent.key.slotValue),
              RelationType.PER_CITY_OF_DEATH, RelationType.PER_STATE_OR_PROVINCES_OF_DEATH, RelationType.PER_COUNTRY_OF_DEATH);
          break;
        case PER_COUNTRIES_OF_RESIDENCE:
          geoEntailments(entailments, Maybe.<String>Nothing(), Maybe.<String>Nothing(), Maybe.Just(antecedent.key.slotValue),
              RelationType.PER_CITIES_OF_RESIDENCE, RelationType.PER_STATE_OR_PROVINCES_OF_RESIDENCE, RelationType.PER_COUNTRIES_OF_RESIDENCE);
          break;
        case PER_ORIGIN:
          for (String country : this.countryForNationality(antecedent.key.slotValue)) {
            entailments.setCount(Pair.makePair(RelationType.PER_COUNTRIES_OF_RESIDENCE, country), entailmentWeight);
          }
          break;
        case ORG_COUNTRY_OF_HEADQUARTERS:
          geoEntailments(entailments, Maybe.<String>Nothing(), Maybe.<String>Nothing(), Maybe.Just(antecedent.key.slotValue),
              RelationType.ORG_CITY_OF_HEADQUARTERS, RelationType.ORG_STATE_OR_PROVINCES_OF_HEADQUARTERS, RelationType.ORG_COUNTRY_OF_HEADQUARTERS);
          break;
        default:
          break;
      }
    }
    return entailments;
  }

  private static Maybe<WorldKnowledgePostProcessor> cachedSingleton = Maybe.Nothing();

  /**
   * Return a cached world knowledge post processor.
   * This is not necessarily because the class can only be a singleton, but rather as an
   * efficiency tweak so we don't load the information from disk a bunch of times.
   * @param dir The directory to find the relevant files in.
   * @return A cached singleton processor
   */
  public static WorldKnowledgePostProcessor singleton(File dir) {
    if (cachedSingleton == null || !cachedSingleton.isDefined()) {
      if (!dir.exists()) {
        err(RED, "Could not find world knowledge dir; returning NULL so as to not crash junit tests, but this will eventually come back and bite you");
        return null;
      }
      cachedSingleton = Maybe.Just(new WorldKnowledgePostProcessor(dir));
    }
    return cachedSingleton.get();
  }

  private static String capitalize(String inputOrNull) {
    if (inputOrNull == null) { return null; }
    StringBuilder b = new StringBuilder();
    boolean lastCharIsSpace = true;
    for (char c : inputOrNull.toCharArray()) {
      if (lastCharIsSpace && c != ' ') {
        b.append(Character.toUpperCase(c));
        lastCharIsSpace = false;
      } else if (lastCharIsSpace) {
        // this really shouldn't happen -- multiple spaces in a row
        warn("capitalizing entry with multiple spaces in a row");
      } else if (c == ' ') {
        lastCharIsSpace = true;
        b.append(c);
      } else {
        lastCharIsSpace = false;
        b.append(Character.toLowerCase(c));
      }
    }
    return b.toString();
  }

}
