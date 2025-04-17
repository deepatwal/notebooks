file_lock = threading.Lock()

def split_camel_case_to_lower_words(name):
    """Splits CamelCase or PascalCase and returns lowercase words separated by spaces."""
    if not name:
        return ""
    # Handle simple cases first
    if name.islower() or '_' in name or not re.search('[A-Z]', name):
        # Replace underscores and lowercase
        return name.replace('_', ' ').lower()

    # Insert space before uppercase letters (except at the start)
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
    # Insert space before uppercase letters that follow lowercase or digit
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)
    return s2.lower()  # Convert the whole result to lowercase


def clean_uri_for_llm_key(uri_str):
    """Cleans a predicate URI string into a readable key (lowercase, space-separated)."""
    if not uri_str:
        return "unknown property"

    # Specific overrides first (already lowercase)
    if uri_str == str(RDF.type):
        return "type"
    if uri_str == str(RDFS.label):
        return "label"
    if uri_str == str(FOAF.name):
        return "name"
    # Add other specific overrides if needed (e.g., DCTERMS.subject -> "subject")

    # General cleaning - extract local name
    if '#' in uri_str:
        name = uri_str.split('#')[-1]
    else:
        name = uri_str.split('/')[-1]

    # Split camel case and convert to lowercase words
    return split_camel_case_to_lower_words(name)


def clean_uri_for_llm_value(uri_str):
    """Cleans a resource URI string into a readable value for LLM output."""
    if not uri_str:
        return "Unknown Resource"
    try:
        if '#' in uri_str:
            name = uri_str.split('#')[-1]
        else:
            name = uri_str.split('/')[-1]
        # Basic URL decoding for parentheses and other common chars
        name = name.replace('%28', '(').replace('%29', ')')
        name = name.replace('%2C', ',').replace('%27', "'")
        # Special handling for Wikidata URIs to just show the QID
        if uri_str.startswith("http://www.wikidata.org/entity/"):
            return name  # Just return QID like Q215380
        # Default: replace underscores with spaces
        return name.replace('_', ' ')
    except Exception: # Handle potential errors if splitting fails
        logger.warning(f"Could not cleanly parse URI for value: {uri_str}", exc_info=False)
        return uri_str # Return original URI as fallback


def format_rdf_term_for_llm_value(term_data):
    """
    Formats a term (represented as dict from Step 1 or URI string)
    into a simple string value for LLM output.
    """
    if isinstance(term_data, dict):  # Literal dictionary
        val = term_data.get("value", "")
        # Clean common literal suffixes for LLM readability
        val = re.sub(r'@\w+(-[A-Za-z0-9]+)*$', '', val)  # Remove @lang tags
        val = re.sub(r'\^\^<.*>$', '', val)  # Remove ^^<datatype>
        val = val.strip('"')  # Remove surrounding quotes if any
        return val
    elif isinstance(term_data, str):  # URI string
        return clean_uri_for_llm_value(term_data)
    else:
        return str(term_data)  # Fallback


def format_rdf_term(term):
    """Creates the intermediate structured representation for RDF terms."""
    if isinstance(term, Literal):
        dt = str(term.datatype) if term.datatype else None
        # Assign default datatypes if missing
        if dt is None and term.language:
            dt = str(RDF.langString)
        elif dt is None:
            dt = str(XSD.string)

        try:
            # Attempt to access term.value to ensure parsing works
            _ = term.value
        except ValueError as ve:
            logger.exception(f"[Warning] Failed to parse literal '{term}' with datatype {dt}: {ve}")
            # return plain string with datatype info
            return {"value": str(term), "language": term.language, "datatype": dt}

        return {"value": str(term), "language": term.language, "datatype": dt}
    elif isinstance(term, URIRef):
        return str(term)
    else:  # Handle Blank Nodes etc.
        return str(term)


def extract_structured_description(rdf_n3_string, instance_iri):
    """Parses N3 RDF data and extracts outgoing/incoming relationships."""
    if not rdf_n3_string:
        # Return empty structure if no N3 data provided (e.g., empty DESCRIBE/CONSTRUCT)
        return {"instance_iri": instance_iri, "outgoing": {}, "incoming": {}}
    g = Graph()
    try:
        # Use instance_iri as base URI for resolving relative URIs if any
        g.parse(data=rdf_n3_string, format="n3", publicID=instance_iri)
    except Exception as e:
        # Log parsing errors specifically
        logger.error(f"[Error] Parsing N3 data for {instance_iri}: {type(e).__name__} - {e}")
        return None  # Indicate failure
    instance_ref = URIRef(instance_iri)
    outgoing_data = defaultdict(list)
    incoming_data = defaultdict(list)

    # Outgoing properties
    for pred, obj in g.predicate_objects(subject=instance_ref):
        pred_uri_str = str(pred)
        formatted_obj = format_rdf_term(obj)
        # Avoid adding exact duplicates (important for literals)
        if formatted_obj not in outgoing_data[pred_uri_str]:
            outgoing_data[pred_uri_str].append(formatted_obj)

    # Incoming relationships
    for subj, pred in g.subject_predicates(object=instance_ref):
        # Avoid reflexive triples (where subject is the instance itself)
        if subj == instance_ref:
            continue
        pred_uri_str = str(pred)
        subj_uri_str = str(subj)
        # Avoid adding duplicate incoming subjects for the same predicate
        if subj_uri_str not in incoming_data[pred_uri_str]:
            incoming_data[pred_uri_str].append(subj_uri_str)

    # Final Structure: Convert defaultdicts, sort values for consistency
    final_outgoing = {pred: sorted(values, key=str)
                      for pred, values in outgoing_data.items()}
    # Sort incoming subjects as well
    final_incoming = {pred: sorted(values)
                      for pred, values in incoming_data.items()}
    return {"instance_iri": instance_iri, "outgoing": final_outgoing, "incoming": final_incoming}


def format_for_llm_custom_layout(structured_data):
    """
    Takes the structured dictionary and formats it into the specific
    two-part layout requested by the user (revised key/predicate format).
    """
    if not structured_data or (not structured_data.get("outgoing") and not structured_data.get("incoming")):
        instance_iri = structured_data.get("instance_iri", "Unknown Instance")
        instance_name = clean_uri_for_llm_value(instance_iri)
        # Provide a minimal output even if no data found after parsing
        return f"name: {instance_name}\n(No description properties found)"

    instance_iri = structured_data.get("instance_iri")
    instance_name_cleaned = clean_uri_for_llm_value(instance_iri)

    output_lines_part1 = []
    output_lines_part2 = []

    # --- Part 1: Outgoing Properties (key: value) ---
    outgoing_properties = structured_data.get("outgoing", {})
    primary_name_val = instance_name_cleaned  # Default name

    temp_outgoing_formatted = {}
    for pred_uri in sorted(outgoing_properties.keys()):
        llm_key = clean_uri_for_llm_key(pred_uri)
        values = outgoing_properties[pred_uri]
        cleaned_values_for_key = []
        for term_data in values:
            cleaned_val = format_rdf_term_for_llm_value(term_data)
            # Add value if it's not empty and not already added
            if cleaned_val and cleaned_val not in cleaned_values_for_key:
                cleaned_values_for_key.append(cleaned_val)
        if cleaned_values_for_key:
            # Sort the cleaned values before joining
            value_string = ", ".join(sorted(cleaned_values_for_key))
            temp_outgoing_formatted[llm_key] = value_string
            # Update primary name if this is the 'name' key
            if llm_key == 'name':
                primary_name_val = value_string

    # Generate output lines for part 1, ensuring 'name' is first
    if 'name' in temp_outgoing_formatted:
        output_lines_part1.append(f"name: {temp_outgoing_formatted['name']}")
    elif instance_name_cleaned:  # Add fallback if no name property found
        output_lines_part1.append(f"name: {instance_name_cleaned}")

    # Add other properties sorted by key
    for key in sorted(temp_outgoing_formatted.keys()):
        if key == 'name':
            continue  # Skip name as it's already added
        output_lines_part1.append(f"{key}: {temp_outgoing_formatted[key]}")

    # --- Part 2: Incoming Relationships (Subject : Predicate : Object) ---
    incoming_relationships = structured_data.get("incoming", {})
    instance_name_for_part2 = primary_name_val  # Use name determined in Part 1

    incoming_tuples = []
    for pred_uri, subjects in incoming_relationships.items():
        # Get cleaned predicate name (lowercase, space-separated)
        if '#' in pred_uri:
            pred_local_name = pred_uri.split('#')[-1]
        else:
            pred_local_name = pred_uri.split('/')[-1]
        pred_cleaned_for_output = split_camel_case_to_lower_words(
            pred_local_name)

        # Create a separate entry for each subject
        for subj_uri in subjects:
            cleaned_subj = clean_uri_for_llm_value(subj_uri)
            if cleaned_subj:
                # Add tuple: (cleaned_subject, cleaned_predicate, instance_name)
                incoming_tuples.append(
                    (cleaned_subj, pred_cleaned_for_output, instance_name_for_part2))

    # Sort the tuples primarily by subject name, then by predicate name
    incoming_tuples.sort()

    # Generate output lines for part 2 from sorted tuples
    for subj, pred, obj in incoming_tuples:
        output_lines_part2.append(f"{subj} : {pred} : {obj}")

    # --- Combine Output ---
    final_output = "\n".join(output_lines_part1)
    # Add separator only if both parts have content
    if output_lines_part1 and output_lines_part2:
        final_output += "\n\n"  # Add blank line separator
    if output_lines_part2:
        final_output += "\n".join(output_lines_part2)

    # Handle cases where only incoming relationships were found
    if not output_lines_part1 and output_lines_part2:
        final_output = f"name: {instance_name_cleaned}\n(No outgoing properties found)\n\n" + \
            final_output

    return final_output


# Step 1: Fetch all ontology classes
def fetch_classes():
    logger.info("Fetching ontology classes...")
    
    class_query = r"""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>

    SELECT ?class
    FROM <http://dbpedia.org/model>
    WHERE {
      ?class a owl:Class .
      FILTER (
        regex(STRAFTER(STR(?class), "http://dbpedia.org/ontology/"), "^[\\x00-\\x7F]+$")
      )
    }
    ORDER BY ?class
    """
    try:
        sparql = get_sparql(return_format=JSON)
        sparql.setQuery(class_query)
        results = sparql.query().convert()
        classes = [result["class"]["value"] for result in results["results"]["bindings"]]
        logger.info(f"Fetched {len(classes)} classes.")
        return classes
    except Exception as e:
        logger.exception(f"[Error] Fetching classes: {type(e).__name__} - {e}")
        return []


# Step 2: Fetch instances of a class
def fetch_instances_for_class(ontology_class):
    logger.info(f"Fetching intances of type class {ontology_class}")
    
    instance_query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?instance
    FROM <http://dbpedia.org/model>
    FROM <http://dbpedia.org/data>
    WHERE {{
        BIND(<{ontology_class}> AS ?entity)
        ?instance a ?entity .
    }}
    ORDER BY ?instance
    """
    try:
        sparql = get_sparql(return_format=JSON)
        sparql.setQuery(instance_query)
        results = sparql.query().convert()
        instances = [result["instance"]["value"] for result in results["results"]["bindings"]]
        return instances
    except Exception as e:
        logger.exception(f"[Error] Fetching instances for {ontology_class}: {type(e).__name__} - {e}")
        try:
            with file_lock: # Acquire lock before writing to shared log file
                with open(FAILED_CLASS_LOG, "a", encoding="utf-8") as f:
                    f.write(ontology_class + "\n")
        except Exception as file_err:
            logger.exception(f"[Error] Saving failed class IRI to {FAILED_CLASS_LOG}: {file_err}")
        return []


# Step 3: Describe a single instance
def describe_instance(instance_iri):
    
    query = f"DESCRIBE <{instance_iri}>"
    
    try:
        sparql = get_sparql(return_format=N3)
        sparql.setQuery(query)
        result_bytes = sparql.query().convert()
        # rdf_n3_string = result_bytes.decode('utf-8') if result_bytes else ""
        rdf_n3_string = result_bytes.decode('utf-8') if isinstance(result_bytes, bytes) else str(result_bytes)

        # Extract structured data
        structured_data = extract_structured_description(rdf_n3_string, instance_iri)

        if structured_data is None:
            return None

        # Format for LLM
        llm_input_string = format_for_llm_custom_layout(structured_data)
        return llm_input_string

    except Exception as e:
        logger.exception(f"[Error] Describing {instance_iri}: {type(e).__name__} - {e}")
        try:
            with file_lock:
                with open(FAILED_INSTANCE_LOG, "a", encoding="utf-8") as f:
                    f.write(instance_iri + "\n")
        except Exception as file_err:
            logger.exception(f"[Error] Saving failed instance IRI to {FAILED_INSTANCE_LOG}: {file_err}")
        return None

# Step 1.1: process ontology class)
def process_class(ontology_class, output_filename, lock):
    """Fetches instances for a class and describes them, writing results to file."""
    thread_name = threading.current_thread().name # Included in logger format
    logger.info(f"Processing class: {ontology_class}")
    # Initialize count outside the main try block
    processed_instance_count = 0

    try: # Add a broad try block for the whole function body
        instances = fetch_instances_for_class(ontology_class)
        if not instances:
            logger.info(f"No instances found or fetch failed for class {ontology_class}.")
            # Return 0 instances processed, this is a successful completion of the task (finding nothing)
            return 0

        instance_count = len(instances)
        logger.info(f"Fetched {instance_count} instances for class {ontology_class}. Describing...")

        for i, iri in enumerate(instances):
            # Optional: Add a small delay *between* instance descriptions if hitting rate limits aggressively
            # time.sleep(0.05)
            # logger.debug(f"Describing instance {i+1}/{instance_count}: {iri}")

            # Inner try-except for describe_instance and file writing per instance
            try:
                describe_instance_str = describe_instance(iri) # This might return None or raise an exception

                if describe_instance_str is not None:
                    output_data = {
                        "iri": iri,
                        "description": describe_instance_str
                    }

                    # ensure_ascii=False is important for non-ASCII characters
                    json_line = json.dumps(output_data, ensure_ascii=False)

                    # Acquire lock before writing to the shared file
                    with lock:
                        with open(output_filename, "a", encoding="utf-8") as f:
                            f.write(json_line + "\n")
                    processed_instance_count += 1 # This is int += 1, should be safe

            except Exception as inner_e:
                # Log errors occurring during description or writing for a single instance
                # but continue processing the rest of the instances for this class.
                logger.error(f"Error processing instance {iri} in class {ontology_class}: {inner_e}", exc_info=False)
                # Optionally log to failed instance log here as well if describe_instance didn't already
                try:
                    with file_lock:
                        # Avoid duplicate logging if describe_instance already logged it
                        # This logs errors during the json.dumps or file write phase specifically
                        if not isinstance(inner_e, (TypeError, ValueError)): # Example filter
                             with open(FAILED_INSTANCE_LOG, "a", encoding="utf-8") as f:
                                 f.write(f"{iri} (Error in process_class loop: {type(inner_e).__name__})\n")
                except Exception as file_err:
                    logger.exception(f"Saving failed instance IRI (process_class loop) to {FAILED_INSTANCE_LOG} failed: {file_err}")


        logger.info(f"Finished processing class {ontology_class}. Successfully described {processed_instance_count}/{instance_count} instances.")
        # Return the count of successfully processed instances for this class
        return processed_instance_count

    except TypeError as te:
        # Explicitly catch the TypeError that the main thread reported seeing
        logger.exception(f"FATAL TypeError occurred while processing class {ontology_class}: {te}. Aborting processing for this class.")
        # Log failure to the class log
        try:
            with file_lock:
                with open(FAILED_CLASS_LOG, "a", encoding="utf-8") as f:
                    f.write(f"{ontology_class} (FATAL TypeError in process_class)\n")
        except Exception as file_err:
            logger.exception(f"Saving failed class IRI (TypeError) to {FAILED_CLASS_LOG} failed: {file_err}")
        # Reraise the exception so future.exception() in the main thread gets it
        raise te

    except Exception as e:
        # Catch any other unexpected errors during the class processing (e.g., during fetch_instances)
        logger.exception(f"Unexpected fatal error processing class {ontology_class}: {e}. Aborting processing for this class.")
        # Log failure to the class log
        try:
            with file_lock:
                with open(FAILED_CLASS_LOG, "a", encoding="utf-8") as f:
                    f.write(f"{ontology_class} (Unexpected Error: {type(e).__name__})\n")
        except Exception as file_err:
             logger.exception(f"Saving failed class IRI (Unexpected Error) to {FAILED_CLASS_LOG} failed: {file_err}")
        # Reraise the exception so future.exception() in the main thread gets it
        raise e


def main():
    try:
        os.makedirs(FAILED_LOG_DIR, exist_ok=True)
        os.makedirs(OUTPUT_FILENAME_DIR, exist_ok=True)
        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f: pass
        with open(FAILED_CLASS_LOG, "w", encoding="utf-8") as f: pass
        with open(FAILED_INSTANCE_LOG, "w", encoding="utf-8") as f: pass
    except Exception as e:
        logger.exception(f"[Error] Creating directories or clearing files: {e}")
        return

    owl_classes = fetch_classes()
    if not owl_classes:
        logger.info("No classes fetched. Exiting.")
        return
    
    total_classes = len(owl_classes)
    processed_class_count = 0
    total_successful_instances = 0
    futures = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for owl_class in owl_classes:
            future = executor.submit(process_class, owl_class, OUTPUT_FILENAME, file_lock)
            futures.append(future)
            logger.info(f"{len(futures)} tasks submitted. Waiting for completion...")

        for future in concurrent.futures.as_completed(futures):
            task_exception = future.exception()
            if task_exception:
                 # Log exceptions raised within the worker threads
                 logger.error(f"A class processing task failed: {task_exception}", exc_info=False)
            else:
                try:
                    instances_processed_in_class = future.result()
                    total_successful_instances += instances_processed_in_class
                except Exception as e:
            
                    logger.error(f"A class processing task failed: {e}", exc_info=False) # Set exc_info=True for full traceback

            processed_class_count += 1
            # Log progress periodically or on each completion
            if processed_class_count % 10 == 0 or processed_class_count == total_classes:
                 logger.info(f"Progress: {processed_class_count}/{total_classes} classes processed.")
                 
    logger.info("-" * 30)
    logger.info(f"Processing complete.")
    logger.info(f"Total classes processed: {processed_class_count}/{total_classes}")
    logger.info(f"Total successful instance descriptions written: {total_successful_instances}")
    logger.info(f"Descriptions saved to {OUTPUT_FILENAME}")
    logger.info(f"Check {FAILED_CLASS_LOG} and {FAILED_INSTANCE_LOG} for any errors.")
    logger.info("-" * 30)