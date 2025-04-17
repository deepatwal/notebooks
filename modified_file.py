
MAX_WORKERS = 4 # << REDUCE THIS SIGNIFICANTLY (Start with 2-4)
MAX_RETRIES = 3 # Number of retries for network errors
RETRY_DELAY = 1.5 # Seconds to wait between retries


file_lock = threading.Lock()


def split_camel_case_to_lower_words(name):
    # ... (no changes) ...
    if not name: return ""
    if name.islower() or '_' in name or not re.search('[A-Z]', name): return name.replace('_', ' ').lower()
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)
    return s2.lower()

def clean_uri_for_llm_key(uri_str):
    # ... (no changes) ...
    if not uri_str: return "unknown property"
    if uri_str == str(RDF.type): return "type"
    if uri_str == str(RDFS.label): return "label"
    if uri_str == str(FOAF.name): return "name"
    if '#' in uri_str: name = uri_str.split('#')[-1]
    else: name = uri_str.split('/')[-1]
    return split_camel_case_to_lower_words(name)

def clean_uri_for_llm_value(uri_str):
    # ... (no changes) ...
    if not uri_str: return "Unknown Resource"
    try:
        if '#' in uri_str: name = uri_str.split('#')[-1]
        else: name = uri_str.split('/')[-1]
        name = name.replace('%28', '(').replace('%29', ')')
        name = name.replace('%2C', ',').replace('%27', "'")
        if uri_str.startswith("http://www.wikidata.org/entity/"): return name
        return name.replace('_', ' ')
    except Exception:
        logger.warning(f"Could not cleanly parse URI for value: {uri_str}", exc_info=False)
        return uri_str

def format_rdf_term_for_llm_value(term_data):
    # ... (no changes) ...
    if isinstance(term_data, dict):
        val = term_data.get("value", "")
        val = re.sub(r'@\w+(-[A-Za-z0-9]+)*$', '', val)
        val = re.sub(r'\^\^<.*>$', '', val)
        val = val.strip('"')
        return val
    elif isinstance(term_data, str): return clean_uri_for_llm_value(term_data)
    else: return str(term_data)

def format_rdf_term(term):
    # ... (no changes) ...
    if isinstance(term, Literal):
        dt = str(term.datatype) if term.datatype else None
        lang = term.language if term.language else None
        if dt is None and lang: dt = str(RDF.langString)
        elif dt is None: dt = str(XSD.string)
        try: term_str_value = str(term)
        except ValueError as ve:
            logger.warning(f"[Warning] Failed to parse literal '{term}' with datatype {dt}: {ve}")
            return {"value": str(term), "language": lang, "datatype": dt}
        return {"value": term_str_value, "language": lang, "datatype": dt}
    elif isinstance(term, URIRef): return str(term)
    else: return str(term)

def extract_structured_description(rdf_n3_string, instance_iri):
    # ... (no changes) ...
    if not rdf_n3_string: return {"instance_iri": instance_iri, "outgoing": {}, "incoming": {}}
    g = Graph()
    try: g.parse(data=rdf_n3_string, format="n3", publicID=instance_iri)
    except Exception as e:
        logger.error(f"Parsing N3 data for {instance_iri}: {type(e).__name__} - {e}", exc_info=False)
        return None
    instance_ref = URIRef(instance_iri)
    outgoing_data = defaultdict(list)
    incoming_data = defaultdict(list)
    for pred, obj in g.predicate_objects(subject=instance_ref):
        pred_uri_str = str(pred)
        formatted_obj = format_rdf_term(obj)
        if formatted_obj not in outgoing_data[pred_uri_str]: outgoing_data[pred_uri_str].append(formatted_obj)
    for subj, pred in g.subject_predicates(object=instance_ref):
        if subj == instance_ref: continue
        pred_uri_str = str(pred)
        subj_uri_str = str(subj)
        if subj_uri_str not in incoming_data[pred_uri_str]: incoming_data[pred_uri_str].append(subj_uri_str)
    def sort_key(item):
        if isinstance(item, dict): return (str(item.get('value', '')), str(item.get('language', '')), str(item.get('datatype', '')))
        return str(item)
    final_outgoing = {pred: sorted(values, key=sort_key) for pred, values in outgoing_data.items()}
    final_incoming = {pred: sorted(values) for pred, values in incoming_data.items()}
    return {"instance_iri": instance_iri, "outgoing": final_outgoing, "incoming": final_incoming}

def format_for_llm_custom_layout(structured_data):
    # ... (no changes) ...
    if not structured_data or (not structured_data.get("outgoing") and not structured_data.get("incoming")):
        instance_iri = structured_data.get("instance_iri", "Unknown Instance")
        instance_name = clean_uri_for_llm_value(instance_iri)
        return f"name: {instance_name}\n(No description properties found)"
    instance_iri = structured_data.get("instance_iri")
    instance_name_cleaned = clean_uri_for_llm_value(instance_iri)
    output_lines_part1 = []
    output_lines_part2 = []
    outgoing_properties = structured_data.get("outgoing", {})
    primary_name_val = instance_name_cleaned
    temp_outgoing_formatted = {}
    sorted_pred_uris = sorted(outgoing_properties.keys(), key=clean_uri_for_llm_key)
    for pred_uri in sorted_pred_uris:
        llm_key = clean_uri_for_llm_key(pred_uri)
        values = outgoing_properties[pred_uri]
        cleaned_values_for_key = []
        for term_data in values:
            cleaned_val = format_rdf_term_for_llm_value(term_data)
            if cleaned_val and cleaned_val not in cleaned_values_for_key: cleaned_values_for_key.append(cleaned_val)
        if cleaned_values_for_key:
            value_string = ", ".join(cleaned_values_for_key)
            temp_outgoing_formatted[llm_key] = value_string
            if llm_key in ['name', 'label']:
                 if primary_name_val == instance_name_cleaned or llm_key == 'name': primary_name_val = value_string
    name_key_found = None
    if 'name' in temp_outgoing_formatted:
        output_lines_part1.append(f"name: {temp_outgoing_formatted['name']}")
        name_key_found = 'name'
    elif 'label' in temp_outgoing_formatted:
        output_lines_part1.append(f"label: {temp_outgoing_formatted['label']}")
        name_key_found = 'label'
    elif instance_name_cleaned: output_lines_part1.append(f"name: {instance_name_cleaned}")
    for key in sorted(temp_outgoing_formatted.keys()):
        if key == name_key_found: continue
        output_lines_part1.append(f"{key}: {temp_outgoing_formatted[key]}")
    incoming_relationships = structured_data.get("incoming", {})
    instance_name_for_part2 = primary_name_val
    incoming_tuples = []
    sorted_incoming_pred_uris = sorted(incoming_relationships.keys(), key=clean_uri_for_llm_key)
    for pred_uri in sorted_incoming_pred_uris:
        subjects = incoming_relationships[pred_uri]
        pred_cleaned_for_output = clean_uri_for_llm_key(pred_uri)
        for subj_uri in subjects:
            cleaned_subj = clean_uri_for_llm_value(subj_uri)
            if cleaned_subj: incoming_tuples.append((cleaned_subj, pred_cleaned_for_output, instance_name_for_part2))
    incoming_tuples.sort()
    for subj, pred, obj in incoming_tuples: output_lines_part2.append(f"{subj} : {pred} : {obj}")
    final_output = "\n".join(output_lines_part1)
    if output_lines_part1 and output_lines_part2: final_output += "\n\n---\n\n"
    if output_lines_part2: final_output += "\n".join(output_lines_part2)
    if not output_lines_part1 and output_lines_part2:
        name_line = f"name: {instance_name_cleaned}"
        if 'name' in temp_outgoing_formatted: name_line = f"name: {temp_outgoing_formatted['name']}"
        elif 'label' in temp_outgoing_formatted: name_line = f"label: {temp_outgoing_formatted['label']}"
        final_output = f"{name_line}\n(No outgoing properties found)\n\n---\n\n" + "\n".join(output_lines_part2)
    elif not output_lines_part1 and not output_lines_part2: return f"name: {instance_name_cleaned}\n(No description properties found)"
    return final_output

# --- Data Fetching and Processing Functions ---

def _execute_sparql_query(sparql, query_context="query"):
    """Executes a SPARQL query with retry logic for specific network errors."""
    retries = 0
    while retries <= MAX_RETRIES:
        try:
            # logger.debug(f"Attempt {retries+1}/{MAX_RETRIES+1} for {query_context}")
            result = sparql.query().convert()
            return result # Success
        except urllib.error.URLError as e:
            # Check if it's the specific WinError 10048 or a general timeout/network issue
            should_retry = False
            if isinstance(e.reason, OSError) and e.reason.winerror == 10048:
                logger.warning(f"Port exhaustion error (WinError 10048) during {query_context}. Retrying in {RETRY_DELAY}s... ({retries+1}/{MAX_RETRIES})")
                should_retry = True
            elif "timed out" in str(e).lower():
                 logger.warning(f"Timeout error during {query_context}. Retrying in {RETRY_DELAY}s... ({retries+1}/{MAX_RETRIES})")
                 should_retry = True
            # Add other retryable URLError conditions if needed

            if should_retry and retries < MAX_RETRIES:
                retries += 1
                time.sleep(RETRY_DELAY * (retries)) # Exponential backoff factor
            else:
                logger.error(f"URLError during {query_context} after {retries} retries: {e}", exc_info=False)
                raise e # Reraise the exception if max retries reached or not retryable
        except Exception as e:
            # Catch other potential exceptions from SPARQLWrapper or network stack
            logger.error(f"Unexpected error during {query_context}: {type(e).__name__} - {e}", exc_info=False)
            raise e # Reraise immediately, not typically retryable
    # Should not be reached if loop logic is correct, but raise error if it is
    raise Exception(f"Query failed for {query_context} after {MAX_RETRIES} retries.")


# Step 1: Fetch all ontology classes
def fetch_classes():
    logger.info("Fetching ontology classes...")
    class_query = r"""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX dbo: <http://dbpedia.org/ontology/>
    SELECT DISTINCT ?class WHERE {
      GRAPH <http://dbpedia.org/ontology/> { ?class a owl:Class . }
      FILTER (!isBlank(?class))
      FILTER STRSTARTS(STR(?class), STR(dbo:))
    } ORDER BY ?class LIMIT 10000
    """
    try:
        sparql = get_sparql(return_format=JSON)
        sparql.setQuery(class_query)
        # Use the retry executor
        results = _execute_sparql_query(sparql, "fetch_classes")
        classes = [result["class"]["value"] for result in results["results"]["bindings"]]
        logger.info(f"Fetched {len(classes)} classes.")
        return classes
    except Exception as e:
        # Error already logged by _execute_sparql_query or get_sparql
        logger.exception(f"Fetching classes failed ultimately: {type(e).__name__} - {e}")
        return []


# Step 2: Fetch instances of a class
def fetch_instances_for_class(ontology_class):
    instance_query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dbo: <http://dbpedia.org/ontology/>
    SELECT DISTINCT ?instance WHERE {{
        ?instance a <{ontology_class}> .
        FILTER (!isBlank(?instance))
        FILTER (STRSTARTS(STR(?instance), "http://dbpedia.org/resource/"))
    }} ORDER BY ?instance LIMIT 10000
    """
    try:
        sparql = get_sparql(return_format=JSON)
        sparql.setQuery(instance_query)
        # Use the retry executor
        results = _execute_sparql_query(sparql, f"fetch_instances_for_class({ontology_class})")
        instances = [result["instance"]["value"] for result in results["results"]["bindings"]]
        return instances
    except Exception as e:
        # Error already logged by _execute_sparql_query
        logger.error(f"Fetching instances for {ontology_class} failed ultimately.", exc_info=False)
        try:
            with file_lock:
                with open(FAILED_CLASS_LOG, "a", encoding="utf-8") as f:
                    f.write(f"{ontology_class} (Fetch failed: {type(e).__name__})\n")
        except Exception as file_err:
            logger.exception(f"Saving failed class IRI to {FAILED_CLASS_LOG} failed: {file_err}")
        return []


# Step 3: Describe a single instance
def describe_instance(instance_iri):
    query = f"DESCRIBE <{instance_iri}>"
    try:
        sparql = get_sparql(return_format=N3)
        sparql.setQuery(query)
        # Use the retry executor
        result_bytes = _execute_sparql_query(sparql, f"describe_instance({instance_iri})")

        # --- Rest of the function remains the same ---
        if isinstance(result_bytes, bytes):
             try: rdf_n3_string = result_bytes.decode('utf-8')
             except UnicodeDecodeError:
                 logger.warning(f"UTF-8 decoding failed for DESCRIBE <{instance_iri}>. Trying latin-1.")
                 rdf_n3_string = result_bytes.decode('latin-1', errors='ignore')
        elif result_bytes: rdf_n3_string = str(result_bytes)
        else: rdf_n3_string = ""

        if not rdf_n3_string:
             logger.warning(f"Received empty DESCRIBE result for <{instance_iri}>")
             structured_data = {"instance_iri": instance_iri, "outgoing": {}, "incoming": {}}
        else:
            structured_data = extract_structured_description(rdf_n3_string, instance_iri)

        if structured_data is None:
            logger.warning(f"Failed to extract structured data for {instance_iri} (likely parsing error).")
            try:
                with file_lock:
                    with open(FAILED_INSTANCE_LOG, "a", encoding="utf-8") as f:
                        f.write(f"{instance_iri} (parse error)\n")
            except Exception as file_err:
                logger.exception(f"Saving failed instance IRI (parse error) to {FAILED_INSTANCE_LOG} failed: {file_err}")
            return None

        llm_input_string = format_for_llm_custom_layout(structured_data)
        return llm_input_string

    except Exception as e:
        # Catch errors from _execute_sparql_query or subsequent processing
        # Error should have been logged by _execute_sparql_query if it was a query failure
        if not isinstance(e, urllib.error.URLError): # Log again if it wasn't a URLError already logged by helper
             logger.error(f"Describing {instance_iri} failed: {type(e).__name__} - {e}", exc_info=False)
        try:
            with file_lock:
                with open(FAILED_INSTANCE_LOG, "a", encoding="utf-8") as f:
                    f.write(f"{instance_iri} (Describe failed: {type(e).__name__})\n")
        except Exception as file_err:
            logger.exception(f"Saving failed instance IRI to {FAILED_INSTANCE_LOG} failed: {file_err}")
        return None


# Step 4: Process a single ontology class
def process_class(ontology_class, output_filename, lock):
    """Fetches instances for a class and describes them, writing results to file."""
    thread_name = threading.current_thread().name
    logger.info(f"Processing class: {ontology_class}")
    processed_instance_count = 0

    try:
        instances = fetch_instances_for_class(ontology_class)
        if not instances:
            logger.info(f"No instances found or fetch failed for class {ontology_class}.")
            return 0 # Successful completion, zero instances

        instance_count = len(instances)
        logger.info(f"Fetched {instance_count} instances for class {ontology_class}. Describing...")

        for i, iri in enumerate(instances):
            # Optional: Add delay here if reducing workers isn't enough
            # time.sleep(0.05)
            try:
                describe_instance_str = describe_instance(iri) # Retries are inside describe_instance

                if describe_instance_str is not None:
                    output_data = { "iri": iri, "class": ontology_class, "description": describe_instance_str }
                    json_line = json.dumps(output_data, ensure_ascii=False)
                    with lock:
                        with open(output_filename, "a", encoding="utf-8") as f:
                            f.write(json_line + "\n")
                    processed_instance_count += 1

            except Exception as inner_e:
                # Log errors during json.dumps or file write (less likely)
                logger.error(f"Error processing instance {iri} after description: {inner_e}", exc_info=False)

        logger.info(f"Finished processing class {ontology_class}. Successfully described {processed_instance_count}/{instance_count} instances.")
        return processed_instance_count

    except Exception as e:
        # Catch unexpected errors during the overall class processing
        logger.exception(f"FATAL error processing class {ontology_class}: {e}. Aborting processing for this class.")
        try:
            with file_lock:
                with open(FAILED_CLASS_LOG, "a", encoding="utf-8") as f:
                    f.write(f"{ontology_class} (FATAL Error: {type(e).__name__})\n")
        except Exception as file_err:
             logger.exception(f"Saving failed class IRI (FATAL Error) to {FAILED_CLASS_LOG} failed: {file_err}")
        # Reraise so the main thread knows the task failed completely
        raise e


# --- Main Execution Logic ---
def main():
    try:
        os.makedirs(FAILED_LOG_DIR, exist_ok=True)
        os.makedirs(OUTPUT_FILENAME_DIR, exist_ok=True)
        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f: pass
        with open(FAILED_CLASS_LOG, "w", encoding="utf-8") as f: pass
        with open(FAILED_INSTANCE_LOG, "w", encoding="utf-8") as f: pass
        logger.info(f"Output will be saved to: {OUTPUT_FILENAME}")
        logger.info(f"Failed classes log: {FAILED_CLASS_LOG}")
        logger.info(f"Failed instances log: {FAILED_INSTANCE_LOG}")
        logger.info(f"Using MAX_WORKERS = {MAX_WORKERS}, MAX_RETRIES = {MAX_RETRIES}, RETRY_DELAY = {RETRY_DELAY}s")

    except Exception as e:
        logger.exception(f"Creating directories or clearing files failed: {e}")
        return

    start_time = time.time()
    owl_classes = fetch_classes() # Retries included here
    if not owl_classes:
        logger.info("No classes fetched. Exiting.")
        return

    total_classes = len(owl_classes)
    logger.info(f"Fetched {total_classes} classes. Starting parallel processing...")

    processed_class_count = 0
    total_successful_instances = 0
    futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="Worker") as executor:
        for owl_class in owl_classes:
            future = executor.submit(process_class, owl_class, OUTPUT_FILENAME, file_lock)
            futures.append(future)

        logger.info(f"{len(futures)} tasks submitted. Waiting for completion...")
        for future in concurrent.futures.as_completed(futures):
            task_exception = future.exception()
            if task_exception:
                 logger.error(f"A class processing task failed ultimately: {task_exception}", exc_info=False)
            else:
                try:
                    instances_processed_in_class = future.result()
                    if instances_processed_in_class is not None:
                        total_successful_instances += instances_processed_in_class
                    else:
                        logger.warning("A class processing task completed successfully but returned None.")
                except Exception as e:
                    logger.error(f"Error retrieving result from a completed task: {e}", exc_info=False)

            processed_class_count += 1
            if processed_class_count % 20 == 0 or processed_class_count == total_classes:
                 logger.info(f"Progress: {processed_class_count}/{total_classes} classes processed.")

    end_time = time.time()
    logger.info("-" * 40)
    logger.info(f"Processing complete.")
    logger.info(f"Total classes processed (attempted): {processed_class_count}/{total_classes}")
    logger.info(f"Total successful instance descriptions written: {total_successful_instances}")
    logger.info(f"Total time taken: {end_time - start_time:.2f} seconds")
    logger.info(f"Descriptions saved to {OUTPUT_FILENAME}")
    logger.info(f"Check {FAILED_CLASS_LOG} and {FAILED_INSTANCE_LOG} for any errors.")
    logger.info("-" * 40)
