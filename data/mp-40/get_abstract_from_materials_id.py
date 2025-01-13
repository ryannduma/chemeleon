import os
import logging
import json
from pathlib import Path
from tqdm import tqdm
from collections import Counter

import requests
import dotenv
import pandas as pd
import bibtexparser
from pymatgen.ext.matproj import MPRester

# Basic configurations
dotenv.load_dotenv()
BASE_PATH = Path(".")
MP_API_KEY = os.getenv("MP_API_KEY")
SCOPUS_API_KEY = os.getenv("SCOPUS_API_KEY")
if not MP_API_KEY or not SCOPUS_API_KEY:
    raise ValueError("API keys not found in environment variables.")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
error_counter = Counter()


def parse_bibtex_reference(bibtex_string: str) -> dict:
    """
    Safely parse a BibTeX string and return the first entry as a dictionary.
    Returns an empty dictionary if parsing fails or no entries exist.
    """
    try:
        bib_database = bibtexparser.loads(bibtex_string)
        if bib_database.entries:
            return bib_database.entries[0]
        else:
            logger.warning("BibTeX string parsed, but no entries found.")
            error_counter["BibTeX_Error:NoEntries"] += 1
            return {}
    except Exception as e:
        logger.error(f"Failed to parse BibTeX string: {e}")
        error_counter[f"BibTeX_Error_{type(e).__name__}"] += 1
        return {}


def search_crossref_by_metadata(title: str, year: str) -> dict:
    """
    Searches Crossref by provided title and year.
    Returns a dictionary with 'url', 'doi', 'title', 'publisher', and 'abstract'.
    In case of errors or no matches, returns an empty dictionary.
    """
    url = "https://api.crossref.org/works"
    params = {
        "query.title": title,
        "filter": f"from-pub-date:{year},until-pub-date:{year}",
        "rows": 1,  # Limit to 1 result for efficiency
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            items = data.get("message", {}).get("items", [])
            if items:
                item = items[0]
                return {
                    "url": item.get("URL"),
                    "doi": item.get("DOI"),
                    "title": item.get("title", []),
                    "publisher": item.get("publisher"),
                    "abstract": item.get("abstract"),
                }
            else:
                logger.info("No matching records found in Crossref.")
                error_counter["Crossref_Error:NoMatches"] += 1
                return {}
        else:
            logger.error(f"Crossref request failed with status {response.status_code}")
            error_counter["Crossref_Error:RequestFailed"] += 1
            return {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Crossref request error: {e}")
        error_counter["Crossref_Error:RequestError"] += 1
        return {}


def search_scopus_by_doi(doi: str) -> dict:
    """
    Searches Scopus by DOI.
    Returns a dictionary containing the abstract text under the 'abstract' key.
    If the request fails or the abstract is not found, returns an empty dict.
    """
    if not doi:
        logger.warning("No DOI provided to search Scopus.")
        return {}

    base_url = "https://api.elsevier.com/content/abstract/doi/"
    headers = {"Accept": "application/json", "X-ELS-APIKey": SCOPUS_API_KEY}

    try:
        response = requests.get(f"{base_url}{doi}", headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            abstract = (
                data.get("abstracts-retrieval-response", {})
                .get("coredata", {})
                .get("dc:description", "")
            )
            if abstract:
                return {"abstract": abstract}
            else:
                logger.info("Abstract not found in Scopus response.")
                error_counter["Scopus_Error:NoAbstract"] += 1
                return {}
        else:
            logger.error(
                f"Failed to retrieve Scopus data for DOI={doi}. "
                f"Status code: {response.status_code}, Response: {response.text}"
            )
            error_counter["Scopus_Error:RequestFailed"] += 1
            return {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Scopus request error for DOI={doi}: {e}")
        error_counter["Scopus_Error:RequestError"] += 1
        return {}


def main():
    # Read the CSV file with material IDs
    df_mp_api = pd.read_csv(BASE_PATH / "mp-api.csv")
    material_ids = df_mp_api["material_id"].values
    logger.info(f"Total material IDs loaded: {len(material_ids)}")

    save_dir = BASE_PATH / "abstract_data"
    save_dir.mkdir(exist_ok=True, parents=True)

    # Iterate over material IDs and retrieve references
    for material_id in tqdm(material_ids, desc="Processing materials"):
        # Retrieve references via MP Rester
        try:
            with MPRester(MP_API_KEY) as mpr:
                docs = mpr.get_material_id_references(
                    material_id
                )  # pylint: disable=E1129
        except Exception as e:
            logger.error(f"Error retrieving references for {material_id}: {e}")
            error_counter[f"MPRester_Error_{type(e).__name__}"] += 1
            continue

        # The last reference is typically an MP reference, so ignore it
        for i, bibtex_str in enumerate(docs[:-1]):
            data = {}
            logger.info(f"### Processing {material_id}_doi_{i}... ###")
            bibtex_entry = parse_bibtex_reference(bibtex_str)
            logger.info(f"BibTeX entry for {material_id}_doi_{i}: {bibtex_entry}")
            data.update(bibtex_entry)
            # Search Crossref
            title = bibtex_entry.get("title", "")
            year = bibtex_entry.get("year", "")
            crossref_data = search_crossref_by_metadata(title, year)
            logger.info(f"Crossref data for {material_id}_doi_{i}: {crossref_data}")
            data.update(crossref_data)

            # If Crossref gave us a DOI, attempt Scopus search
            if "doi" in crossref_data and crossref_data["doi"]:
                scopus_data = search_scopus_by_doi(crossref_data["doi"])
                logger.info(f"Scopus data for {material_id}_doi_{i}: {scopus_data}")
                if scopus_data:
                    data.update(scopus_data)

            logging.info(f"Final data for {material_id}_doi_{i}: {data}")
            # Save the data
            json_path = save_dir / f"{material_id}_doi_{i}.json"
            try:
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(data, f)
            except Exception as e:
                logger.error(f"Failed to write JSON for {material_id}_doi_{i}: {e}")
                error_counter[f"JSONWriteError_{type(e).__name__}"] += 1

            logger.info("-" * 40)

    # -- Print or log the final error counts --
    logger.info("===== ERROR SUMMARY =====")
    for err_type, count in error_counter.items():
        logger.info(f"{err_type}: {count}")


if __name__ == "__main__":
    main()
