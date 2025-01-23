import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import dotenv
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from pymatgen.core import Composition, Structure

dotenv.load_dotenv()


def composition_augmentation(composition: Composition):
    comp_list = list(
        set(
            [
                composition.reduced_formula,  # reduced formula
                composition.reduced_composition.alphabetical_formula,  # reduced alphabetical formula
                composition.reduced_composition.iupac_formula,  # reduced IUPAC formula
                composition.reduced_composition.hill_formula,  # reduced Hill formula
            ]
        )
    )
    return comp_list


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def process_row(row, client, prompt_dir, df_abstract_data):
    st = Structure.from_str(row["cif"], fmt="cif")
    # composition
    composition = st.composition
    reduced_formula = composition.reduced_formula
    comp_list = composition_augmentation(composition)

    # crystal system
    crystal_system = row["crystal_system"]

    # stability: stable < 0 eV, metastable < 0.25 eV, unstable > 0.25 eV
    stability = (
        "stable"
        if row["energy_above_hull"] == 0
        else "metastable" if row["energy_above_hull"] < 0.25 else "unstable"
    )

    # metallic: metallic < 0.1 eV, insulator > 0.1 eV
    metallic = "metallic" if row["band_gap"] < 0.1 else "insulator"

    # mineral
    mineral = "" if pd.isna(row["mineral"]) else row["mineral"]

    # paper data
    paper_data = ""
    if row["material_id"] in df_abstract_data["material_id"].values:
        df = df_abstract_data[df_abstract_data["material_id"] == row["material_id"]]
        for i, (_, row) in enumerate(df.iterrows()):
            paper_data += f"paper{i+1} - title: {row['title'][2:-2]} | abstract: {row['abstract']}\n"
    paper_data = paper_data[:3000]  # limit the length of paper_data

    template = f"""provide concise captions for "{reduced_formula}" with the following properties:


{crystal_system}
{stability}
{metallic}
{mineral}
{paper_data}

Here are some examples for other crystal systems:
1. Orthorhombic crystal structure of ZnMnO4
2. metastable crystal structure of LiO2
3. Si1 C1 crystal structure with metallic properties

Please provide "five concise captions" to describe the crystal structure with the various compound name: {", ".join(comp_list)}
"""
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": template,
            }
        ],
        model="gpt-4o-mini",
    )
    output = chat_completion.choices[0].message.content

    with open(prompt_dir / f"{row['material_id']}.txt", "w") as f:
        f.write(output)


def main(data_dir: str = "."):

    data_dir = Path(data_dir)
    prompt_dir = data_dir / "prompts"
    prompt_dir.mkdir(exist_ok=True, parents=True)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OpenAI API key is not set")

    # Read the data
    df_total = pd.read_csv(data_dir / "mp-total.csv")
    print(f"Total crystal text: {len(df_total)}")
    df_abstract_data = pd.read_csv(data_dir / "abstract-data.csv")
    df_abstract_data = df_abstract_data[~df_abstract_data["abstract"].isna()]
    print(f"Abstract data: {len(df_abstract_data)}")
    # remove rows if there is prompt already
    already_prompt = prompt_dir.glob("*.txt")
    already_prompt = [p.stem for p in already_prompt]
    print(f"Already prompt: {len(already_prompt)}")
    df_total = df_total[~df_total["material_id"].isin(already_prompt)]
    print(f"Remaining materials captions: {len(df_total)}")

    # Create a client
    client = client = OpenAI(
        api_key=openai_api_key,
    )
    print("OpenAI client created")

    # Assuming df_crystal_text is your DataFrame
    with ThreadPoolExecutor() as executor:
        # Submit all tasks and create a list of futures
        futures = [
            executor.submit(process_row, row, client, prompt_dir, df_abstract_data)
            for i, row in df_total.iterrows()
        ]

        # Wrap as_completed in tqdm to display progress
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing rows"
        ):
            try:
                result = (
                    future.result()
                )  # In this case, we don't expect a result since the function writes to a file
            except Exception as exc:
                print(f"Task generated an exception: {exc}")


if __name__ == "__main__":
    main()
