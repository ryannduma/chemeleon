from typing import List
from pathlib import Path
import warnings
import pandas as pd

from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice

from torch.utils.data import Dataset
from torch_geometric.data import Data

from chemeleon.datasets.dataset_utils import (
    atoms_to_pyg_data,
    convert_lattice_polar_decomposition,
)


warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF:")


class MPDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        text_guide: bool = False,
        text_targets: List[str] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.text_guide = text_guide
        self.text_targets = text_targets

        path_data = Path(self.data_dir, f"{self.split}.csv")
        self.data = pd.read_csv(path_data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Data:
        data = self.data.iloc[idx]
        str_cif = data.cif
        # Read cif
        st = Structure.from_str(str_cif, fmt="cif")
        # Niggli reduction
        reduced_st = st.get_reduced_structure()
        canonical_st = Structure(
            lattice=Lattice.from_parameters(*reduced_st.lattice.parameters),
            species=reduced_st.species,
            coords=reduced_st.frac_coords,
            coords_are_cartesian=False,
        )
        # Convert to ase atoms
        atoms = canonical_st.to_ase_atoms()
        # Convert lattice to polar decomposition to make it symmetric
        atoms.set_cell(convert_lattice_polar_decomposition(atoms.cell))
        # Add text guide
        if self.text_guide:
            properties = data[self.text_targets].values
            if len(self.text_targets) == 1:
                text = str(properties[0])
            else:
                text = ", ".join(
                    [
                        f"{self.text_targets[i]}: {properties[i]}"
                        for i in range(len(properties))
                    ]
                )
            return atoms_to_pyg_data(atoms, text=text)
        else:
            return atoms_to_pyg_data(atoms)
