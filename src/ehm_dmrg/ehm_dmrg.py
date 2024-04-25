from pathlib import Path

import pandas as pd
import pyscf.tools.fcidump

import ehm_dmrg.dmrg_features as dmrg_features


def dmrg_features_from_fcidump(
    fcidump_filepath,
    MOLPRO_ORBSYM=True,
    calc_resistance_matrix_properties=True,
    verbosity=2,
):
    fcidump_filepath = Path(fcidump_filepath)

    # Load the fcidump file
    fcidump_dict = pyscf.tools.fcidump.read(
        filename=fcidump_filepath, molpro_orbsym=MOLPRO_ORBSYM, verbose=bool(verbosity)
    )

    # Extract the parameters
    num_orbitals = fcidump_dict["NORB"]
    num_electrons = fcidump_dict["NELEC"]
    num_spin_orbitals = 2 * num_orbitals
    two_S = fcidump_dict["MS2"]
    orbsym = fcidump_dict["ORBSYM"]
    isym = fcidump_dict["ISYM"]
    one_body_tensor = fcidump_dict["H1"]
    two_body_tensor = fcidump_dict["H2"]
    two_body_tensor = pyscf.ao2mo.restore(1, two_body_tensor, num_orbitals)
    # Tensors assume the Hamiltonian is in the form:
    # H = E_0 + h_ij a†_i a_j + 0.5*g_ijkl a†_i a†_k a_l a_j
    # with no permutation symmetry compression for two_body_tensor. ijkl are spatial orbitals.
    print("one_body_tensor.shape", one_body_tensor.shape)
    print("two_body_tensor.shape", two_body_tensor.shape)
    feature_dict = {
        "num_orbitals": num_orbitals,
        "num_electrons": num_electrons,
        "num_spin_orbitals": num_spin_orbitals,
        "two_S": two_S,
        "orbsym": orbsym,
        "isym": isym,
    }
    print(feature_dict)

    tensors_features_dict = dmrg_features.get_dmrg_features(
        one_body_tensor=one_body_tensor,
        two_body_tensor=two_body_tensor,
        num_electrons=num_electrons,
        # spin_orbitals_bool=True,
        # tbt_one_half_convention=True,
        calc_resistance_matrix_properties=calc_resistance_matrix_properties,
    )

    feature_dict.update(tensors_features_dict)

    return feature_dict


def load_dmrg_output_data(datafile_path):
    # Data is assumed to be in a csv file
    datafile_path = Path(datafile_path)
    # Read from csv
    data_df = pd.read_csv(datafile_path)
