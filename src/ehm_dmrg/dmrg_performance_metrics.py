from pathlib import Path

import numpy as np


def load_dmrg_output_data_from_csv(datafile_path):
    # Data is assumed to be in a csv file
    datafile_path = Path(datafile_path)
    # Read from csv
    # data_df = pd.read_csv(datafile_path)
    data_arrays = np.genfromtxt(datafile_path, delimiter=",", skip_header=1)
    # loop_index = data_arrays[:, 0]
    dmrg_energies = data_arrays[:, 0]
    bond_dimensions = data_arrays[:, 1]
    discarded_weights = data_arrays[:, 2]
    cpu_time_sec = data_arrays[:, 3]
    wall_time_sec = data_arrays[:, 4]
    return (
        dmrg_energies,
        bond_dimensions,
        discarded_weights,
        cpu_time_sec,
        wall_time_sec,
    )


def get_coarse_bond_dimension_needed_many_files(
    csv_file_list, folder_path, energy_change_threshold_hartrees=1e-3
):
    folder_path = Path(folder_path)

    coarse_bond_dimension_list = []
    total_loops_cpu_time_sec_list = []
    total_loops_wall_time_sec_list = []
    id_list = []
    for csv_file in csv_file_list:
        
        csv_file_path = folder_path / csv_file
        (
            dmrg_energies,
            bond_dimensions,
            discarded_weights,
            cpu_time_sec,
            wall_time_sec,
        ) = load_dmrg_output_data_from_csv(csv_file_path)

        # Sort by bond dimensions
        sorted_indices = np.argsort(bond_dimensions)
        bond_dimensions = bond_dimensions[sorted_indices]
        dmrg_energies = dmrg_energies[sorted_indices]
        discarded_weights = discarded_weights[sorted_indices]
        cpu_time_sec = cpu_time_sec[sorted_indices]
        wall_time_sec = wall_time_sec[sorted_indices]

        # Find the energy change
        energy_change = np.abs(np.diff(dmrg_energies))

        # Find the point where the energy change is less than the threshold
        # The coarse bond dimension is the bond dimension at this point
        # print(dmrg_energies)
        # print(energy_change)
        # print(np.where(energy_change < energy_change_threshold_hartrees))

        e_diff_threshold_index_array = np.where(
            energy_change < energy_change_threshold_hartrees
        )[0]
        if len(e_diff_threshold_index_array) == 0:
            print(f"No energy change less than {energy_change_threshold_hartrees} for {csv_file}")
            print(f"Min Energy change: {np.min(energy_change)}")
            continue
        e_diff_threshold_index=e_diff_threshold_index_array[0]
        coarse_bond_dimension = int(bond_dimensions[e_diff_threshold_index + 1])

        total_loops_cpu_time_sec = np.sum(
            cpu_time_sec[: e_diff_threshold_index + 1 + 1]
        )
        total_loops_wall_time_sec = np.sum(
            wall_time_sec[: e_diff_threshold_index + 1 + 1]
        )

        # print(e_diff_threshold_index)
        # print(bond_dimensions)
        # print(coarse_bond_dimension)
        # coarse_bond_dimension = bond_dimensions[np.where(energy_change < energy_change_threshold_hartrees)[0][0]]
        id_list.append(csv_file[:-4])
        coarse_bond_dimension_list.append(coarse_bond_dimension)
        total_loops_cpu_time_sec_list.append(total_loops_cpu_time_sec)
        total_loops_wall_time_sec_list.append(total_loops_wall_time_sec)

    return (
        id_list,
        coarse_bond_dimension_list,
        total_loops_cpu_time_sec_list,
        total_loops_wall_time_sec_list,
    )
