""" This code calculates DMRG features for a given electronic Hamiltonian."""

import time

import numpy as np

# import pyscf
import pyscf.fci.direct_spin1
import scipy as sp
import scipy.linalg as spla
import scipy.special as sps

import ehm_dmrg.graph_mapping as gm


def generate_coulomb_integral_matrix(two_body_tensor):
    """Generates the Coulomb integral matrix from the two-body tensor,
    using the definition from https://doi.org/10.1063/1.1824891
    J_ij = V_ijji = G_iijj, where H = 0.5*G_ijkl a†_i a_j a†_k a_l ."""
    # num_sites = two_body_tensor.shape[0]
    # coulomb_matrix_old = np.zeros(
    #     (
    #         num_sites,
    #         num_sites,
    #     )
    # )
    # for i in range(num_sites):
    #     for j in range(num_sites):
    #         coulomb_matrix_old[i, j] = two_body_tensor[i, i, j, j]

    # Do the above in a more efficient way
    coulomb_matrix = np.einsum("iijj->ij", two_body_tensor)

    # print(np.allclose(coulomb_matrix, coulomb_matrix_old))
    # assert np.allclose(coulomb_matrix, coulomb_matrix_old)

    return coulomb_matrix


def generate_exchange_integral_matrix(two_body_tensor):
    """Generates the exchange integral matrix from the two-body tensor,
    using the definition from https://doi.org/10.1063/1.1824891
    K_ij = V_ijij = g_ijji, where H = 0.5*G_ijkl a†_i a_j a†_k a_l ."""
    # num_sites = two_body_tensor.shape[0]
    # exchange_matrix_old = np.zeros(
    #     (
    #         num_sites,
    #         num_sites,
    #     )
    # )
    # for i in range(num_sites):
    #     for j in range(num_sites):
    #         exchange_matrix_old[i, j] = two_body_tensor[i, j, j, i]

    # Do the above in a more efficient way
    exchange_matrix = np.einsum("ijji->ij", two_body_tensor)
    # print(np.allclose(exchange_matrix, exchange_matrix_old))
    # assert np.allclose(exchange_matrix, exchange_matrix_old)
    return exchange_matrix


def generate_mean_field_matrix(coulomb_matrix, exchange_matrix):
    """Generates the mean field matrix from the Coulomb and exchange integral matrices,
    using the definition from https://doi.org/10.1063/1.1824891
    M_ij = 2J_ij - K_ij."""
    mean_field_matrix = 2 * coulomb_matrix - exchange_matrix
    return mean_field_matrix


def generate_tbt_matrices(two_body_tensor, verbosity=2):
    """Generates the Coulomb, exchange, and mean field matrices from the two-body tensor."""
    start_time = time.process_time_ns()
    coulomb_matrix = generate_coulomb_integral_matrix(two_body_tensor)
    end_time = time.process_time_ns()
    if verbosity >= 2:
        print("Coulomb matrix calculation time:", (end_time - start_time) / 1e9)

    start_time = time.process_time_ns()
    exchange_matrix = generate_exchange_integral_matrix(two_body_tensor)
    end_time = time.process_time_ns()
    if verbosity >= 2:
        print("Exchange matrix calculation time:", (end_time - start_time) / 1e9)

    start_time = time.process_time_ns()
    mean_field_matrix = generate_mean_field_matrix(coulomb_matrix, exchange_matrix)
    end_time = time.process_time_ns()
    if verbosity >= 2:
        print("Mean field matrix calculation time:", (end_time - start_time) / 1e9)

    return coulomb_matrix, exchange_matrix, mean_field_matrix


def get_dmrg_features(
    one_body_tensor=None,
    two_body_tensor=None,
    num_electrons=None,
    # spin_orbitals_bool=True,
    # tbt_one_half_convention=True,
    calc_resistance_matrix_properties=True,
    verbosity=2,
):
    # The tensors are assumed to be defined in terms of the following Hamiltonian:
    # H = E_0 + h_ij a†_i a_j + 0.5*g_ijkl a†_i a†_k a_l a_j
    # with no permutation symmetry compression for two_body_tensor. ijkl are spatial orbitals.

    # if not tbt_one_half_convention:
    #     # By convention, 0.5*g_pqrs are the coefficients for the two-electron terms
    #     # If the two-body tensor is not in this convention, we need to convert it
    #     local_two_body_tensor = 2 * two_body_tensor
    # else:
    #     local_two_body_tensor = two_body_tensor

    feature_dict = {}
    num_orbitals = one_body_tensor.shape[0]
    # if spin_orbitals_bool:
    #     num_spin_orbitals = 2 * num_sites
    # else:
    #     num_spin_orbitals = num_sites
    num_spin_orbitals = 2 * num_orbitals
    # Number of possible states, given num_electrons and num_spin_orbitals
    total_num_states = sps.comb(num_spin_orbitals, num_electrons, exact=False)
    feature_dict["total_num_states"] = total_num_states
    feature_dict["log10_hilbert_space_size"] = np.log10(total_num_states)

    # Combine one and two body tensors into a single tensor
    # absorb_h1e assumes the Hamiltonian is H = E_0 + h_ij a†_i a_j + 0.5*g_ijkl a†_i a†_k a_l a_j
    # and returns the tbt assuming H = 0.5*G_ijkl a†_i a_j a†_k a_l
    # Note that this transformation is not general and is only valid for the specified number of
    # electrons.
    if verbosity >= 1:
        print("one_body_tensor.shape", one_body_tensor.shape)
        print("two_body_tensor.shape", two_body_tensor.shape)
    one_and_two_body_tensor = pyscf.fci.direct_spin1.absorb_h1e(
        h1e=one_body_tensor,
        eri=two_body_tensor,
        norb=num_orbitals,
        nelec=num_electrons,
        fac=1,
    )
    one_and_two_body_tensor = pyscf.ao2mo.restore(
        1, one_and_two_body_tensor, num_orbitals
    )
    if verbosity >= 1:
        print("one_body_tensor.shape", one_body_tensor.shape)
        print("two_body_tensor.shape", two_body_tensor.shape)
        print("one_and_two_body_tensor.shape", one_and_two_body_tensor.shape)
        # print("one_body_tensor", one_body_tensor)
        # print("two_body_tensor", two_body_tensor)
        # print("one_and_two_body_tensor", one_and_two_body_tensor)
        print("num_orbitals", num_orbitals)
        print("num_electrons", num_electrons)
        # local_two_body_tensor *= 0.5

    # Various bandwiths and other matrix properties for the Coulomb, exchange, and mean field matrices
    tbt_matrices_calc_start = time.process_time_ns()
    coulomb_matrix, exchange_matrix, mean_field_matrix = generate_tbt_matrices(
        one_and_two_body_tensor, verbosity=verbosity
    )
    tbt_matrices_calc_end = time.process_time_ns()
    feature_dict["tbt_matrices_calc_time_s"] = (
        tbt_matrices_calc_end - tbt_matrices_calc_start
    ) / 1e9
    if verbosity >= 1:
        print("Generated tbt matrices, time:", feature_dict["tbt_matrices_calc_time_s"])

        print("coulomb_matrix.shape", coulomb_matrix.shape)
        print("exchange_matrix.shape", exchange_matrix.shape)
        print("mean_field_matrix.shape", mean_field_matrix.shape)
        # print("coulomb_matrix", coulomb_matrix)
        # print("exchange_matrix", exchange_matrix)
        # print("mean_field_matrix", mean_field_matrix)

    bandwidth_start = time.process_time_ns()
    coulomb_matrix_bandwidths = spla.bandwidth(coulomb_matrix)
    exchange_matrix_bandwidths = spla.bandwidth(exchange_matrix)
    mean_field_matrix_bandwidths = spla.bandwidth(mean_field_matrix)

    bandwidth_end = time.process_time_ns()
    feature_dict["bandwidth_calc_time_s"] = (bandwidth_end - bandwidth_start) / 1e9
    if verbosity >= 1:
        print("Bandwidth calculation time:", feature_dict["bandwidth_calc_time_s"])

    feature_dict["coulomb_matrix_bandwidth_upper"] = coulomb_matrix_bandwidths[1]
    feature_dict["coulomb_matrix_bandwidth_lower"] = coulomb_matrix_bandwidths[0]
    feature_dict["exchange_matrix_bandwidth_upper"] = exchange_matrix_bandwidths[1]
    feature_dict["exchange_matrix_bandwidth_lower"] = exchange_matrix_bandwidths[0]
    feature_dict["mean_field_matrix_bandwidth_upper"] = mean_field_matrix_bandwidths[1]
    feature_dict["mean_field_matrix_bandwidth_lower"] = mean_field_matrix_bandwidths[0]

    coulomb_start = time.process_time_ns()
    new_feature_dict = gm.get_graph_properties_from_matrix(
        matrix=coulomb_matrix,
        feature_prefix="coulomb_matrix_",
        calc_resistance_matrix_properties=calc_resistance_matrix_properties,
    )
    coulomb_end = time.process_time_ns()
    feature_dict["coulomb_calc_time_s"] = (coulomb_end - coulomb_start) / 1e9
    feature_dict.update(new_feature_dict)
    if verbosity >= 1:
        print("Coulomb matrix features calculated")
        print("Coulomb matrix calculation time:", feature_dict["coulomb_calc_time_s"])

    exchange_start = time.process_time_ns()
    new_feature_dict = gm.get_graph_properties_from_matrix(
        matrix=exchange_matrix,
        feature_prefix="exchange_matrix_",
        calc_resistance_matrix_properties=calc_resistance_matrix_properties,
    )
    exchange_end = time.process_time_ns()
    feature_dict["exchange_calc_time_s"] = (exchange_end - exchange_start) / 1e9
    feature_dict.update(new_feature_dict)
    if verbosity >= 1:
        print("Exchange matrix features calculated")
        print("Exchange matrix calculation time:", feature_dict["exchange_calc_time_s"])

    mean_field_start = time.process_time_ns()
    new_feature_dict = gm.get_graph_properties_from_matrix(
        matrix=mean_field_matrix,
        feature_prefix="mean_field_matrix_",
        calc_resistance_matrix_properties=calc_resistance_matrix_properties,
    )
    mean_field_end = time.process_time_ns()
    feature_dict["mean_field_calc_time_s"] = (mean_field_end - mean_field_start) / 1e9
    feature_dict.update(new_feature_dict)
    if verbosity >= 1:
        print("Mean field matrix features calculated")
        print(
            "Mean field matrix calculation time:",
            feature_dict["mean_field_calc_time_s"],
        )

    # Various properties of the absolute version of the two-body tensor to graph mapping
    tbt_graph_mapping_start = time.process_time_ns()
    tbt_abs_graph_mapping = gm.get_tbt_graph_adjacency_matrix_abs_element(
        one_and_two_body_tensor, threshold=1e-8
    )
    tbt_graph_mapping_end = time.process_time_ns()
    feature_dict["tbt_graph_mapping_calc_time_s"] = (
        tbt_graph_mapping_end - tbt_graph_mapping_start
    ) / 1e9
    if verbosity >= 1:
        print(
            "tbt_graph_mapping_calc_time_s",
            feature_dict["tbt_graph_mapping_calc_time_s"],
        )

    tbt_abs_graph_mapping_calc_start = time.process_time_ns()
    new_feature_dict = gm.get_graph_properties_from_matrix(
        matrix=tbt_abs_graph_mapping,
        feature_prefix="tbt_abs_graph_mapping_",
        calc_resistance_matrix_properties=calc_resistance_matrix_properties,
    )
    tbt_abs_graph_mapping_calc_end = time.process_time_ns()
    feature_dict["tbt_abs_graph_mapping_calc_time_s"] = (
        tbt_abs_graph_mapping_calc_end - tbt_abs_graph_mapping_calc_start
    ) / 1e9
    feature_dict.update(new_feature_dict)
    if verbosity >= 1:
        print("tbt_abs_graph_mapping features calculated")
        print(
            "tbt_abs_graph_mapping_calc_time_s",
            feature_dict["tbt_abs_graph_mapping_calc_time_s"],
        )

    # Various properties of one-body tensor
    one_body_tensor_start = time.process_time_ns()
    # print(one_body_tensor)
    # new_feature_dict = gm.get_graph_properties_from_matrix(
    #     matrix=one_body_tensor,
    #     feature_prefix="one_body_tensor_",
    #     # calc_resistance_matrix_properties=calc_resistance_matrix_properties,
    #     calc_resistance_matrix_properties=False,
    # )

    # feature_dict.update(new_feature_dict)

    obt_bandwidths = spla.bandwidth(one_body_tensor)
    one_body_tensor_end = time.process_time_ns()
    feature_dict["one_body_tensor_calc_time_s"] = (
        one_body_tensor_end - one_body_tensor_start
    ) / 1e9
    feature_dict["one_body_tensor_bandwidth_upper"] = obt_bandwidths[1]
    feature_dict["one_body_tensor_bandwidth_lower"] = obt_bandwidths[0]
    if verbosity >= 1:
        print("one_body_tensor features calculated")
        print(
            "one_body_tensor_calc_time_s", feature_dict["one_body_tensor_calc_time_s"]
        )

    # Various properties of the two-body tensor
    tbt_density_hopping_start = time.process_time_ns()
    new_feature_dict = tbt_density_hopping_features(one_and_two_body_tensor)
    feature_dict.update(new_feature_dict)
    tbt_density_hopping_end = time.process_time_ns()
    feature_dict["tbt_density_hopping_calc_time_s"] = (
        tbt_density_hopping_end - tbt_density_hopping_start
    ) / 1e9
    if verbosity >= 1:
        print("tbt_density_hopping features calculated")
        print(
            "tbt_density_hopping_calc_time_s",
            feature_dict["tbt_density_hopping_calc_time_s"],
        )

    tbt_start = time.process_time_ns()
    new_feature_dict = tbt_features(one_and_two_body_tensor)
    feature_dict.update(new_feature_dict)
    tbt_end = time.process_time_ns()
    feature_dict["tbt_calc_time_s"] = (tbt_end - tbt_start) / 1e9
    if verbosity >= 1:
        print("tbt features calculated")
        print("tbt_calc_time_s", feature_dict["tbt_calc_time_s"])

    # Hubbard distance
    hubbard_distance_start = time.process_time_ns()
    l2_norm_diff_obt, l2_norm_diff_tbt, l2_norm_diff_total = calc_hubbard_distance(
        one_body_tensor, two_body_tensor
    )
    feature_dict["l2_norm_diff_obt"] = l2_norm_diff_obt
    feature_dict["l2_norm_diff_tbt"] = l2_norm_diff_tbt
    feature_dict["l2_norm_diff_total"] = l2_norm_diff_total
    hubbard_distance_end = time.process_time_ns()
    feature_dict["hubbard_distance_calc_time_s"] = (
        hubbard_distance_end - hubbard_distance_start
    ) / 1e9
    if verbosity >= 1:
        print("Hubbard distance features calculated")
        print(
            "hubbard_distance_calc_time_s",
            feature_dict["hubbard_distance_calc_time_s"],
        )

    # Diagonal dominance
    diagonal_dominance_start = time.process_time_ns()
    (
        obt_l2_norm,
        obt_diag_l2_norm,
        obt_ratio,
        tbt_l2_norm,
        tbt_diag_l2_norm,
        tbt_ratio,
        diagonal_dominance,
    ) = calc_diagonal_dominance(one_body_tensor, two_body_tensor)
    feature_dict["obt_l2_norm"] = obt_l2_norm
    feature_dict["obt_diag_l2_norm"] = obt_diag_l2_norm
    feature_dict["obt_diag_orig_l2_ratio"] = obt_ratio
    feature_dict["tbt_l2_norm"] = tbt_l2_norm
    feature_dict["tbt_diag_l2_norm"] = tbt_diag_l2_norm
    feature_dict["tbt_diag_orig_l2_ratio"] = tbt_ratio
    feature_dict["diagonal_dominance"] = diagonal_dominance
    diagonal_dominance_end = time.process_time_ns()
    feature_dict["diagonal_dominance_calc_time_s"] = (
        diagonal_dominance_end - diagonal_dominance_start
    ) / 1e9
    if verbosity >= 1:
        print("Diagonal dominance features calculated")
        print(
            "diagonal_dominance_calc_time_s",
            feature_dict["diagonal_dominance_calc_time_s"],
        )

    return feature_dict


def tbt_features(tbt_tensor):
    ############
    # new features

    tbt_mean = np.mean(tbt_tensor)
    tbt_std = np.std(tbt_tensor)
    tbt_max = np.max(tbt_tensor)
    tbt_min = np.min(tbt_tensor)
    tbt_coeff_variation = (tbt_std.real + 1e-16) / (tbt_mean.real + 1e-16)
    # tbt_ent = power_spectral_entropy(tbt_tensor)
    tbt_inv_coeff_variation = (tbt_mean.real + 1e-16) / (tbt_std.real + 1e-16)

    feature_dict = {
        # tbt features
        "tbt_mean": tbt_mean.real,
        "tbt_std": tbt_std.real,
        "tbt_max": tbt_max.real,
        "tbt_min": tbt_min.real,
        "tbt_coeff_variation": tbt_coeff_variation,
        # "tbt_ent": tbt_ent,
        "tbt_inv_coeff_variation": tbt_inv_coeff_variation,
    }
    return feature_dict


def tbt_density_hopping_features(tbt_tensor):
    ############
    # new features

    density_hopping_tensor = np.einsum("iikl->ikl", tbt_tensor)

    density_hopping_tensor_mean = np.mean(density_hopping_tensor)
    density_hopping_tensor_std = np.std(density_hopping_tensor)
    density_hopping_tensor_max = np.max(density_hopping_tensor)
    density_hopping_tensor_min = np.min(density_hopping_tensor)
    density_hopping_tensor_coeff_variation = (
        density_hopping_tensor_std.real + 1e-16
    ) / (density_hopping_tensor_mean.real + 1e-16)
    # density_hopping_tensor_ent = power_spectral_entropy(density_hopping_tensor)
    density_hopping_tensor_inv_coeff_variation = (density_hopping_tensor_mean.real) / (
        density_hopping_tensor_std.real + 1e-16
    )

    density_hopping_tensor_sum_abs = np.sum(np.abs(density_hopping_tensor))
    feature_dict = {
        # density_hopping_tensor features
        "density_hopping_tensor_mean": density_hopping_tensor_mean.real,
        "density_hopping_tensor_std": density_hopping_tensor_std.real,
        "density_hopping_tensor_max": density_hopping_tensor_max.real,
        "density_hopping_tensor_min": density_hopping_tensor_min.real,
        "density_hopping_tensor_coeff_variation": density_hopping_tensor_coeff_variation,
        # "density_hopping_tensor_ent": density_hopping_tensor_ent,
        "density_hopping_tensor_inv_coeff_variation": density_hopping_tensor_inv_coeff_variation,
        "density_hopping_tensor_sum_abs": density_hopping_tensor_sum_abs,
    }
    return feature_dict


# def two_site_coupling_strength(site_1_element, site_2_element, coupling_element):
#     """Calculates the coupling strength between two sites"""
#     if np.allclose(site_1_element, site_2_element):
#         diagonalized_value = stuff  # Get the from sympy
#         return diagonalized_value
#     else:
#         return (1e-16 + np.abs(coupling_element)) / (
#             1e-16 + np.abs(site_1_element - site_2_element)
#         )


def calc_diagonal_dominance(ob_tensor, tb_tensor):
    # ob_tensor is a 2D array
    # tb_tensor is a 4D array
    num_orbitals = ob_tensor.shape[0]
    obt_diag = np.diagonal(ob_tensor)
    obt_diag_l2_norm = np.linalg.norm(obt_diag, ord=2)
    obt_l2_norm = np.linalg.norm(ob_tensor, ord=2)

    # tbt_diag is the following number operator terms
    # g_pqrs:
    # p=q, r=s (g_pprr)
    # p=r, q=s (g_prpr)
    # p=s, q=r (g_prrp)
    # Create scipy sparse tensor
    # tbt_diag =spsparse.coo_matrix((num_orbitals,num_orbitals,num_orbitals,num_orbitals))
    tbt_diag_l2_norm = 0
    for p in range(num_orbitals):
        for r in range(num_orbitals):
            tbt_diag_l2_norm += tb_tensor[p, p, r, r] ** 2
            tbt_diag_l2_norm += tb_tensor[p, r, p, r] ** 2
            tbt_diag_l2_norm += tb_tensor[p, r, r, p] ** 2

            # tbt_diag[p,p,r,r] = tb_tensor[p,p,r,r]
            # tbt_diag[p,r,p,r] = tb_tensor[p,r,p,r]
            # tbt_diag[p,r,r,p] = tb_tensor[p,r,r,p]
    tbt_diag_l2_norm = 0.5 * np.sqrt(tbt_diag_l2_norm)

    tbt_l2_norm = np.linalg.norm(tb_tensor.flat, ord=2)

    obt_ratio = obt_diag_l2_norm / obt_l2_norm
    tbt_ratio = tbt_diag_l2_norm / tbt_l2_norm

    diagonal_dominance = obt_ratio + tbt_ratio
    return (
        obt_l2_norm,
        obt_diag_l2_norm,
        obt_ratio,
        tbt_l2_norm,
        tbt_diag_l2_norm,
        tbt_ratio,
        diagonal_dominance,
    )


def calc_hubbard_distance(ob_tensor, tb_tensor):
    # ob_tensor is a 2D array
    # tb_tensor is a 4D array
    num_orbitals = ob_tensor.shape[0]
    obt_diag = np.diagonal(ob_tensor)
    chem_potential_plus_U = -1 * np.mean(obt_diag)
    first_off_diag = np.diag(ob_tensor, k=1)
    hopping_parameter = -1 * np.mean(first_off_diag)

    on_site_interaction = np.einsum("pppp->", tb_tensor)
    on_site_interaction /= num_orbitals
    # on_site_interaction *= 0.5
    # chem_potential += on_site_interaction/2

    on_site_energy_matrix = -chem_potential_plus_U * np.eye(num_orbitals)
    hopping_matrix = -1 * hopping_parameter * np.eye(num_orbitals, k=1)
    hopping_matrix += -1 * hopping_parameter * np.eye(num_orbitals, k=-1)
    hubbard_matrix = on_site_energy_matrix + hopping_matrix
    l2_norm_diff_obt = np.linalg.norm(ob_tensor - hubbard_matrix, ord=2)

    tb_tensor_copy = 0.5 * tb_tensor.copy()

    # Just work with nn term, dropping n in order to get half-filling
    for p in range(num_orbitals):
        tb_tensor_copy[p, p, p, p] -= 0.5 * on_site_interaction

    l2_norm_diff_tbt = np.linalg.norm(tb_tensor_copy.flat, ord=2)

    l2_norm_diff_total = l2_norm_diff_obt + l2_norm_diff_tbt
    return l2_norm_diff_obt, l2_norm_diff_tbt, l2_norm_diff_total
