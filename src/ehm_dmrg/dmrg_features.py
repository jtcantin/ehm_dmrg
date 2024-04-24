""" This code calculates DMRG features for a given electronic Hamiltonian."""

import time

import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.special as sps

import graph_mapping as gm


def generate_coulomb_integral_matrix(two_body_tensor):
    """Generates the Coulomb integral matrix from the two-body tensor,
    using the definition from https://doi.org/10.1063/1.1824891
    J_ij = V_ijji = g_iijj, where g_pqrs is the two-body tensor."""
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
    K_ij = V_ijij = g_ijji, where g_pqrs is the two-body tensor."""
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


def generate_tbt_matrices(two_body_tensor):
    """Generates the Coulomb, exchange, and mean field matrices from the two-body tensor."""
    coulomb_matrix = generate_coulomb_integral_matrix(two_body_tensor)
    exchange_matrix = generate_exchange_integral_matrix(two_body_tensor)
    mean_field_matrix = generate_mean_field_matrix(coulomb_matrix, exchange_matrix)
    return coulomb_matrix, exchange_matrix, mean_field_matrix


def get_dmrg_features(
    one_body_tensor=None,
    two_body_tensor=None,
    num_electrons=None,
    spin_orbitals_bool=True,
    tbt_one_half_convention=False,
    calc_resistance_matrix_properties=True,
):

    if not tbt_one_half_convention:
        # By convention, 0.5*g_pqrs are the coefficients for the two-electron terms
        # If the two-body tensor is not in this convention, we need to convert it
        local_two_body_tensor = 2 * two_body_tensor
    else:
        local_two_body_tensor = two_body_tensor

    feature_dict = {}
    num_sites = one_body_tensor.shape[0]
    if spin_orbitals_bool:
        num_spin_orbitals = 2 * num_sites
    else:
        num_spin_orbitals = num_sites

    # Number of possible states, given num_electrons and num_spin_orbitals
    total_num_states = sps.comb(num_spin_orbitals, num_electrons, exact=True)
    feature_dict["total_num_states"] = total_num_states

    # Various bandwiths and other matrix properties for the Coulomb, exchange, and mean field matrices
    tbt_matrices_calc_start = time.time()
    coulomb_matrix, exchange_matrix, mean_field_matrix = generate_tbt_matrices(
        local_two_body_tensor
    )
    tbt_matrices_calc_end = time.time()
    feature_dict["tbt_matrices_calc_time"] = (
        tbt_matrices_calc_end - tbt_matrices_calc_start
    )

    bandwidth_start = time.time()
    coulomb_matrix_bandwidths = spla.bandwidth(coulomb_matrix)
    exchange_matrix_bandwidths = spla.bandwidth(exchange_matrix)
    mean_field_matrix_bandwidths = spla.bandwidth(mean_field_matrix)

    bandwidth_end = time.time()
    feature_dict["bandwidth_calc_time"] = bandwidth_end - bandwidth_start
    # print("Bandwidth calculation time:", bandwidth_end - bandwidth_start)

    feature_dict["coulomb_matrix_bandwidth_upper"] = coulomb_matrix_bandwidths[1]
    feature_dict["coulomb_matrix_bandwidth_lower"] = coulomb_matrix_bandwidths[0]
    feature_dict["exchange_matrix_bandwidth_upper"] = exchange_matrix_bandwidths[1]
    feature_dict["exchange_matrix_bandwidth_lower"] = exchange_matrix_bandwidths[0]
    feature_dict["mean_field_matrix_bandwidth_upper"] = mean_field_matrix_bandwidths[1]
    feature_dict["mean_field_matrix_bandwidth_lower"] = mean_field_matrix_bandwidths[0]

    coulomb_start = time.time()
    new_feature_dict = gm.get_graph_properties_from_matrix(
        matrix=coulomb_matrix,
        feature_prefix="coulomb_matrix_",
        calc_resistance_matrix_properties=calc_resistance_matrix_properties,
    )
    coulomb_end = time.time()
    feature_dict["coulomb_calc_time"] = coulomb_end - coulomb_start
    feature_dict.update(new_feature_dict)

    exchange_start = time.time()
    new_feature_dict = gm.get_graph_properties_from_matrix(
        matrix=exchange_matrix,
        feature_prefix="exchange_matrix_",
        calc_resistance_matrix_properties=calc_resistance_matrix_properties,
    )
    exchange_end = time.time()
    feature_dict["exchange_calc_time"] = exchange_end - exchange_start
    feature_dict.update(new_feature_dict)

    # mean_field_start = time.time()
    # new_feature_dict = gm.get_graph_properties_from_matrix(
    #     matrix=mean_field_matrix,
    #     feature_prefix="mean_field_matrix_",
    #     calc_resistance_matrix_properties=calc_resistance_matrix_properties,
    # )
    # mean_field_end = time.time()
    # feature_dict["mean_field_calc_time"] = mean_field_end - mean_field_start
    # feature_dict.update(new_feature_dict)

    # Various properties of the absolute version of the two-body tensor to graph mapping
    tbt_graph_mapping_start = time.time()
    tbt_abs_graph_mapping = gm.get_tbt_graph_adjacency_matrix_abs_element(
        local_two_body_tensor, threshold=1e-8
    )
    tbt_graph_mapping_end = time.time()
    feature_dict["tbt_graph_mapping_calc_time"] = (
        tbt_graph_mapping_end - tbt_graph_mapping_start
    )

    tbt_abs_graph_mapping_calc_start = time.time()
    new_feature_dict = gm.get_graph_properties_from_matrix(
        matrix=tbt_abs_graph_mapping,
        feature_prefix="tbt_abs_graph_mapping_",
        calc_resistance_matrix_properties=calc_resistance_matrix_properties,
    )
    tbt_abs_graph_mapping_calc_end = time.time()
    feature_dict["tbt_abs_graph_mapping_calc_time"] = (
        tbt_abs_graph_mapping_calc_end - tbt_abs_graph_mapping_calc_start
    )
    feature_dict.update(new_feature_dict)

    # Various properties of one-body tensor
    one_body_tensor_start = time.time()
    new_feature_dict = gm.get_graph_properties_from_matrix(
        matrix=one_body_tensor,
        feature_prefix="one_body_tensor_",
        calc_resistance_matrix_properties=calc_resistance_matrix_properties,
    )
    feature_dict.update(new_feature_dict)

    obt_bandwidths = spla.bandwidth(one_body_tensor)
    one_body_tensor_end = time.time()
    feature_dict["one_body_tensor_calc_time"] = (
        one_body_tensor_end - one_body_tensor_start
    )
    feature_dict["one_body_tensor_bandwidth_upper"] = obt_bandwidths[1]
    feature_dict["one_body_tensor_bandwidth_lower"] = obt_bandwidths[0]

    # Various properties of the two-body tensor
    tbt_density_hopping_start = time.time()
    new_feature_dict = tbt_density_hopping_features(local_two_body_tensor)
    feature_dict.update(new_feature_dict)
    tbt_density_hopping_end = time.time()
    feature_dict["tbt_density_hopping_calc_time"] = (
        tbt_density_hopping_end - tbt_density_hopping_start
    )

    tbt_start = time.time()
    new_feature_dict = tbt_features(local_two_body_tensor)
    feature_dict.update(new_feature_dict)
    tbt_end = time.time()
    feature_dict["tbt_calc_time"] = tbt_end - tbt_start

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
