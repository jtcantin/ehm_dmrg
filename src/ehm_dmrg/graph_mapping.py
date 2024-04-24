import time

import networkx as nx
import numpy as np

# import np.linalg
import pandas as pd


def resistance_matrix(G):
    """
    By Mohsen Bagherimehrab
    """
    n = G.number_of_nodes()
    L = nx.laplacian_matrix(G)
    unitMtx = np.ones((n, n))

    Gamma = (
        L + unitMtx / n
    )  # .getA()  # matrix.getA() returns self as an ndarray object.
    invGamma = np.linalg.pinv(Gamma)  # pinv is pseudo inverse

    return np.add.outer(invGamma.diagonal(), invGamma.diagonal()) - 2 * invGamma


def power_spectral_entropy(x):
    """
    By Mohsen Bagherimehrab
    """
    ps = np.abs(np.fft.fft(x)) ** 2

    # power spectrum density
    psd = (ps + 1e-16) / (sum(ps) + 1e-16)

    # Compute entropy
    ent = 0
    for i in psd:
        ent -= i * np.log2(i)
    return ent


def get_market_graph(corrMtx, corrThreshold):
    """
    By Mohsen Bagherimehrab
    """

    # replace values less than threshold by zero
    corrMtx[abs(corrMtx) < corrThreshold] = 0
    G = nx.from_numpy_matrix(corrMtx)

    return G


def get_tbt_graph_adjacency_matrix_abs_element(tbt_tensor, threshold=1e-8):
    """
    Get a mapping fo the two body tensor to a graph,
    where the nodes are the (spin) orbitals
    and the edges are the sum of the absolute values of the elements connecting each site.
    By Joshua T. Cantin
    """
    # replace values less than threshold by zero
    tbt_tensor[abs(tbt_tensor) < threshold] = 0
    tbt_tensor_indices = np.transpose(np.nonzero(tbt_tensor))

    adjacency_matrix = np.zeros((tbt_tensor.shape[0], tbt_tensor.shape[0]))

    for iiter, jiter, kiter, liter in tbt_tensor_indices:
        element = tbt_tensor[iiter, jiter, kiter, liter]
        # Use abs to get sense of connection strength
        abs_element = abs(element)
        adjacency_matrix[iiter, jiter] += abs_element
        adjacency_matrix[jiter, kiter] += abs_element
        adjacency_matrix[kiter, liter] += abs_element
        adjacency_matrix[liter, iiter] += abs_element

    return adjacency_matrix


def get_feature_vector(tbt_tensor, threshold=1e-8) -> dict:
    """
    By Mohsen Bagherimehrab for discrete portfolio optimization
    Modified by Joshua T. Cantin for chemistry tbt tensor
    """

    matrix_representation = get_tbt_graph_adjacency_matrix_abs_element(
        tbt_tensor, threshold
    )
    G = nx.from_numpy_matrix(matrix_representation)

    algebraic_connectivity = nx.algebraic_connectivity(G)
    # diam = nx.diameter(G)
    # rad = nx.radius(G)
    transitivity = nx.transitivity(G)

    ############
    # new features
    edgeCount = G.number_of_edges()
    tbt_mean = np.mean(tbt_tensor)
    tbt_std = np.std(tbt_tensor)
    tbt_max = np.max(tbt_tensor)
    tbt_min = np.min(tbt_tensor)
    tbt_coeff_variation = tbt_std.real / tbt_mean.real
    # tbt_ent = power_spectral_entropy(tbt_tensor)
    tbt_inv_coeff_variation = tbt_mean.real / tbt_std.real

    ############

    start = time.time()  #
    specA = nx.adjacency_spectrum(G)
    stop = time.time()  #
    specA_calc_time = stop - start  #
    start = time.time()  #
    specA_min = np.min(specA)
    specA_Q2 = np.median(specA)
    specA_max = np.max(specA)
    specA_mean = np.mean(specA)
    specA_std = np.std(specA)
    specA_coeff_variation = specA_std.real / specA_mean.real  #
    specA_power_spectral_entropy = power_spectral_entropy(specA)
    stop = time.time()  #
    specA_stat_calc_time = stop - start  #

    start = time.time()  #
    specL = nx.laplacian_spectrum(G)
    stop = time.time()  #
    specL_calc_time = stop - start  #
    start = time.time()  #
    specL_min = np.min(specL)
    specL_Q2 = np.median(specL)
    specL_max = np.max(specL)
    specL_mean = np.mean(specL)
    specL_std = np.std(specL)
    specL_coeff_variation = specL_std.real / specL_mean.real  #
    specL_power_spectral_entropy = power_spectral_entropy(specL)
    stop = time.time()  #
    specL_stat_calc_time = stop - start  #

    start = time.time()  #
    R = resistance_matrix(G)
    specR = np.linalg.eigvals(R)
    stop = time.time()  #
    specR_calc_time = stop - start  #
    start = time.time()  #
    specR_min = np.min(specR)
    specR_Q2 = np.median(specR)
    specR_max = np.max(specR)
    specR_mean = np.mean(specR)
    specR_std = np.std(specR)
    specR_coeff_variation = specR_std.real / specR_mean.real  #
    specR_power_spectral_entropy = power_spectral_entropy(specR)
    stop = time.time()  #
    specR_stat_calc_time = stop - start  #

    featureVec = {
        # General graph features
        "algebraic_connectivity": algebraic_connectivity,
        "transitivity": transitivity,
        "edgeCount": edgeCount,
        # tbt features
        "tbt_mean": tbt_mean.real,
        "tbt_std": tbt_std.real,
        "tbt_max": tbt_max.real,
        "tbt_min": tbt_min.real,
        "tbt_coeff_variation": tbt_coeff_variation,
        # "tbt_ent": tbt_ent,
        "tbt_inv_coeff_variation": tbt_inv_coeff_variation,
        # Adjacency matrix spectrum features
        "specA_min": specA_min.real,
        "specA_Q2": specA_Q2.real,
        "specA_max": specA_max.real,
        "specA_mean": specA_mean.real,
        "specA_std": specA_std.real,
        "specA_coeff_variation": specA_coeff_variation,
        "specA_power_spectral_entropy": specA_power_spectral_entropy,
        "specA_calc_time": specA_calc_time,
        "specA_stat_calc_time": specA_stat_calc_time,
        # Laplacian matrix spectrum features
        "specL_min": specL_min.real,
        "specL_Q2": specL_Q2.real,
        "specL_max": specL_max.real,
        "specL_mean": specL_mean.real,
        "specL_std": specL_std.real,
        "specL_coeff_variation": specL_coeff_variation,
        "specL_power_spectral_entropy": specL_power_spectral_entropy,
        "specL_calc_time": specL_calc_time,
        "specL_stat_calc_time": specL_stat_calc_time,
        # Resistance matrix spectrum features
        "specR_min": specR_min.real,
        "specR_Q2": specR_Q2.real,
        "specR_max": specR_max.real,
        "specR_mean": specR_mean.real,
        "specR_std": specR_std.real,
        "specR_coeff_variation": specR_coeff_variation,
        "specR_power_spectral_entropy": specR_power_spectral_entropy,
        "specR_calc_time": specR_calc_time,
        "specR_stat_calc_time": specR_stat_calc_time,
    }

    return featureVec


def get_graph_properties_from_matrix(
    matrix, feature_prefix="", calc_resistance_matrix_properties=True
) -> dict:
    """
    By Mohsen Bagherimehrab for discrete portfolio optimization
    Modified by Joshua T. Cantin for chemistry tbt tensor
    """

    # G = get_tbt_graph(tbt_tensor, threshold)
    start = time.time()  #
    G = nx.from_numpy_array(matrix)
    stop = time.time()  #
    graph_creation_time = stop - start  #

    edgeCount = G.number_of_edges()

    start = time.time()  #
    algebraic_connectivity = nx.algebraic_connectivity(G)
    stop = time.time()  #
    algebraic_connectivity_calc_time = stop - start  #

    # diam = nx.diameter(G)
    # rad = nx.radius(G)
    start = time.time()  #
    transitivity = nx.transitivity(G)
    stop = time.time()  #
    transitivity_calc_time = stop - start  #

    start = time.time()  #
    specA = nx.adjacency_spectrum(G)
    stop = time.time()  #
    specA_calc_time = stop - start  #
    start = time.time()  #
    specA_min = np.min(specA)
    specA_Q2 = np.median(specA)
    specA_max = np.max(specA)
    specA_mean = np.mean(specA)
    specA_std = np.std(specA)
    specA_coeff_variation = (specA_std.real + 1e-16) / (specA_mean.real + 1e-16)  #
    specA_power_spectral_entropy = power_spectral_entropy(specA)
    stop = time.time()  #
    specA_stat_calc_time = stop - start  #

    start = time.time()  #
    specL = nx.laplacian_spectrum(G)
    stop = time.time()  #
    specL_calc_time = stop - start  #
    start = time.time()  #
    specL_min = np.min(specL)
    specL_Q2 = np.median(specL)
    specL_max = np.max(specL)
    specL_mean = np.mean(specL)
    specL_std = np.std(specL)
    specL_coeff_variation = (specL_std.real + 1e-16) / (specL_mean.real + 1e-16)  #
    specL_power_spectral_entropy = power_spectral_entropy(specL)
    stop = time.time()  #
    specL_stat_calc_time = stop - start  #

    featureVec = {
        # General graph features
        f"{feature_prefix}algebraic_connectivity": algebraic_connectivity,
        f"{feature_prefix}transitivity": transitivity,
        f"{feature_prefix}edgeCount": edgeCount,
        f"{feature_prefix}graph_creation_time": graph_creation_time,
        f"{feature_prefix}algebraic_connectivity_calc_time": algebraic_connectivity_calc_time,
        f"{feature_prefix}transitivity_calc_time": transitivity_calc_time,
        # Adjacency matrix spectrum features
        f"{feature_prefix}specA_min": specA_min.real,
        f"{feature_prefix}specA_Q2": specA_Q2.real,
        f"{feature_prefix}specA_max": specA_max.real,
        f"{feature_prefix}specA_mean": specA_mean.real,
        f"{feature_prefix}specA_std": specA_std.real,
        f"{feature_prefix}specA_coeff_variation": specA_coeff_variation,
        f"{feature_prefix}specA_power_spectral_entropy": specA_power_spectral_entropy,
        f"{feature_prefix}specA_calc_time": specA_calc_time,
        f"{feature_prefix}specA_stat_calc_time": specA_stat_calc_time,
        # Laplacian matrix spectrum features
        f"{feature_prefix}specL_min": specL_min.real,
        f"{feature_prefix}specL_Q2": specL_Q2.real,
        f"{feature_prefix}specL_max": specL_max.real,
        f"{feature_prefix}specL_mean": specL_mean.real,
        f"{feature_prefix}specL_std": specL_std.real,
        f"{feature_prefix}specL_coeff_variation": specL_coeff_variation,
        f"{feature_prefix}specL_power_spectral_entropy": specL_power_spectral_entropy,
        f"{feature_prefix}specL_calc_time": specL_calc_time,
        f"{feature_prefix}specL_stat_calc_time": specL_stat_calc_time,
    }

    if calc_resistance_matrix_properties:
        start = time.time()  #
        R = resistance_matrix(G)
        specR = np.linalg.eigvals(R)
        stop = time.time()  #
        specR_calc_time = stop - start  #
        start = time.time()  #
        specR_min = np.min(specR)
        specR_Q2 = np.median(specR)
        specR_max = np.max(specR)
        specR_mean = np.mean(specR)
        specR_std = np.std(specR)
        specR_coeff_variation = (specR_std.real + 1e-16) / (specR_mean.real + 1e-16)  #
        specR_power_spectral_entropy = power_spectral_entropy(specR)
        stop = time.time()  #
        specR_stat_calc_time = stop - start  #

        res_mat_prop = {
            # Resistance matrix spectrum features
            f"{feature_prefix}specR_min": specR_min.real,
            f"{feature_prefix}specR_Q2": specR_Q2.real,
            f"{feature_prefix}specR_max": specR_max.real,
            f"{feature_prefix}specR_mean": specR_mean.real,
            f"{feature_prefix}specR_std": specR_std.real,
            f"{feature_prefix}specR_coeff_variation": specR_coeff_variation,
            f"{feature_prefix}specR_power_spectral_entropy": specR_power_spectral_entropy,
            f"{feature_prefix}specR_calc_time": specR_calc_time,
            f"{feature_prefix}specR_stat_calc_time": specR_stat_calc_time,
        }

        featureVec.update(res_mat_prop)

    return featureVec


def generate_ml_data(data_dicts, threshold=1e-8):
    """
    By Mohsen Bagherimehrab for discrete portfolio optimization
    Modified by Joshua T. Cantin for chemistry tbt tensor
    """
    mlDataDics = []

    n = len(data_dicts)

    for j in range(n):
        local_dict = data_dicts[j]
        tbt_tensor = local_dict["tbt_tensor"]
        fvec = get_feature_vector(tbt_tensor=tbt_tensor, threshold=threshold)

        for i in local_dict.keys():
            if i != "tbt_tensor" and i != "obt_tensor":
                fvec[i] = local_dict[i]

        mlDataDics.append(fvec)

    mlDataDf = pd.DataFrame.from_dict(mlDataDics)
    return mlDataDf
