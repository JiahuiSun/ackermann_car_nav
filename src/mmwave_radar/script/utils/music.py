import numpy as np
import numpy.linalg as LA


def cov_matrix(x):
    """ Calculates the spatial covariance matrix (Rxx) for a given set of input data (x=inputData). 
        Assumes rows denote Vrx axis.

    Args:
        x (ndarray): A 2D-Array with shape (rx, adc_samples) slice of the output of the 1D range fft

    Returns:
        Rxx (ndarray): A 2D-Array with shape (rx, rx)
    """
    
    if x.ndim > 2:
        raise ValueError("x has more than 2 dimensions.")

    # if x.shape[0] > x.shape[1]:
    #     warnings.warn("cov_matrix input should have Vrx as rows. Needs to be transposed", RuntimeWarning)
    #     x = x.T

    _, num_adc_samples = x.shape
    Rxx = x @ np.conjugate(x.T)
    Rxx = np.divide(Rxx, num_adc_samples)

    return Rxx

def _noise_subspace(covariance, num_sources):
    """helper function to get noise_subspace.
    """
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance matrix should be a 2D square matrix.")
    if num_sources >= covariance.shape[0]:
        raise ValueError("number of sources should be less than number of receivers.")
    _, v = LA.eigh(covariance) 
    
    return v[:, :-num_sources]

def aoa_music_1D(steering_vec, rx_chirps, num_sources):
    """Implmentation of 1D MUltiple SIgnal Classification (MUSIC) algorithm on ULA (Uniformed Linear Array). 
    
    Current implementation assumes covariance matrix is not rank deficient and ULA spacing is half of the wavelength.
    .. math::
        P_{} (\\theta) = \\frac{1}{a^{H}(\\theta) \mathbf{E}_\mathrm{n}\mathbf{E}_\mathrm{n}^H a(\\theta)}
    where :math:`E_{n}` is the noise subpace and :math:`a` is the steering vector.
    

    Args:
        steering_vec (~np.ndarray): steering vector with the shape of (FoV/angel_resolution, num_ant). 
         FoV/angel_resolution is usually 181. It is generated from gen_steering_vec() function.
        rx_chirps (~np.ndarray): Ouput of the 1D range FFT. The shape is (num_ant, num_chirps_per_frame).
        num_sources (int): Number of sources in the scene. Needs to be smaller than num_ant for ULA.
    
    Returns:
        (~np.ndarray): the spectrum of the MUSIC. Objects should be holes for the equation and thus sharp peaks.
    """
    num_antennas = rx_chirps.shape[0]
    assert num_antennas == steering_vec.shape[1], "Mismatch between number of receivers in "
    if num_antennas < num_sources:
        raise ValueError("number of sources shoule not exceed number ")
    
    # 8 x 8
    R = cov_matrix(rx_chirps)
    # 8 x 7
    noise_subspace = _noise_subspace(R, num_sources)
    # 7 x 181
    v = noise_subspace.T.conj() @ steering_vec.T
    # 181
    spectrum = np.reciprocal(np.sum(v * v.conj(), axis=0).real)

    return spectrum

def aoa_music_1D_mat(steering_vec, rx_chirps):
    """Implmentation of 1D MUltiple SIgnal Classification (MUSIC) algorithm on ULA (Uniformed Linear Array). 
    
    Current implementation assumes covariance matrix is not rank deficient and ULA spacing is half of the wavelength.
    .. math::
        P_{} (\\theta) = \\frac{1}{a^{H}(\\theta) \mathbf{E}_\mathrm{n}\mathbf{E}_\mathrm{n}^H a(\\theta)}
    where :math:`E_{n}` is the noise subpace and :math:`a` is the steering vector.
    

    Args:
        steering_vec (~np.ndarray): steering vector with the shape of (FoV/angel_resolution, num_ant). 
         FoV/angel_resolution is usually 181. It is generated from gen_steering_vec() function.
        rx_chirps (~np.ndarray): Ouput of the 1D range FFT. The shape is (num_obj, num_ant, num_chirps_per_frame).
        num_sources (int): Number of sources in the scene. Needs to be smaller than num_ant for ULA.
    
    Returns:
        (~np.ndarray): the spectrum of the MUSIC. Objects should be holes for the equation and thus sharp peaks.
    """
    # N x M x 8 x 1 * N x M x 1 x 8 -> N x M x 8 x 8
    covariance = np.matmul(rx_chirps, np.conjugate(rx_chirps).transpose(0, 1, 3, 2))
    # N x M x 8 x 7
    _, v = LA.eigh(covariance) 
    noise_subspace = v[..., :-1]
    # 1 x 1 x 181 x 8
    steering_vec = steering_vec[np.newaxis][np.newaxis]
    # N x M x 181 x 7
    v = np.matmul(steering_vec, noise_subspace.conj())
    # N x M x 181
    spectrum = np.reciprocal(np.sum(v * v.conj(), axis=-1).real)
    return spectrum

def aoa_root_music_1D(steering_vec, rx_chirps, num_sources):
    """Implmentation of 1D root MUSIC algorithm on ULA (Uniformed Linear Array). 
    
    The root MUSIC follows the same equation as the original MUSIC, only to solve the equation instead of perform 
    matrix multiplication.
    This implementations referred to the github.com/morriswmz/doatools.py

    Args:
        steering_vec (~np.ndarray): steering vector with the shape of (FoV/angel_resolution, num_ant). 
         FoV/angel_resolution is usually 181. It is generated from gen_steering_vec() function.
        rx_chirps (~np.ndarray): Ouput of the 1D range FFT. The shape is (num_ant, num_chirps_per_frame).
        num_sources (int): Number of sources in the scene. Needs to be smaller than num_ant for ULA.
    
    Returns:
        (~np.ndarray): the spectrum of the MUSIC. Objects should be holes for the equation and thus sharp peaks.
    """
    num_antennas = rx_chirps.shape[0]
    assert num_antennas == steering_vec.shape[1], "Mismatch between number of receivers in "
    if num_antennas < num_sources:
        raise ValueError("number of sources shoule not exceed number ")
    
    R = cov_matrix(rx_chirps)
    noise_subspace = _noise_subspace(R, num_sources)
    v = noise_subspace @ noise_subspace.T.conj()
    coeffs = np.zeros(num_antennas-1, dtype=np.complex64)
    for i in range(1, num_antennas):
            coeffs[i - 1] += np.sum(np.diag(v, i))
    coeffs = np.hstack((coeffs[::-1], np.sum(np.diag(v)), coeffs.conj()))
    
    z = np.roots(coeffs)
    z = np.abs(z[z <= 1.0])
    if len(z) < num_sources:
        return None
    z.sort()
    z = z[-num_sources:]

    # Assume to be half wavelength spacing
    sin_vals = np.angle(z) / np.pi
    locations = np.rad2deg(np.arcsin(sin_vals))

    return locations    

def aoa_spatial_smoothing(covariance_matrix, num_subarrays, forward_backward=False):
    """Perform spatial smoothing on the precomputed covariance matrix.
    
    Spatial smoothing is to decorrelate the coherent sources. It is performed over covariance matrix.
    This implementations referred to the github.com/morriswmz/doatools.py
    
    Args:
        covariance_matrx (~np.ndarray): Covariance matrix of input signal.
        num_subarrays (int): Number of subarrays to perform the spatial smoothing.
        forward_backward (bool): If True, perform backward smoothing as well.
    
    Returns:
        (~np.ndarray): Decorrelated covariance matrix.
    """
    num_receivers = covariance_matrix.shape[0]
    assert num_subarrays >=1 and num_subarrays <= num_receivers, "num_subarrays is wrong"

    # Forward pass
    result = covariance_matrix[:num_receivers-num_subarrays+1, :num_receivers-num_subarrays+1].copy()
    for i in range(1, num_subarrays):
        result += covariance_matrix[i:i+num_receivers-num_subarrays+1, i:i+num_receivers-num_subarrays+1]
    result /= num_subarrays
    if not forward_backward:
        return result
    
    # Adds backward pass
    if np.iscomplexobj(result):
        return 0.5 * (result + np.flip(result).conj())
    else:
        return 0.5 * (result + np.flip(result))

def aoa_esprit(steering_vec, rx_chirps, num_sources, displacement):
    """ Perform Estimation of Signal Parameters via Rotation Invariance Techniques (ESPIRIT) for Angle of Arrival.
    
    ESPRIT exploits the structure in the signal subspace.

    Args:
        steering_vec (~np.ndarray): steering vector with the shape of (FoV/angel_resolution, num_ant). 
         FoV/angel_resolution is usually 181. It is generated from gen_steering_vec() function.
        rx_chirps (~np.ndarray): Ouput of the 1D range FFT. The shape is (num_ant, num_chirps_per_frame).
        num_sources (int): Number of sources in the scene. Needs to be smaller than num_ant for ULA.
        displacement (int): displacmenet between two subarrays.
    
    Returns:
        (~np.ndarray): the spectrum of the ESPRIT. Objects should be holes for the equation and thus sharp peaks.
    """
    num_antennas = rx_chirps.shape[0]
    if displacement > num_antennas/2 or displacement <= 0:
        raise ValueError("The separation between two subarrays can only range from 1 to half of the original array size.")
        
    subarray1 = rx_chirps[:num_antennas - displacement]
    subarray2 = rx_chirps[displacement:]
    assert subarray1.shape == subarray2.shape, "separating subarrays encounters error."

    R1 = cov_matrix(subarray1)
    R2 = cov_matrix(subarray2)
    _, v1 = LA.eigh(R1)
    _, v2 = LA.eigh(R2)
    
    E1 = v1[:, -num_sources:]
    E2 = v2[:, -num_sources:]
    C = np.concatenate((E1.T.conj(), E2.T.conj()), axis=0) @ np.concatenate((E1, E2), axis=1)
    _, Ec = LA.eigh(C)
    Ec = Ec[::-1, :]

    phi = -Ec[:num_antennas, num_antennas:] @ LA.inv(Ec[num_antennas:, num_antennas:])
    w, _ = LA.eig(phi)

    sin_vals = np.angle(w) / np.pi
    locations = np.rad2deg(np.arcsin(sin_vals))

    return locations