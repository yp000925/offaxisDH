import warnings
from multiprocessing.dummy import Pool as ThreadPool

from  vis import save_scaled_image
from fftutils import FFT, fftshift

import numpy as np

from scipy.ndimage import gaussian_filter
from scipy.signal.windows import tukey

from skimage.restoration import unwrap_phase as skimage_unwrap_phase
from skimage.io import imread
from skimage.feature import blob_doh

# from astropy.utils.exceptions import AstropyUserWarning
# from astropy.convolution import convolve_fft, MexicanHat2DKernel

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

__all__ = ['Hologram', 'ReconstructedWave', 'unwrap_phase']
random_seed = 42
two_to_n = [2 ** i for i in range(13)]
float_precision = np.float64
complex_precision = np.complex128

def rebin_image(a, binning_factor):
    # Courtesy of J.F. Sebastian: http://stackoverflow.com/a/8090605
    if binning_factor == 1:
        return a

    new_shape = (a.shape[0]//binning_factor, a.shape[1]//binning_factor)
    sh = (new_shape[0], a.shape[0]//new_shape[0], new_shape[1],
          a.shape[1]//new_shape[1])
    return a.reshape(sh).mean(-1).mean(1)


def _load_hologram(hologram_path):
    """
    Load a hologram from path ``hologram_path`` using scikit-image and numpy.
    """
    try:
        from PIL import Image
        return np.array(Image.open(hologram_path, 'r'), dtype=np.float64)
    except ImportError:
        return np.array(imread(hologram_path), dtype=np.float64)


def _find_peak_centroid(image, gaussian_width=10):
    """
    Smooth the image, find centroid of peak in the image.
    """
    smoothed_image = gaussian_filter(image, gaussian_width)
    return np.array(np.unravel_index(smoothed_image.argmax(),
                                     image.shape))


def _crop_image(image, crop_fraction):
    """
    Crop an image by a factor of ``crop_fraction``.
    """
    if crop_fraction == 0:
        return image

    crop_length = int(image.shape[0] * crop_fraction)

    # if crop_length not in two_to_n:
    #     message = ("Final dimensions after crop should be a power of 2^N. "
    #                "Crop fraction of {0} yields dimensions ({1}, {1})"
    #                .format(crop_fraction, crop_length))
    #     warnings.warn(message, CropEfficiencyWarning)

    cropped_image = image[crop_length//2:crop_length//2 + crop_length,
                    crop_length//2:crop_length//2 + crop_length]
    return cropped_image


def _crop_to_square(image):
    """
    Ensure that hologram is square.
    """
    sh = image.shape
    if sh[0] != sh[1]:
        square_image = image[:min(sh), :min(sh)]
    else:
        square_image = image

    return square_image


# class CropEfficiencyWarning(AstropyUserWarning):
#     pass

class Hologram(object):
    """
    Container for holograms and methods to reconstruct them.
    """
    def __init__(self, hologram, crop_fraction=None, wavelength=405e-9,
                 rebin_factor=1, dx=3.45e-6, dy=3.45e-6, threads=2):
        """
        Parameters
        ----------
        hologram : `~numpy.ndarray`
            Input hologram
        crop_fraction : float
            Fraction of the image to crop for analysis
        wavelength : float [meters]
            Wavelength of laser
        rebin_factor : int
            Rebin the image by factor ``rebin_factor``. Must be an even integer.
        dx : float [meters]
            Pixel width in x-direction (unbinned)
        dy : float [meters]
            Pixel width in y-direction (unbinned)
        Notes
        -----
        Non-square holograms will be cropped to a square with the dimensions of
        the smallest dimension.
        """
        self.crop_fraction = crop_fraction
        self.rebin_factor = rebin_factor

        # Rebin the hologram
        square_hologram = _crop_to_square(float_precision(hologram))
        binned_hologram = rebin_image(square_hologram, self.rebin_factor)

        # Crop the hologram by factor crop_factor, centered on original center
        if crop_fraction is not None:
            self.hologram = _crop_image(binned_hologram, crop_fraction)
        else:
            self.hologram = binned_hologram

        # Construct an FFT object with shape/dtype of hologram:
        self.fft = FFT(self.hologram.shape, float_precision, complex_precision,
                       threads=threads)

        self.n = self.hologram.shape[0]
        self.wavelength = wavelength
        self.wavenumber = 2*np.pi / self.wavelength
        self.reconstructions = dict()
        self.dx = dx*rebin_factor
        self.dy = dy*rebin_factor
        self.mgrid = np.mgrid[0:self.n, 0:self.n]
        self.random_seed = random_seed
        self.apodization_window_function = None

    @classmethod
    def from_tif(cls, hologram_path, **kwargs):
        """
        Load a hologram from a TIF file.
        This class method takes the path to the TIF file as the first argument.
        All other arguments are the same as `~shampoo.Hologram`.
        Parameters
        ----------
        hologram_path : str
            Path to the hologram to load
        """
        hologram = _load_hologram(hologram_path)
        return cls(hologram, **kwargs)

    def reconstruct(self, propagation_distance,
                    plot_aberration_correction=False, plot_fourier_peak=False,
                    cache=False):
        """
        Reconstruct the wave at ``propagation_distance``.
        If ``cache`` is `True`, the reconstructed wave will be cached onto the
        `~shampoo.reconstruction.Hologram` object for quick retrieval.
        Parameters
        ----------
        propagation_distance : float
            Propagation distance [m]
        plot_aberration_correction : bool
            Plot the abberation correction visualization? Default is False.
        plot_fourier_peak : bool
            Plot the peak-centroiding visualization of the fourier transform
            of the hologram? Default is False.
        cache : bool
            Cache reconstructions onto the hologram object? Default is False.
        Returns
        -------
        reconstructed_wave : `~shampoo.reconstruction.ReconstructedWave`
            The reconstructed wave.
        """
        if cache:
            # Cache dictionary is accessible by keys = propagation distances
            cache_key = propagation_distance

            # If this reconstruction is cached, get it.
            if cache_key in self.reconstructions:
                reconstructed_wave = self.reconstructions[cache_key]

            # If this reconstruction is not cached, calculate it and cache it
            else:
                reconstructed_wave = self._reconstruct(propagation_distance,
                                                       plot_aberration_correction=plot_aberration_correction,
                                                       plot_fourier_peak=plot_fourier_peak)
                self.reconstructions[cache_key] = reconstructed_wave

        else:
            reconstructed_wave = self._reconstruct(propagation_distance,
                                                   plot_aberration_correction=plot_aberration_correction,
                                                   plot_fourier_peak=plot_fourier_peak)

        return ReconstructedWave(reconstructed_wave)

    def _reconstruct(self, propagation_distance,
                     plot_aberration_correction=False,
                     plot_fourier_peak=False):
        """
        Reconstruct wave from hologram stored in file ``hologram_path`` at
        propagation distance ``propagation_distance``.
        Parameters
        ----------
        propagation_distance : float
            Propagation distance [m]
        plot_aberration_correction : bool
            Plot the abberation correction visualization? Default is False.
        plot_fourier_peak : bool
            Plot the peak-centroiding visualization of the fourier transform
            of the hologram? Default is False.
        Returns
        -------
        reconstructed_wave : `~numpy.ndarray` (complex)
            Reconstructed wave from hologram
        """
        # Read input image
        apodized_hologram = self.apodize(self.hologram)

        # Isolate the real image in Fourier space, find spectral peak
        ft_hologram = self.fft.fft2(apodized_hologram)

        # Create mask based on coords of spectral peak:
        if self.rebin_factor != 1:
            mask_radius = 150./self.rebin_factor
        elif self.crop_fraction is not None and self.crop_fraction != 0:
            mask_radius = 150./abs(np.log(self.crop_fraction)/np.log(2))
        else:
            mask_radius = 150.

        x_peak, y_peak = self.fourier_peak_centroid(ft_hologram, mask_radius,
                                                    plot=plot_fourier_peak)

        mask = self.real_image_mask(x_peak, y_peak, mask_radius)

        # Calculate Fourier transform of impulse response function
        G = self.ft_impulse_resp_func(propagation_distance)

        # Now calculate digital phase mask. First center the spectral peak:
        shifted_ft_hologram = fftshift(ft_hologram * mask, [-x_peak, -y_peak])

        # Apodize the result
        psi = self.apodize(shifted_ft_hologram * G)
        digital_phase_mask = self.get_digital_phase_mask(psi,
                                                         plots=plot_aberration_correction)

        # Reconstruct the image
        psi = G * fftshift(self.fft.fft2(apodized_hologram * digital_phase_mask) * mask,
                           [-x_peak, -y_peak])

        reconstructed_wave = fftshift(self.fft.ifft2(psi))
        return reconstructed_wave

    def get_digital_phase_mask(self, psi, plots=False):
        """
        Calculate the digital phase mask (i.e. reference wave), as in Colomb et
        al. 2006, Eqn. 26 [1]_.
        Fit for a second order polynomial, numerical parametric lens with least
        squares to remove tilt, spherical aberration.
        .. [1] http://www.ncbi.nlm.nih.gov/pubmed/16512526
        Parameters
        ----------
        psi : `~numpy.ndarray`
            The product of the Fourier transform of the hologram and the Fourier
            transform of impulse response function
        plots : bool
            Display plots after calculation if `True`
        Returns
        -------
        phase_mask : `~numpy.ndarray`
            Digital phase mask, used for correcting phase aberrations.
        """
        # Need to flip mgrid indices for this least squares solution
        y, x = self.mgrid - self.n/2

        inverse_psi = fftshift(self.fft.ifft2(psi))

        unwrapped_phase_image = unwrap_phase(inverse_psi)/2/self.wavenumber
        smooth_phase_image = gaussian_filter(unwrapped_phase_image, 50)

        high = np.percentile(unwrapped_phase_image, 99)
        low = np.percentile(unwrapped_phase_image, 1)

        smooth_phase_image[high < unwrapped_phase_image] = high
        smooth_phase_image[low > unwrapped_phase_image] = low

        # Fit the smoothed phase image with a 2nd order polynomial surface with
        # mixed terms using least-squares.
        v = np.array([np.ones(len(x[0, :])), x[0, :], y[:, 0], x[0, :]**2,
                      x[0, :] * y[:, 0], y[:, 0]**2])
        coefficients = np.linalg.lstsq(v.T, smooth_phase_image)[0]
        field_curvature_mask = np.dot(v.T, coefficients)

        digital_phase_mask = np.exp(-1j*self.wavenumber * field_curvature_mask)

        if plots:

            # Set up figure and image grid
            fig = plt.figure(figsize=(12, 5))

            grid = ImageGrid(fig, 111,
                             nrows_ncols=(1, 2),
                             axes_pad=0.15,
                             share_all=True,
                             cbar_location="right",
                             cbar_mode="single",
                             cbar_size="7%",
                             cbar_pad=0.15,
                             )

            # Add data to image grid
            for ax, arr, title in zip(grid,
                                      [smooth_phase_image, field_curvature_mask],
                                      ['smothed phase image', 'curvature fit']):
                im = ax.imshow(arr, vmin=smooth_phase_image.min(),
                               vmax=smooth_phase_image.max(),
                               cmap=plt.cm.plasma, origin='lower',
                               interpolation='nearest')
                ax.set_title(title)
            # Colorbar
            ax.cax.colorbar(im)
            ax.cax.toggle_label(True)
            plt.show()

        return digital_phase_mask

    def apodize(self, array, alpha=0.075):
        """
        Force the magnitude of an array to go to zero at the boundaries.
        Parameters
        ----------
        array : `~numpy.ndarray`
            Array to apodize
        alpha : float between zero and one
            Alpha parameter for the Tukey window function. For best results,
            keep between 0.075 and 0.2.
        Returns
        -------
        apodized_arr : `~numpy.ndarray`
            Apodized array
        """
        if self.apodization_window_function is None:
            x, y = self.mgrid
            n = len(x[0])
            tukey_window = tukey(n, alpha)
            self.apodization_window_function = tukey_window[:, np.newaxis] * tukey_window

        apodized_array = array * self.apodization_window_function
        return apodized_array

    def ft_impulse_resp_func(self, propagation_distance):
        """
        Calculate the Fourier transform of impulse response function, sometimes
        represented as ``G`` in the literature.
        For reference, see Eqn 3.22 of Schnars & Juptner (2002) Meas. Sci.
        Technol. 13 R85-R101 [1]_,
        .. [1] http://x-ray.ucsd.edu/mediawiki/images/d/df/Digital_recording_numerical_reconstruction.pdf
        Parameters
        ----------
        propagation_distance : float
            Propagation distance [m]
        Returns
        -------
        G : `~numpy.ndarray`
            Fourier transform of impulse response function
        """
        x, y = self.mgrid - self.n/2
        first_term = (self.wavelength**2 * (x + self.n**2 * self.dx**2 /
                                            (2.0 * propagation_distance * self.wavelength))**2 /
                      (self.n**2 * self.dx**2))
        second_term = (self.wavelength**2 * (y + self.n**2 * self.dy**2 /
                                             (2.0 * propagation_distance * self.wavelength))**2 /
                       (self.n**2 * self.dy**2))
        G = np.exp(-1j * self.wavenumber * propagation_distance *
                   np.sqrt(1.0 - first_term - second_term))
        return G

    def real_image_mask(self, center_x, center_y, radius):
        """
        Calculate the Fourier-space mask to isolate the real image

        Parameters
        ----------
        center_x : int
            ``x`` centroid [pixels] of real image in Fourier space
        center_y : int
            ``y`` centroid [pixels] of real image in Fourier space
        radius : float
            Radial width of mask [pixels] to apply to the real image in Fourier
            space

        Returns
        -------
        mask : `~numpy.ndarray`
            Binary-valued mask centered on the real-image peak in the Fourier
            transform of the hologram.
        """
        x, y = self.mgrid
        mask = np.zeros((self.n, self.n))
        mask[(x-center_x)**2 + (y-center_y)**2 < radius**2] = 1.0

        # exclude corners
        buffer = 20
        mask[(x < buffer) | (y < buffer) |
             (x > len(x) - buffer) | (y > len(y) - buffer)] = 0.0

        return mask

    def fourier_peak_centroid(self, fourier_arr, mask_radius=None,
                              margin_factor=0.1, plot=False):
        """
        Calculate the centroid of the signal spike in Fourier space near the
        frequencies of the real image.
        Parameters
        ----------
        fourier_arr : `~numpy.ndarray`
            Fourier-transform of the hologram
        margin_factor : int
            Fraction of the length of the Fourier-transform of the hologram
            to ignore near the edges, where spurious peaks occur there.
        plot : bool
            Plot the peak-centroiding visualization of the fourier transform
            of the hologram? Default is False.
        Returns
        -------
        pixel : `~numpy.ndarray`
            Pixel at the centroid of the spike in Fourier transform of the
            hologram near the real image.
        """
        margin = int(self.n*margin_factor)
        #abs_fourier_arr = np.abs(fourier_arr)[margin:-margin, margin:-margin]
        abs_fourier_arr = np.abs(fourier_arr)[margin:self.n//2, margin:-margin]
        spectrum_centroid = _find_peak_centroid(abs_fourier_arr,
                                                gaussian_width=10) + margin

        if plot:
            fig, ax = plt.subplots()
            ax.imshow(np.log(np.abs(fourier_arr)), interpolation='nearest',
                      origin='lower')
            ax.plot(spectrum_centroid[1], spectrum_centroid[0], 'o')

            if mask_radius is not None:
                amp = mask_radius
                theta = np.linspace(0, 2*np.pi, 100)
                ax.plot(amp*np.cos(theta) + spectrum_centroid[1],
                        amp*np.sin(theta) + spectrum_centroid[0],
                        color='w', lw=2)
                ax.axvline(20)
                ax.axhline(20)
            plt.show()
        return spectrum_centroid

    def _reconstruct_multithread(self, propagation_distances, threads=4):
        """
        Reconstruct phase or intensity for multiple distances, for one hologram.
        Parameters
        ----------
        propagation_distances : `~numpy.ndarray` or list
            Propagation distances to reconstruct
        threads : int
            Number of threads to use via `~multiprocessing`
        Returns
        -------
        wave_cube : `~numpy.ndarray`
            Reconstructed waves for each propagation distance in a data cube of
            dimensions (N, m, m) where N is the number of propagation distances
            and m is the number of pixels on each axis of each reconstruction.
        """

        n_z_slices = len(propagation_distances)

        wave_shape = self.hologram.shape
        wave_cube = np.zeros((n_z_slices, wave_shape[0], wave_shape[1]),
                             dtype=np.complex128)

        def __reconstruct(index):
            # Reconstruct image, add to data cube
            wave = self.reconstruct(propagation_distances[index])
            wave_cube[index, ...] = wave._reconstructed_wave

        # Make the Pool of workers
        pool = ThreadPool(threads)
        pool.map(__reconstruct, range(n_z_slices))

        # close the pool and wait for the work to finish
        pool.close()
        pool.join()

        return wave_cube

    # def _detect_specimens(self, reconstructed_wave, propagation_distance,
    #                       margin=100, kernel_radius=4.0, save_png_to_disk=None):
    #     cropped_img = reconstructed_wave.phase[margin:-margin, margin:-margin]
    #     best_convolved_phase = convolve_fft(cropped_img,
    #                                         MexicanHat2DKernel(kernel_radius))
    #
    #     best_convolved_phase_copy = best_convolved_phase.copy(order='C')
    #
    #     # Find positive peaks
    #     blob_doh_kwargs = dict(threshold=0.00007,
    #                            min_sigma=2,
    #                            max_sigma=10)
    #     blobs = blob_doh(best_convolved_phase_copy, **blob_doh_kwargs)
    #
    #     # Find negative peaks
    #     negative_phase = -best_convolved_phase_copy
    #     negative_phase += (np.median(best_convolved_phase_copy) -
    #                        np.median(negative_phase))
    #     negative_blobs = blob_doh(negative_phase, **blob_doh_kwargs)
    #
    #     all_blobs = []
    #     for blob in blobs:
    #         if blob.size > 0:
    #             all_blobs.append(blob)
    #
    #     for neg_blob in negative_blobs:
    #         if neg_blob.size > 0:
    #             all_blobs.append(neg_blob)
    #
    #     if len(all_blobs) > 0:
    #         all_blobs = np.vstack(all_blobs)
    #
    #     # If save pngs:
    #     if save_png_to_disk is not None:
    #         path = "{0}/{1:.4f}.png".format(save_png_to_disk,
    #                                         propagation_distance)
    #         save_scaled_image(reconstructed_wave.phase, path, margin, all_blobs)
    #
    #     # Blobs get returned in rows with [x, y, radius], so save each
    #     # set of blobs with the propagation distance to record z
    #
    #     # correct blob positions for margin:
    #     all_blobs = np.float64(all_blobs)
    #     if len(all_blobs) > 0:
    #         all_blobs[:, 0] += margin
    #         all_blobs[:, 1] += margin
    #         all_blobs[:, 2] = propagation_distance
    #         return all_blobs
    #     else:
    #         return None


def unwrap_phase(reconstructed_wave, seed=random_seed):
    """
    2D phase unwrap a complex reconstructed wave.
    Essentially a wrapper around the `~skimage.restoration.unwrap_phase`
    function. The output will be type float64.
    Parameters
    ----------
    reconstructed_wave : `~numpy.ndarray`
        Complex reconstructed wave
    seed : float (optional)
        Random seed, optional.
    Returns
    -------
    `~numpy.ndarray`
        Unwrapped phase image
    """
    return skimage_unwrap_phase(2 * np.arctan(reconstructed_wave.imag /
                                              reconstructed_wave.real),
                                seed=seed)


class ReconstructedWave(object):
    """
    Container for reconstructed waves and their intensity and phase
    arrays.
    """
    def __init__(self, reconstructed_wave):
        self._reconstructed_wave = reconstructed_wave
        self._intensity_image = None
        self._phase_image = None
        self.random_seed = random_seed

    @property
    def intensity(self):
        """
        Reconstructed intensity (`~numpy.ndarray`, real)
        """
        if self._intensity_image is None:
            self._intensity_image = np.abs(self._reconstructed_wave)
        return self._intensity_image

    @property
    def phase(self):
        """
        Reconstructed, unwrapped phase (`~numpy.ndarray`, real)
        Phase unwrapping comes from `~skimage.restoration.unwrap_phase`.
        """
        if self._phase_image is None:
            self._phase_image = unwrap_phase(self._reconstructed_wave)

        return self._phase_image

    @property
    def reconstructed_wave(self):
        """
        Reconstructed wave (`~numpy.ndarray`, complex)
        """
        return self._reconstructed_wave

    def plot(self, phase=False, intensity=False, all=False,
             cmap=plt.cm.binary_r):
        """
        Plot the reconstructed phase and/or intensity images.
        Parameters
        ----------
        phase : bool
            Toggle unwrapped phase plot. Default is False.
        intensity : bool
            Toggle intensity plot. Default is False.
        all : bool
            Toggle unwrapped phase plot and . Default is False.
        cmap : `~matplotlib.colors.Colormap`
            Matplotlib colormap for phase and intensity plots.
        Returns
        -------
        fig : `~matplotlib.figure.Figure`
            Matplotlib figure object
        ax : `~matplotlib.axes.Axes`
            Matplotlib axis object
        """

        all_kwargs = dict(origin='lower', interpolation='nearest', cmap=cmap)

        phase_kwargs = all_kwargs.copy()
        phase_kwargs.update(dict(vmin=np.percentile(self.phase, 0.1),
                                 vmax=np.percentile(self.phase, 99.9)))

        fig = None
        if not all:
            if phase and not intensity:
                fig, ax = plt.subplots(figsize=(10,10))
                ax.imshow(self.phase, **phase_kwargs)
            elif intensity and not phase:
                fig, ax = plt.subplots(figsize=(10,10))
                ax.imshow(self.intensity, **all_kwargs)

        if fig is None:
            fig, ax = plt.subplots(1, 2, figsize=(18,8), sharex=True,
                                   sharey=True)
            ax[0].imshow(self.intensity, **all_kwargs)
            ax[0].set(title='Intensity')
            ax[1].imshow(self.phase, **phase_kwargs)
            ax[1].set(title='Phase')

        return fig, ax