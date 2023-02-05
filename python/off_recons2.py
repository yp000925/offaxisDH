import numpy as np
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from fftutils import FFT, fftshift
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase as skimage_unwrap_phase
from mpl_toolkits.axes_grid1 import ImageGrid
random_seed = 42
float_precision = np.float64
complex_precision = np.complex128

def plot_sp(arr):
    fig, ax = plt.subplots()
    ax.imshow(np.log(np.abs(arr)), interpolation='nearest',
              origin='lower')
    plt.show()
    return

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

def _load_hologram(hologram_path):
    """
    Load a hologram from path ``hologram_path`` using scikit-image and numpy.
    """
    try:
        from PIL import Image
        return np.array(Image.open(hologram_path, 'r'), dtype=np.float64)
    except ImportError:
        return np.array(imread(hologram_path), dtype=np.float64)
def rebin_image(a, binning_factor):
    # Courtesy of J.F. Sebastian: http://stackoverflow.com/a/8090605
    #down sample the image
    if binning_factor == 1:
        return a

    new_shape = (a.shape[0]//binning_factor, a.shape[1]//binning_factor)
    sh = (new_shape[0], a.shape[0]//new_shape[0], new_shape[1],
          a.shape[1]//new_shape[1])
    return a.reshape(sh).mean(-1).mean(1)

def _find_peak_centroid(image, gaussian_width=10,smoothed=False):
    """
    Smooth the image, find centroid of peak in the image.
    """
    if smoothed:
        smoothed_image = gaussian_filter(image, gaussian_width)
    else:
        smoothed_image = image
    return np.array(np.unravel_index(smoothed_image.argmax(),
                                     image.shape))
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
def _crop_image(image,crop_size):
    if crop_size == 0 or crop_size>=image.shape[0]:
        return image
    cropped_image = image[(image.shape[0]//2-crop_size//2):(image.shape[0]//2-crop_size//2) + crop_size,
                    (image.shape[0]//2-crop_size//2):(image.shape[0]//2-crop_size//2) + crop_size]
    return cropped_image

class Hologram(object):

    def __init__(self, hologram, crop_fraction=None,crop_size = None,wavelength=650e-9,
                 rebin_factor=1, dx=3.45e-6, dy=3.45e-6, threads=2,bg=None):
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
        if bg:
            self.bg = _load_hologram(bg)
        else:
            self.bg = np.zeros(hologram.shape)
        self.hologram = hologram-self.bg


        # Rebin the hologram
        self.hologram = _crop_to_square(self.hologram)
        self.hologram = rebin_image(self.hologram, self.rebin_factor) #down sample the image


        # Crop the hologram by factor crop_factor, centered on original center
        if crop_size is not None:
            self.hologram = _crop_image(self.hologram, crop_size)
        else:
            self.hologram = self.hologram

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
        self.mask_radius = 0
        self.shifted_spectrum = np.zeros(self.hologram.shape)
        self.origin_spectrum = self.fft.fft2(self.hologram)

    @classmethod
    def from_file(cls,file_path,**kwargs):
        """
        Load a hologram from a file.

        This class method takes the path to the TIF file as the first argument.
        All other arguments are the same as `~shampoo.Hologram`.

        Parameters
        ----------
        hologram_path : str
            Path to the hologram to load
        """
        hologram = _load_hologram(file_path)
        return cls(hologram, **kwargs)

    def get_shifted_spectrum(self,plot_fourier_peak=False,interested_region=None):
        ft_hologram = self.fft.fft2(self.hologram)

        # Create mask based on coords of spectral peak:
        if self.rebin_factor != 1:
            self.mask_radius = 100./self.rebin_factor
        elif self.crop_fraction is not None and self.crop_fraction != 0:
            self.mask_radius = 100./abs(np.log(self.crop_fraction)/np.log(2))
        else:
            self.mask_radius = 100.

        x_peak, y_peak = self.fourier_peak_centroid(ft_hologram, self.mask_radius,
                                                    plot=plot_fourier_peak,interested_region=interested_region)

        mask = self.real_image_mask(x_peak, y_peak)

        # Now calculate digital phase mask. First center the spectral peak:
        shifted_ft_hologram = fftshift(ft_hologram * mask, [-x_peak, -y_peak])
        return  shifted_ft_hologram

    def reconstruct(self, propagation_distance=15.5e-2,
                    plot_fourier_peak=False,search_area=None):
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

        # # Calculate Fourier transform of impulse response function
        G = self.ft_impulse_resp_func(propagation_distance)

        shifted_ft_hologram = self.get_shifted_spectrum(plot_fourier_peak=plot_fourier_peak,search_area=search_area)
        self.shifted_spectrum = shifted_ft_hologram
        psi = G*shifted_ft_hologram
        # #
        reconstructed_wave = fftshift(self.fft.ifft2(psi))
        return reconstructed_wave

    def fourier_peak_centroid(self, fourier_arr, mask_radius=None, plot=False,interested_region = None):
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
        #abs_fourier_arr = np.abs(fourier_arr)[margin:-margin, margin:-margin]
        # abs_fourier_arr = np.abs(fourier_arr)[margin:self.n//2, margin:-margin]
        mask_margin=10
        if interested_region:
            mask = np.zeros(fourier_arr.shape)
            mid_point = int(self.hologram.shape[0]/2)
            [vmin,vmax,hmin,hmax] = interested_region
            mask[vmin:vmax,hmin:hmax] = 1
            target_spectrum = fourier_arr * mask
        else:
            target_spectrum=fourier_arr
        # fig, ax = plt.subplots()
        # ax.imshow(mask)
        # plt.show()
        abs_target_spectrum = np.abs(target_spectrum)
        fig, ax = plt.subplots()
        ax.imshow(np.log(np.abs(abs_target_spectrum)),interpolation='nearest')
        plt.show()
        spectrum_centroid = _find_peak_centroid(abs_target_spectrum,
                                                gaussian_width=10)

        if self.n-spectrum_centroid[0]-mask_margin>self.mask_radius and self.n-spectrum_centroid[1]-mask_margin>self.mask_radius:
            self.mask_radius = self.mask_radius
        else:
            self.mask_radius = max([min([self.n-spectrum_centroid[0]-mask_margin,self.n-spectrum_centroid[1]-mask_margin]),0])


        if plot:
            fig, ax = plt.subplots()
            ax.imshow(np.log(np.abs(fourier_arr)), interpolation='nearest')
            ax.plot(spectrum_centroid[1], spectrum_centroid[0], 'o')
            if mask_radius is not None:
                amp = self.mask_radius
                theta = np.linspace(0, 2*np.pi, 100)
                ax.plot(amp*np.cos(theta) + spectrum_centroid[1],
                        amp*np.sin(theta) + spectrum_centroid[0],
                        color='w', lw=2)
                # ax.axvline(20)
                # ax.axhline(20)
            plt.show()
        return spectrum_centroid
    def real_image_mask(self, center_x, center_y):
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
        # if self.n-center_x-buffer>self.mask_radius and self.n-center_y-buffer>self.mask_radius:
        #     self.mask_radius = self.mask_radius
        # else:
        #     self.mask_radius = max([min([self.n-center_x-buffer,self.n-center_y-buffer]),0])
        mask[(x-center_x)**2 + (y-center_y)**2 < self.mask_radius**2] = 1.0
        return mask

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
if __name__ == "__main__":

    hologram_path = '/Users/zhangyunping/PycharmProjects/offaxisDH/Data/offaxis/holo2.bmp'
    bg =  '/Users/zhangyunping/PycharmProjects/offaxisDH/Data/offaxis/bg_both.bmp'
    # hologram_path = '/Users/zhangyunping/PycharmProjects/offaxisDH/Data/checkbg/bg_both7.bmp'
    # Construct the hologram object, reconstruct the complex wave
    h = Hologram.from_file(hologram_path,crop_size=1024,bg=bg,wavelength=650e-9, dx=3.45e-6*2, dy=3.45e-6*2)
    ori_spectrum = h.origin_spectrum
    fig,ax = plt.subplots()
    ax.imshow(np.log(np.abs(ori_spectrum)),interpolation='nearest')
    plt.show()

    # interested region = [vmin,vmax,hmin,hmax]
    shifted_spectrum= h.get_shifted_spectrum(plot_fourier_peak=True,interested_region=[300,600,400,700])
    # h.reconstruct(plot_fourier_peak=True,search_area='bottom_left')
    fig,ax = plt.subplots()
    ax.imshow(np.log(np.abs(shifted_spectrum)),interpolation='nearest')
    plt.show()
    #generate kernel
    propagation_distance = 15.5e-2
    G = h.ft_impulse_resp_func(propagation_distance)
    reconstructed_wave = fftshift(h.fft.ifft2(G*shifted_spectrum))
    fig,ax = plt.subplots()
    ax.imshow(np.abs(reconstructed_wave),interpolation='nearest')
    plt.show()
    # h.reconstruct(plot_fourier_peak=True,search_area='upper_right')