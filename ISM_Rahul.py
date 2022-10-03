import numpy as np
import pyroomacoustics as pra
# from plotter import plot_on_sphere
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from pyDOE import lhs
# from icecream import ic
# import numba
# from numba import jit
import click
import h5py
from tqdm import tqdm
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from scipy.linalg import LinAlgWarning

warnings.filterwarnings(action='ignore', category=LinAlgWarning, module='sklearn')

print(2)

# @jit(nopython=True)
def grid_sphere_fib(n_points):
    """
    This function computes nearly equidistant points on the sphere
    using the fibonacci method
    Parameters
    ----------
    n_points: int
        The number of points to sample
    spherical_points: ndarray, optional
        A 2 x n_points array of spherical coordinates with azimuth in
        the top row and colatitude in the second row. Overrides n_points.
    References
    ----------
    http://lgdv.cs.fau.de/uploads/publications/spherical_fibonacci_mapping.pdf
    http://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    """
    # If no list was provided, samples points on the sphere
    # as uniformly as possible

    # Fibonnaci sampling
    offset = 2 / n_points
    increment = np.pi * (3 - np.sqrt(5))

    z = (np.arange(n_points) * offset - 1) + offset / 2
    rho = np.sqrt(1 - z ** 2)

    phi = np.arange(n_points) * increment

    x = np.cos(phi) * rho
    y = np.sin(phi) * rho

    return np.concatenate((np.atleast_2d(x), np.atleast_2d(y), np.atleast_2d(z))).T


def save_h5_from_dict(dictionary, savepath='/TD_point_sources.h5'):
    import os
    dirname = './SoundFieldData'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    savepath = dirname + savepath
    with h5py.File(savepath, 'w') as f:
        for key in dictionary.keys():
            f[key] = dictionary[key]
        f.close()


def reference_grid(steps, xmin=-.7, xmax=.7, z=0):
    x = np.linspace(xmin, xmax, steps)
    y = np.linspace(xmin, xmax, steps)
    # z = tf.zeros(shape = (steps,))
    X, Y = np.meshgrid(x, y)
    Z = z * np.ones(X.shape)
    return X, Y, Z


def adjustSNR(sig, snrdB=40, td=True):
    """
    Add zero-mean, Gaussian, additive noise for specific SNR
    to input signal

    Parameters
    ----------
    sig : Tensor
        Original Signal.
    noise : Vector or Tensor, optional
        Noise Tensor. The default is None.
    snrdB : int, optional
        Signal to Noise ratio. The default is 40.

    Returns
    -------
    x : Tensor
        Noisy Signal.

    """
    # Signal power in data from wav file
    sig_zero_mean = sig - sig.mean()
    psig = sig_zero_mean.var()

    # For x dB SNR, calculate linear SNR (SNR = 10Log10(Psig/Pnoise)
    snr_lin = 10.0 ** (snrdB / 10.0)

    # Find required noise power
    pnoise = psig / snr_lin

    if td:
        # Create noise vector
        # noise = np.sqrt(pnoise)*np.random.randn(sig.shape[0], sig.shape[1] )
        noise = np.sqrt(pnoise) * np.random.normal(0, 1, sig.shape)
    else:
        # complex valued white noise
        real_noise = np.random.normal(loc=0, scale=np.sqrt(2) / 2, size=sig.shape)
        imag_noise = np.random.normal(loc=0, scale=np.sqrt(2) / 2, size=sig.shape)
        noise = real_noise + 1j * imag_noise
        noise = np.sqrt(pnoise) * abs(noise) * np.exp(1j * np.angle(noise))

    # Add noise to signal
    sig_plus_noise = sig + noise
    return sig_plus_noise


def disk_grid_fibonacci(n, r, c=(0, 0), z=None):
    """
    Get circular disk grid points
    Parameters
    ----------
    n : integer N, the number of points desired.
    r : float R, the radius of the disk.
    c : tuple of floats C(2), the coordinates of the center of the disk.
    z : float (optional), height of disk
    Returns
    -------
    cg :  real CG(2,N) or CG(3,N) if z != None, the grid points.
    """
    r0 = r / np.sqrt(float(n) - 0.5)
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    gr = np.zeros(n)
    gt = np.zeros(n)
    for i in range(0, n):
        gr[i] = r0 * np.sqrt(i + 0.5)
        gt[i] = 2.0 * np.pi * float(i + 1) / phi

    if z is None:
        cg = np.zeros((3, n))
    else:
        cg = np.zeros((2, n))

    for i in range(0, n):
        cg[0, i] = c[0] + gr[i] * np.cos(gt[i])
        cg[1, i] = c[1] + gr[i] * np.sin(gt[i])
        if z != None:
            cg[2, i] = z
    return cg


def get_ISM_RIRs(room_coords,
                 room_height,
                 source_coords,
                 n_mics,
                 fs,
                 rt60,
                 max_order,
                 plot_RIR=False,
                 plot_room=False,
                 plot_array=False,
                 raytrace=True,
                 distributed_measurements=True,
                 snr=45):
    x_array = np.linspace(-.15, .15, n_mics)
    y_array = np.zeros_like(x_array)
    z_array = np.zeros_like(x_array)

    grid_array = np.stack([x_array, y_array, z_array], axis=0)

    # if you want array shifted but not in centre then comment out line 210
    # grid_array += shift
    # shift = np.array([.5, .2, 1.5])[..., None]

    # grid_array_2 = ....

    # room dimensions (corners in [x,y] meters)
    room_dim = [room_coords[0, 3], room_coords[1, 2], room_height]

    # plot 3d array
    if plot_array:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(grid_array[0], grid_array[1], grid_array[2], marker='d', color='b', s=100)
        # ax.view_init(0, 90)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        fig.show()

    # set uniform absorption coeffs to walls
    e_absorption, _ = pra.inverse_sabine(rt60, room_dim)
    pra.inverse_sabine(rt60, room_dim)

    # receiver centre coords
    receiver_center = np.asarray(room_dim)[:, np.newaxis] / 2

    # The locations of the microphones can then be computed as
    R = receiver_center + grid_array

    # Create the room
    room = pra.Room.from_corners(
        room_coords,
        fs=fs,
        materials=pra.Material(e_absorption),
        max_order=max_order,
        ray_tracing=raytrace,
        air_absorption=True)
    if raytrace:
        room.set_ray_tracing()

    # make room 3d
    room.extrude(height=room_height, materials=pra.Material(e_absorption))

    # add source to room
    room.add_source(source_coords)
    # add arrays to room
    room.add_microphone_array(R)

    # compute RIR
    room.compute_rir()
    # assert that arrays are correctly split (e.g. spherical array and reference array)
    # test_valid = R == room.mic_array.R[:, :n_mics]
    # assert (test_valid.all())

    # truncate to same length
    max_len = len(room.rir[0][0])
    for ii in range(len(room.rir)):
        if max_len < len(room.rir[ii][0]):
            max_len = len(room.rir[ii][0])
    trunc = max_len
    # split the arrays
    RIR_measured = np.zeros((n_mics, trunc))
    for ii in range(len(room.rir)):
        if ii < n_mics:
            RIR_measured[ii, :] = np.hstack((room.rir[ii][0], np.zeros((trunc - len(room.rir[ii][0], )))))

    RIR_measured = np.pad(RIR_measured, ((0, 0), (0, 16384 - trunc)))
    # time samples
    t = np.linspace(0, 16384 / fs, 16384)
    if snr is not None:
        RIR_measured = adjustSNR(RIR_measured, snrdB=snr)
    # plot RIRs
    if plot_RIR:
        fig, ax = plt.subplots(1, 1)
        ax.plot(t[:int(0.2 * fs)], RIR_measured[:3, :int(0.2 * fs)].T)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude')
        ax.grid('both', ls=':', color='k')
        ax.set_title('Generated ISM RIRs')
        # plt.savefig('Generated ISM RIRs.png', dpi = 150)
        plt.show()
    if plot_room:
        fig = plt.figure()
        fig, ax = room.plot(img_order=1)
        # ax.set_xlim([0, round(room_dim[0]) + 1])
        # ax.set_ylim([0, round(room_dim[1]) + 1])
        # ax.set_zlim([0, round(room_dim[2]) + 1])
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        # xyzlim = np.array([ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d()]).T
        # XYZlim = [min(xyzlim[0]),max(xyzlim[1])]
        # ax.set_xlim3d(XYZlim)
        # ax.set_ylim3d(XYZlim)
        # ax.set_zlim3d(XYZlim)
        ax.set_box_aspect((room_dim[0], room_dim[1], room_dim[2]))
        # try:
        #     ax.set_aspect('equal')
        # except NotImplementedError:
        #     pass
        # ax.view_init(azim=90, elev=0)
        plt.show()

    return RIR_measured, grid_array


@click.command()
# options_metavar='<options>'
@click.option('--plot_array', default=True, is_flag=True,
              help='plot the spherical array to visualise in 3d')
@click.option('--plot_room', default=True, is_flag=True,
              help='plot the room with 2nd order image sources \
               to visualise in 3d')
@click.option('--plot_rir', default=True, is_flag=True,
              help='plot the first three RIRs generated with the \
               ISM and view the first 0.2 sec')
@click.option('--raytrace', default=False, is_flag=True,
              help='Use combination of ray tracing and ISM')
@click.option('--RT60', default=0.2, type=float,
              help='Reverberation time from which to calculate \
                     the uniform absorption coefficient with the \
                     Sabine equation')
@click.option('--n_mics', default=5, type=click.IntRange(3, 300),
              help='Number of transducers in array')
@click.option('--max_XY', nargs=2, default=[10, 12], type=click.Tuple([float, float]),
              help='maximum X Y dimensions of random rooms')
@click.option('--n_rooms', default=1, type=int,
              help='Number of rooms to generate')
@click.option('--max_Z', default=3, type=click.FloatRange(2.4, 5.),
              help='Room height maximum')
@click.option('--max_order', default=11, type=click.IntRange(1, 15),
              help='Maximum order of image sources')
@click.option('--sample_rate', default=16000, type=int,
              help='Sample rate in samples/second')
def generate_ISM_data(plot_array, plot_room, plot_rir,
                      raytrace, rt60, n_mics,
                      max_xy, n_rooms, max_z,
                      max_order, sample_rate
                      ):
    num_shbox = n_rooms
    maxX, maxY = max_xy
    for n in range(num_shbox):
        first_corner = [0., 0.]
        second_corner = [0., np.random.uniform(2, maxY)]
        third_corner = [np.random.uniform(2., maxX), second_corner[1]]
        fourth_corner = [third_corner[0], 0.]
        roomdim = [first_corner, second_corner, third_corner, fourth_corner]
        # e.g. for room coords in [x, y]:
        # room_coords = np.array([[0, 0], [1.2, 3.3], [2.4, 3.3], [3.6, 0]]).T
        room_coords = np.array(roomdim).T
        room_height = np.random.uniform(2.4, max_z)
        # rev_time = np.random.uniform(rt60 - rt60 / 2, rt60 + rt60 / 2)
        rev_time = rt60

        # set source, slightly offset from corner and assert that it is within
        # room boundaries
        source_coords = roomdim[np.random.choice([0, 1, 2, 3])]
        source_coords.append(0.)
        if source_coords[0] > 0:
            multx = -1
        else:
            multx = 1
        if source_coords[1] > 0:
            multy = -1
        else:
            multy = 1

        snr = None
        source_coords = list(np.asarray(source_coords) + np.array([multx * .001, multy * .001, .001]))
        rirs_array, grid_array = get_ISM_RIRs(room_coords,
                                              room_height,
                                              source_coords,
                                              n_mics,
                                              sample_rate,
                                              rev_time,
                                              max_order,
                                              plot_RIR=plot_rir,
                                              plot_room=plot_room,
                                              plot_array=plot_array,
                                              raytrace=raytrace,
                                              snr=snr,
                                              )
        # array_data.append(rirs_sphere)
        # reference_data.append(rirs_ref)
        # grids_sphere.append(gridsphere)
        # grid_reference.append(grid_ref)
        # tdsamples = 16384
        # freq = np.fft.rfftfreq(tdsamples, d = 1/sample_rate)
        # recon_rir = reconstruct_FR(np.fft.rfft(rirs_sphere), 1200, freq, gridsphere, grid_ref)
        # rir_sets["pref_{}".format(lsf_number)] = rirs_ref
        # rir_sets["pm_{}".format(lsf_number)] = rirs_sphere
        # rir_sets["prec_{}".format(lsf_number)] = recon_rir
        # rir_sets["array_loc_{}".format(lsf_number)] = gridsphere
        # rir_sets["ref_loc_{}".format(lsf_number)] = grid_ref
        # save_paired_responses(rir_sets, data_dir, index= lsf_number)
        np.savez_compressed('./ISM_sphere.npz', array_data=rirs_array,
                            grid_measured=grid_array,
                            snr=snr, rt60=rev_time, room_coords=room_coords,
                            room_height=room_height, source_coords=source_coords,
                            fs=sample_rate)
    # array_data = np.asarray(array_data)
    # reference_data = np.asarray(reference_data)
    # grids_sphere = np.asarray(grids_sphere)
    # grid_reference = np.asarray(grid_reference)
    # f1 = h5py.File("data_ISM.hdf5", "w")
    #
    # dset1 = f1.create_dataset("array_data", array_data.shape, dtype='f',
    #                           data=array_data, chunks=(1, array_data.shape[1], array_data.shape[-1]))
    # dset2 = f1.create_dataset("reference_data", reference_data.shape, dtype='f', data=reference_data)
    # dset1.attrs['grid'] = grids_sphere
    # dset2.attrs['grid'] = grid_reference
    # f1.close()
    #


if __name__ == '__main__':
    print("Synthesising spherical array response set, please wait...")
    generate_ISM_data()
