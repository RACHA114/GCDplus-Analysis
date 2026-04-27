import matplotlib.pyplot as plt
import matplotlib as mpl    # For mpl.colors.TwoSlopeNorm in Simulation.sfr_comp_map()
import numpy as np
import plotly.graph_objects as go   # For the 3d plot.
import copy # Part of the standard library.
import os

from scipy import stats
from scipy.spatial import cKDTree
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid
from hdbscan import HDBSCAN # For clustering in the InitialConditions class.
from sklearn import tree
from tqdm import tqdm   # Makes dynamic progress bars.

np.seterr(divide = 'ignore')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1


class Simulation:       
    def __init__(self, path, start, end, n_galaxies, split=None, initial_conditions=None, intervals=5e6,
                 abbs=["stars", "gas"], fill=3, idnum_arr=None):
        """
        This method initialiazes a Simulation object when it is created.

        The user can either use split or initial_conditions to associate the particles to their galaxy.

        initial_conditions is slower and does not take new particles into account, but it will not falsely label
        particles. For more information, look a the InitialConditions.__init__() docstring.

        split is faster and handles new particles well, but this leads to a small but nonzero amount of errors in
        labeling. Recommended for single galaxies (just use random numbers to split). For more information, see the
        Simulation.find_split() docstring.
        
        Parameters
        ----------
        path : str
            The path of the data files for the simulation.
        start : int
            The integer value of the first timestep.
        end : int
            The integer value of the last timestep
        n_galaxies : int
            The amount of galaxies.
        split : dict, optional
            Provides a way to differentiate between two galaxies, using a given id to 'split' particles between the
            galaxies in the simulation.
        initial_conditions : InitialConditions, opt
            Provides a way to differentiate between two galaxies using a clustering method on the first timestep.
        intervals : float, opt
            The interval between timesteps in years.
        abbs : list, opt
            A list of the abbs we want to use (not loading in useless abbs increases performance).
        fill : int, opt
            The amount of numbers in the data files. Recent GCD+ output files have 3, but older ones have 6.
        idnum_arr : array, opt
            1d numpy array with the timesteps in order. Useful for old GCD+ outputs where the timesteps are orderded,
            but they are not continuous (it might go from 000043 to 000049 for example).
        """
        self.path = path
        self.start = start
        self.end = end
        self.n_galaxies = n_galaxies
        self.intervals = intervals

        if idnum_arr is None:
            self.idnums = np.arange(start, end+1)
        else:
            self.idnums = idnum_arr

        self.dump_times = np.arange(len(self.idnums)) * intervals / 1e6
        self.len = len(self.idnums)

        self.atomic_mass = {'He': 4, 'C':12, 'N': 14, 'O':16, 'Ne':20, 'Mg':24, 'Si':28, 'Fe':56}

        # We let the user choose between using splits in the particle ids or initial_conditions. Both of these lead to
        # particle labeling, with different strengths and weaknesses.
        if initial_conditions is not None:
            self.split = None
            self.initial = initial_conditions
        elif split is not None:
            self.initial = None    # This is for read_data()'s logic.
            self.split = split
            try:
                for abb in abbs:
                    self.split[abb]
            except KeyError:
                raise KeyError("'split' must be a dictionnary with one entry per abb in 'abbs'.")
        else:
            self.split = None
            print('Initialising initial conditions...')
            self.initial = InitialConditions(path, n_galaxies, abbs, fill)

        self.timesteps = [Timestep(path, i, t, n_galaxies, split, abbs=abbs, fill=fill, dt=intervals) for i, t in
                          zip(self.idnums, self.dump_times)]


    def __getitem__(self, index):
        """
        Get a specific timestep by its index.

        Parameters
        ----------
        index : int
            The index of the timestep to retrieve.

        Returns
        -------
        Timestep
            The corresponding timestep object.
        """
        return self.timesteps[int(np.where(self.idnums == index)[0])]


    def read_data(self):
        """
        Read data for all timesteps in the simulation.
        """
        print('Reading data...')
        for t in tqdm(self.timesteps):
            t.read_data(initial_conditions=self.initial)


    def read_one_data(self, idnum):
        """
        Reads data for a single timestep.
        Used for multiprocessing.

        Parameters
        ----------
        idnum : int
            The idnum of the timestep whose data we want to read.
        """
        self.timesteps[idnum].read_data(initial_conditions=self.initial)


    def rotate_particles(self, axis, angle_degrees):
        """
        Rotate particle positions and velocities around a specified axis.

        Parameters
        ----------
        axis : str
            The axis to rotate around ('x', 'y', or 'z').
        angle_degrees : float
            The angle of rotation in degrees.
        """
        print(f'Rotating particles in the {axis} axis...')
        for t in tqdm(self.timesteps):
            t.rotate_particles(axis, angle_degrees)


    def find_density_peak_center(self, timestep, abb='dark', nbins=100, 
                             x_lim=(-30, 30), y_lim=(-30, 30), z_lim=(-30, 30)):
        """
        Find the center of a galaxy by locating the dark matter density peak.
        
        Parameters
        ----------
        timestep : Timestep object
            The timestep object containing particle data
        abb : str
            Particle type to use for finding center (usually 'dark')
        nbins : int
            Number of bins for the density histogram
        x_lim, y_lim, z_lim : tuple
            Initial limits for searching (should encompass the galaxy)
        
        Returns
        -------
        center : ndarray
            The [x, y, z] position of the density peak
        """
        # Get particle positions
        positions = timestep.particle_positions[abb]
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        
        # Create a 3D histogram (density grid)
        H, edges = np.histogramdd(
            positions, 
            bins=[nbins, nbins, nbins],
            range=[x_lim, y_lim, z_lim]
        )
        
        # Find the bin with maximum density
        max_bin_indices = np.unravel_index(np.argmax(H), H.shape)
        
        # Convert bin indices to physical coordinates
        x_center = (edges[0][max_bin_indices[0]] + edges[0][max_bin_indices[0] + 1]) / 2
        y_center = (edges[1][max_bin_indices[1]] + edges[1][max_bin_indices[1] + 1]) / 2
        z_center = (edges[2][max_bin_indices[2]] + edges[2][max_bin_indices[2] + 1]) / 2
        
        center = np.array([x_center, y_center, z_center])
        return center


    def adjust_center_with_density_peak(self, timestep, galaxy='gal1', abb='dark', 
                                    nbins=100, x_lim=(-50, 50), 
                                    y_lim=(-50, 50), z_lim=(-50, 50)):
    
        if galaxy == 'all':
            galaxy = 'gal1'
        galaxy_idx = int(galaxy[-1]) - 1
        
        # Get particle mask for the galaxy we want to center on
        if abb in timestep.abbs and timestep.particle_labels is not None:
            mask = timestep.particle_labels[abb] == galaxy_idx + 1
        else:
            mask = np.ones(len(timestep.particle_positions[abb]), dtype=bool)
        
        # Find the density peak center
        # First, temporarily "center" on the mean to get a better search range
        temp_center = np.mean(timestep.particle_positions[abb][mask], axis=0)
        
        # Search around this preliminary center
        search_range = 30  # kpc, adjust as needed
        x_lim = (temp_center[0] - search_range, temp_center[0] + search_range)
        y_lim = (temp_center[1] - search_range, temp_center[1] + search_range)
        z_lim = (temp_center[2] - search_range, temp_center[2] + search_range)
        
        # Now find the true density peak
        center = self.find_density_peak_center(timestep, abb=abb, nbins=nbins,
                                        x_lim=x_lim, y_lim=y_lim, z_lim=z_lim)
        
        # Store the centers (for compatibility with multi-galaxy simulations)
        timestep.galaxies_centers = [center]
        timestep.galaxies_velocities = [np.mean(timestep.particle_velocities[abb][mask], axis=0)]
        
        # Create centered positions
        timestep.particle_centered_positions = copy.deepcopy(timestep.particle_positions)
        timestep.particle_centered_velocities = copy.deepcopy(timestep.particle_velocities)
        
        # Shift all particle types
        for abb_type in timestep.abbs:
            if len(timestep.particle_centered_positions[abb_type].shape) == 1:
                continue
            
            timestep.particle_centered_positions[abb_type][:, 0] -= center[0]
            timestep.particle_centered_positions[abb_type][:, 1] -= center[1]
            timestep.particle_centered_positions[abb_type][:, 2] -= center[2]
            
            timestep.particle_centered_velocities[abb_type][:, 0] -= timestep.galaxies_velocities[0][0]
            timestep.particle_centered_velocities[abb_type][:, 1] -= timestep.galaxies_velocities[0][1]
            timestep.particle_centered_velocities[abb_type][:, 2] -= timestep.galaxies_velocities[0][2]


    def compute_and_save_dm_centers(self, output_file="dmcenter.txt",
                                    nbins=100, search_range=30):
        centers = []

        print("Computing DM density peaks...")
        for t in tqdm(self.timesteps):

            # Load only dark matter
            t.read_data(initial_conditions=self.initial)

            positions = t.particle_positions['dark']

            # Rough center estimate
            temp_center = np.mean(positions, axis=0)

            x_lim = (temp_center[0] - search_range, temp_center[0] + search_range)
            y_lim = (temp_center[1] - search_range, temp_center[1] + search_range)
            z_lim = (temp_center[2] - search_range, temp_center[2] + search_range)

            H, edges = np.histogramdd(
                positions,
                bins=[nbins, nbins, nbins],
                range=[x_lim, y_lim, z_lim]
            )

            max_bin = np.unravel_index(np.argmax(H), H.shape)

            x_center = (edges[0][max_bin[0]] + edges[0][max_bin[0] + 1]) / 2
            y_center = (edges[1][max_bin[1]] + edges[1][max_bin[1] + 1]) / 2
            z_center = (edges[2][max_bin[2]] + edges[2][max_bin[2] + 1]) / 2

            centers.append([t.idnum, x_center, y_center, z_center])

            # Free DM memory
            del t.particle_positions['dark']
            del t.particle_velocities['dark']
            del t.particle_masses['dark']
            del t.particle_ids['dark']

        np.savetxt(output_file, centers,
                header="idnum  x_peak  y_peak  z_peak")

        print(f"Saved DM centers to {output_file}")


    def load_dm_centers(self, filename="dmcenter.txt"):

        data = np.loadtxt(filename)

        self.dm_centers = {}

        for row in data:
            idnum = int(row[0])
            self.dm_centers[idnum] = np.array(row[1:4])

        print("Loaded pic DM")


    def center_on_saved_dm_peak(self, timestep):

        if not hasattr(self, "dm_centers"):
            raise RuntimeError("You must load DM centers first.")

        center = self.dm_centers[timestep.idnum]

        timestep.galaxies_centers = [center]
        timestep.galaxies_velocities = [np.zeros(3)]

        timestep.particle_centered_positions = copy.deepcopy(timestep.particle_positions)
        timestep.particle_centered_velocities = copy.deepcopy(timestep.particle_velocities)

        for abb in timestep.abbs:

            if len(timestep.particle_centered_positions[abb].shape) == 1:
                continue

            timestep.particle_centered_positions[abb][:, 0] -= center[0]
            timestep.particle_centered_positions[abb][:, 1] -= center[1]
            timestep.particle_centered_positions[abb][:, 2] -= center[2]


    def rotate_one_timestep(self, axis, angle_degrees, idnum):
        """
        Roatates the single timestep provided.
        Used for multiprocessing.
        
        Parameters
        ----------
        axis : str
            The axis to rotate around ('x', 'y', or 'z').
        angle_degrees : float
            The angle of rotation in degrees.
        idnum : int
            The idnum of the timestep we want to rotate
        """
        self.timesteps[idnum].rotate_particles(axis, angle_degrees)


    def get_sfr(self, galaxy, interval=1, max_dist=np.inf):
        """
        Calculate the star formation rate (SFR) over the simulation period.

        Parameters
        ----------
        galaxy : str
            The selection option ('gal1', 'gal2', 'all').
        interval : int, optional
            The distance between two data points.
        max_dist : float, optional
            The radius for which we want the sfr.

        Returns
        -------
        np.ndarray
            An array of SFR values for each timestep.
        """
        sfrs = []
        temp_sfr = []
        previous_ids = self.timesteps[0].particle_ids['stars']

        for t in self.timesteps:
            if self.split is not None:
                mask = t.get_mask('all', "stars", max_dist=max_dist)
                if galaxy == 'gal1':
                    mask *= ~t.split_mask('stars', self.split['stars'])
                elif galaxy == 'gal2':
                    mask *= t.split_mask('stars', self.split['stars'])
            # If it's for all galaxies and we don't use split, it can still work.
            elif galaxy == 'all':
                mask = t.get_mask('all', 'stars', max_dist=max_dist)
            else:
                raise RuntimeError("Creating a sim object with InitialConditions does not allow to get new stars in"
                                   "a specific galaxy.")

            t.particle_centered_positions["stars"][mask]

            if t == self.timesteps[0]:
                t.sfr = 0
            else:
                mask *= ~np.isin(t.particle_ids['stars'], previous_ids)
                t.sfr = np.sum(t.particle_masses['stars'][mask]) / self.intervals
            temp_sfr.append(t.sfr)
            previous_ids = t.particle_ids['stars']

            if len(temp_sfr) == interval:
                sfrs.append(np.mean(temp_sfr))
                temp_sfr = []

            if t.idnum == self.len and temp_sfr:
                sfrs.append(np.mean(temp_sfr))

        return np.array(sfrs)


    def get_new_stars(self, idnum, galaxy='all', nbtimesteps=1, max_dist=np.inf):
        """
        Returns the new stars for a timestep.

        Parameters
        ----------
        idnum : int
            The idnum of the timestep we want to look at. Between 1 and the total number of timesteps.
        galaxy : str, opt
            The galaxy we want to look at.
        nbtimestep : int, opt
            How recent do the stars have to be to be considered "new" (in timesteps).
        max_dist : float, opt
            The radius inside of which we want the new stars.
        
        Returns
        -------
        new_x, new_y, new_z : array
            The array of the new stars' positions.
        new_mass : array
            An array of the new stars' masses.
        new_ids : array
            An array of the new stars' ids.
        """
        start = idnum - nbtimesteps

        if start < 0 or idnum > self.idnums[-1]:
            raise ValueError("idnum - nbtimesteps must be bigger or equal to 0 and idnum must be smaller or equal to" \
                            f"{self.idnums[-1]}")

        previous_stars = self.timesteps[start].particle_ids['stars']

        t = self.timesteps[idnum]

        if self.split is not None:
            mask = t.get_mask('all', "stars", max_dist=max_dist)
            if galaxy == 'gal1':
                mask *= ~t.split_mask('stars', self.split['stars'])
            elif galaxy == 'gal2':
                mask *= t.split_mask('stars', self.split['stars'])
        # If it's for all galaxies and we don't use split, it can still work.
        elif galaxy == 'all':
            mask = t.get_mask('all', 'stars', max_dist=max_dist)
        else:
            raise RuntimeError("Creating a sim object with InitialConditions does not allow to get new stars in"
                                "a specific galaxy.")
        
        mask *= ~np.isin(t.particle_ids['stars'], previous_stars)

        new_x = t.particle_centered_positions['stars'][mask][:, 0]
        new_y = t.particle_centered_positions['stars'][mask][:, 1]
        new_z = t.particle_centered_positions['stars'][mask][:, 2]
        new_mass = t.particle_masses['stars'][mask]
        new_ids = t.particle_ids['stars'][mask]

        return new_x, new_y, new_z, new_mass, new_ids


    def plot_sfr_map(self, idnum, plot_dir=None, save=False, cmap='hot', nbins=500, colorbar=True, fig_size=(4, 6),
                     x_lim=(-30, 30), y_lim=(-30, 30), z_lim=(-30, 30), vmin=0, vmax=2e-6, dpi=400, galaxy='all',
                     max_dist=np.inf, nbtimesteps=1, plan='xy', make_plot=False):
        """
        Plot the density distribution of new stars.

        Parameters
        ----------
        idnum : int
            The idnum of the timestep which we want to look at.
        plot_dir : str, optional
            Directory to save the plot if `save` is True.
        save : bool, optional
            If True, save the plot to a file. Otherwise, display it.
        cmap : str, optional
            The colormap to use for the plot.
        nbins : int, optional
            Number of bins for the density calculation.
        colorbar : bool, optional
            If True, include a colorbar in the plot.
        fig_size : tuple, optional
            Size of the figure in inches (width, height).
        x_lim, y_lim, z_lim : tuple, optional
            Limits for the spatial axis in kiloparsecs.
        vmin : float, optional
            Minimum value for the color scale.
        vmax : float, optional
            Maximum value for the color scale.
        dpi : int, optional
            Resolution of the plot in dots per inch.
        galaxy : str, optional
            The galaxy to focus on ('gal1', 'gal2', or 'all').
        max_dist : float, optional
            Maximum distance from the galaxy center to include particles.
        nbtimestep : int, opt
            How recent do the stars have to be to be considered "new" (in timesteps).
        plan : {'xy', 'zy', 'xz'}, optional
            Projection plane for the SFR map.
        make_plot : bool, optional
            If True, generates and displays/saves the plot. If False, returns the histogram data.

        Returns
        -------
        sfr : ndarray
            2D array of the SFR (if "make_plot" is False).
        x_edges : ndarray
            Bin edges along the x-axis (if "make_plot" is False).
        y_edges : ndarray
            Bin edges along the y-axis (if "make_plot" is False).
        """
        abb = 'stars'

        # Get the time in years.
        time = nbtimesteps * self.intervals

        t = self.timesteps[idnum]

        mask = t.get_mask(galaxy, abb, max_dist=max_dist)

        # Depending on the plane, it'll plot different things. We also change the definitions of the limits if needed.
        if plan == 'xy':
            zero, one = 0, 1
        elif plan == 'zy':
            zero, one = 2, 1
            x_lim = z_lim
        elif plan == 'xz':
            zero, one = 0, 2
            y_lim = z_lim
        else:
            raise ValueError("'plan' must be either 'xy', 'zy' or 'xz'")

        # Extracting particles positions.
        x = t.particle_centered_positions[abb][mask][:, zero]
        y = t.particle_centered_positions[abb][mask][:, one]

        new_x, new_y, _, new_mass = self.get_new_stars(idnum=idnum, galaxy=galaxy, nbtimesteps=nbtimesteps,
                                                 max_dist=max_dist)

        # Define bins for density plots
        x_len = x_lim[1] - x_lim[0]
        y_len = y_lim[1] - y_lim[0]
        axe_max = min(x_len, y_len)
        
        xy_bins = [int((x_len/axe_max)*nbins), int((y_len/axe_max)*nbins)]

        # Calculate the area of a pixel in kiloparsecs^2
        resolution_element_area = (axe_max/nbins)**2

        # Find the number of new stars per bin.
        sfr, x_edges, y_edges, _ = stats.binned_statistic_2d(new_x, new_y, values=new_mass, statistic='sum',
                                                             bins=xy_bins, range=[x_lim, y_lim])

        # Get the logarithmic density of new stars.
        sfr = sfr / (resolution_element_area * time)

        # All nan values in the new star plots are set to zero (doing ns_xy[ns_xy == np.nan] = 0 does not work).
        sfr[sfr <= 0] = np.nan

        if make_plot:
            return sfr, x_edges, y_edges

        # Plotting
        plt.clf()
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi, ncols=1)
        ax.set_aspect('equal')

        ax.set_title(f'T = {int(t.time)} Myrs', x=0.02, y=0.98, ha='left', va='top', pad=-2)

        # XY plane
        ax1 = ax
        im1 = ax1.pcolormesh(x_edges, y_edges, sfr.T, cmap=cmap, vmin=vmin, vmax=vmax, zorder=10)
        ax1.set_xlabel(f'{plan[0]} [kpc]')
        ax1.set_ylabel(f'{plan[1]} [kpc]')

        ax2 = ax
        ax2.plot(x, y, marker="o", color="0.5", linestyle="None", zorder=0, markersize=0.5)
        ax2.set_xlim(x_lim)
        ax2.set_ylim(y_lim)

        # Color bar
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im1, cax=cax, orientation='vertical', pad=0.2)
            cbar.set_label(r'$\Delta$ SFR [M$_{\odot}$ kpc$^{-2}$ yr$^{-1}$]')

        plt.tight_layout()

        if save:
            plt.savefig(f"{plot_dir}/sfr_{str(t.idnum).zfill(3)}.png")
            plt.close()
        else:
            plt.show()


    def sfr_comp_map(self, sim, idnum, plot_dir=None, save=False, cmap='seismic', nbins=135,  galaxy='all',
                     nbtimesteps=1, x_lim=(-30, 30), y_lim=(-30, 30), z_lim=(-30, 30), colorbar=True,
                     fig_size=(6, 6), vmin=-5e-1, vmax=5e-1, dpi=400, auto_norm=True, plan='xy', make_plot=True):
        """
        Generates and optionally plots a comparative star formation rate (SFR) map between two simulation runs (self
        and "sim").

        Parameters
        ----------
        sim : Simulation object
            The simulation object to compare against.
        idnum : int
            Identifier for the snapshot or timestep.
        plot_dir : str, optional
            Directory to save the plot. Required if "save" is True.
        save : bool, optional
            If True, saves the plot to "plot_dir".
        cmap : str, optional
            Colormap for the SFR difference plot.
        nbins : int, optional
            Number of bins for the 2D histogram along the shortest axis.
        galaxy : str, optional
            Specifies which galaxy to analyze.
        nbtimesteps : int, optional
            Number of timesteps to include in the SFR calculation.
        x_lim, y_lim, z_lim : tuple, optional
            Limits for the spatial axes in kiloparsecs.
        colorbar : bool, optional
            If True, displays a colorbar on the plot.
        fig_size : tuple, optional
            Size of the figure in inches.
        vmin, vmax : float, optional
            Minimum and maximum values for the colormap normalization.
        dpi : int, optional
            Resolution of the saved figure.
        auto_norm : bool, optional
            If True, automatically picks vmin and vamx based on minimum and maximum values.
        plan : {'xy', 'zy', 'xz'}, optional
            Projection plane for the SFR map.
        make_plot : bool, optional
            If True, generates and displays/saves the plot. If False, returns the histogram data.

        Returns
        -------
        sfr : ndarray
            2D array of SFR differences (if "make_plot" is False).
        x_edges : ndarray
            Bin edges along the x-axis (if "make_plot" is False).
        y_edges : ndarray
            Bin edges along the y-axis (if "make_plot" is False).
        """
        x_len = x_lim[1] - x_lim[0]
        y_len = y_lim[1] - y_lim[0]
        z_len = z_lim[1] - y_lim[0]

        if plan == 'xy':
            x_plot_len, y_plot_len = x_len, y_len
            x_plot_lim, y_plot_lim = x_lim, y_lim
        if plan == 'zy':
            x_plot_len, y_plot_len = z_len, y_len
            x_plot_lim, y_plot_lim = z_lim, y_lim
        if plan == 'xz':
            x_plot_len, y_plot_len = x_len, z_len
            x_plot_lim, y_plot_lim = x_lim, z_lim

        axe_max = min(x_plot_len, y_plot_len)

        bins = [int((x_plot_len/axe_max)*nbins), int((y_plot_len/axe_max)*nbins)]

        # Calculate the area of a pixel in kiloparsecs^2.
        resolution_element_area = (axe_max/nbins)**2

        # Get the map of sfr in the first run.
        new_x, new_y, new_z, new_mass, new_ids = self.get_new_stars(idnum=idnum, galaxy=galaxy, nbtimesteps=nbtimesteps)

        if galaxy != 'all':
            # We apply a mask so that we're only looking at SFR in the specified galaxy
            new_x = new_x[new_ids < self.split['stars']]
            new_y = new_y[new_ids < self.split['stars']]
            new_z = new_z[new_ids < self.split['stars']]
            new_mass = new_mass[new_ids < self.split['stars']]

        # Depending on the plane we are plotting, we get the values for the x and y coordinates in the plot.
        if plan == 'xy':
            x, y = new_x, new_y
        if plan == 'zy':
            x, y = new_z, new_y
        if plan == 'xz':
            x, y = new_x, new_z

        sfr, x_edges, y_edges = np.histogram2d(x, y, weights=new_mass, bins=bins, range=[x_plot_lim, y_plot_lim])
        sfr = sfr / (resolution_element_area * nbtimesteps * self.intervals)

        # Get the map of sfr in the second run
        new_x2, new_y2, new_z2, new_mass2, _ = sim.get_new_stars(idnum=idnum, galaxy=galaxy, nbtimesteps=nbtimesteps)

        # Depending on the plane we are plotting, we get the values for the x and y coordinates in the plot.
        if plan == 'xy':
            x2, y2 = new_x2, new_y2
        if plan == 'zy':
            x2, y2 = new_z2, new_y2
        if plan == 'xz':
            x2, y2 = new_x2, new_z2

        sfr2, _, _ = np.histogram2d(x2, y2, weights=new_mass2, bins=bins, range=[x_plot_lim, y_plot_lim])
        sfr2 = sfr2 / (resolution_element_area * nbtimesteps * sim.intervals)

        # What we'll display is sfr in the first run minus sfr in the second run.
        sfr -= sfr2

        # We make an array that has the value 1 where there is star formation in either run and the value nan elsewhere.
        # This is to make it so that pcolormesh only displays where there is star formation and leaves things blank elsewhere.
        nanarr = (sfr <= 0) & (sfr2 <= 0)
        nanarr = ~nanarr
        nanarr = nanarr.astype(float)
        nanarr[nanarr == 0] = np.nan

        # Multiply the sfr array with the nan array.
        sfr *= nanarr

        # If we don't make the plot directly, we simply return the histogram and its edges.
        if not make_plot:
            return sfr, x_edges, y_edges
        
        plt.clf()
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        ax = fig.add_subplot(autoscale_on=False, xlim=x_lim, ylim=y_lim)
        ax.set_aspect('equal')

        ax.set_title(f'T = {int(self.time)} Myrs', x=0.02, y=0.98, ha='left', va='top', pad=-2, color='w')

        # An assymetric norm for the cmap based on the minimum and maximum values (and centered on 0).
        if auto_norm:
            try:
                norm = mpl.colors.TwoSlopeNorm(vmin=np.nanmin(sfr), vcenter=0, vmax=np.nanmax(sfr))
            except ValueError:
                try:
                    norm = mpl.colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=np.nanmax(sfr))
                except ValueError:
                    norm = mpl.colors.TwoSlopeNorm(vmin=np.nanmin(sfr), vcenter=0, vmax=1)

        # The norm is simply based around the vmin and vmax arguments.
        else:
            norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        ax1 = ax
        im1 = ax1.pcolormesh(x_edges, y_edges, sfr.T, cmap=cmap, norm=norm)
        ax1.set_xlabel(f'{plan[0]} [kpc]')
        ax1.set_ylabel(f'{plan[1]} [kpc]')

        # Plot one marker per star (below the sfr plot).
        ax2 = ax
        x = self.particle_centered_positions["stars"][:, 0]
        y = self.particle_centered_positions["stars"][:, 1]
        z = self.particle_centered_positions["stars"][:, 2]

        # Adjusting for the plane we want to plot.
        if plan == 'xy':
            x, y = x, y
        if plan == 'zy':
            x, y = z, y
        if plan == 'xz':
            x, y = x, z

        ax2.plot(x, y, marker="o", color="0.5", linestyle="None", zorder=0, markersize=0.5)
        ax2.set_xlim(x_lim)
        ax2.set_ylim(y_lim)

        # Make a black background.
        ax3 = ax
        im3 = ax3.pcolormesh(np.linspace(-x_plot_len/2, x_plot_len/2, nbins),
                            np.linspace(-y_plot_len/2, y_plot_len/2, nbins),
                            np.ones([nbins, nbins]), cmap='hot', vmin=1, zorder=-10)

        # Colorbar.
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im1, cax=cax, orientation='vertical', pad=0.2)
            cbar.set_label(r'$\Delta$ SFR [M$_{\odot}$ kpc$^{-2}$ yr$^{-1}$]')

        plt.tight_layout()

        if save:
            plt.savefig(f"{plot_dir}/sfr_{str(self.idnum).zfill(3)}.png")
            plt.close()
        else:
            plt.show()


    def sfr_radial_profile(self, idnum, galaxy='all', nbtimesteps=1,
                      r_max=30, nbins=50):

        new_x, new_y, _, new_mass, _ = self.get_new_stars(
            idnum=idnum, galaxy=galaxy, nbtimesteps=nbtimesteps
        )

        t = self.timesteps[idnum]
        # Rayon
        r = np.sqrt(new_x**2 + new_y**2)

        radial_path = f"radialSFR&SFE/radial_{str(t.idnum).zfill(6)}.txt"

        os.makedirs(os.path.dirname(radial_path), exist_ok=True)
        
        # On redéfinit des variables moins longues à écrire
        gas_pos = t.particle_centered_positions['gas']
        gas_mass = t.particle_masses['gas']

        r_gas = np.sqrt(gas_pos[:, 0]**2 + gas_pos[:, 1]**2)

        # Bins radiaux
        bins = np.linspace(0, r_max, nbins+1)

        sfr, edges = np.histogram(r, bins=bins, weights=new_mass)

        gas_mass_bin, _ = np.histogram(r_gas, bins=bins, weights=gas_mass) 
        # Surface des anneaux 
        area = np.pi * (edges[1:]**2 - edges[:-1]**2)

        sfr = sfr / (area * nbtimesteps * self.intervals)

        gas_surface = gas_mass_bin / area

        #Calcul de l'efficacité de la formation d'étoiles 
        sfe = sfr / gas_surface

        #On sauvegarde les données dans un fichier texte
        np.savetxt(radial_path, np.array([edges[:-1], gas_mass_bin, sfr, sfe]).T, 
                   header="Radius(kpc)  M_gas  SFR(M_☉ kpc^-2 yr^-1)  SFE(yr^-1)")

        return sfr, edges, sfe

    def sfr_angular_profile(self, idnum, galaxy='all', nbtimesteps=1,
                        r_min=0, r_max=30, nbins=36):

        new_x, new_y, _, new_mass, _ = self.get_new_stars(
            idnum=idnum, galaxy=galaxy, nbtimesteps=nbtimesteps
        )

        t = self.timesteps[idnum]

        r = np.sqrt(new_x**2 + new_y**2)
        theta = np.arctan2(new_y, new_x)
        
        angular_path = f"AngularSFR&SFE/angular_{str(t.idnum).zfill(6)}.txt"

        os.makedirs(os.path.dirname(angular_path), exist_ok=True)
        
        gas_pos = t.particle_centered_positions['gas']
        gas_mass = t.particle_masses['gas']

        theta_gas = np.arctan2(gas_pos[:, 1], gas_pos[:, 0])
        # Garder seulement une couronne radiale
        mask = (r >= r_min) & (r <= r_max)

        theta = theta[mask]
        mass = new_mass[mask]

        bins = np.linspace(-np.pi, np.pi, nbins+1)

        gas_mass_bin, _ = np.histogram(theta_gas, bins=bins, weights=gas_mass)

        sfr, edges = np.histogram(theta, bins=bins, weights=mass)

        area = (r_max - r_min) * (2 * np.pi / nbins)
        # normalisation 
        sfr = sfr / (nbtimesteps * self.intervals)

        gas_surface = gas_mass_bin / area

        sfe = sfr / gas_surface

        np.savetxt(angular_path, np.array([edges[:-1], gas_mass_bin, sfr, sfe]).T, 
                   header="angle()  M_gas  SFR(M_☉ kpc^-2 yr^-1)  SFE(yr^-1)")

        return sfr, edges, sfe


    def plot_sfr(self, radii=np.inf, plot_dir=None, save=False, galaxy="all", fig_size=(6, 4), interval=1, dpi=400,
                 x_lim=None, y_lim=None, nb_dist_minima=None):
        """
        Plot the star formation rate over time with a possibility of studying for different radii.

        Parameters
        ----------
        radii : tuple, optional
            The different radii we want to study.
        plot_dir : str, optional
            Directory to save the plot if `save` is True.
        save : bool, optional
            If True, save the plot to a file. Otherwise, display it.
        galaxy : str
            The selection option ('gal1', 'gal2', 'all', or 'noise').
        fig_size : tuple, optional
            Size of the figure in inches (width, height).
        interval : int, optional
            The distance between two data points. If it is not 1, the average between two timsteps will be used.
        dpi : int, optional
            Resolution of the plot in dots per inch.
        xlim : list, optional
            Limits for the x axis.
        ylim : list, optional
            Limits for the y axis
        nb_dist_minima : int, optional
            How many of the minima in the distance between two interacting galaxies do we want to plot.
        """

        plt.clf()
        fig = plt.figure(figsize=fig_size, dpi=dpi)

        # Create a list of the times around which we take at the average sfr.
        times = []
        temp_times = []

        # For all of the times except t = 0 (no star formation).
        for i, time in enumerate(self.dump_times[1:]):
            temp_times.append(time)

            if len(temp_times) == interval:
                times.append(np.mean(temp_times))
                temp_times = []

            if i + 1 == self.len and temp_times:
                times.append(np.mean(temp_times))

        # If we want to look for a specific radius or specific radii.
        if radii != np.inf:
            for radius in radii:
                plt.plot(times, self.get_sfr(galaxy, interval=interval, max_dist=radius),
                         label=f"SFR r < {radius} kpc")

        # If we just want the sfr throughout the whole galaxy.
        else:
            plt.plot(times, self.get_sfr(galaxy, interval=interval),
                         label=f"SFR")

        if nb_dist_minima:
            for i in range(nb_dist_minima):
                plt.axvline(x=self.get_distance_minima()[i, 1], color="black")

        plt.ylabel(r'SFR [M$_{\odot}$/yr]')
        plt.xlabel(r'Time [Myrs]')
        plt.legend()

        if x_lim:
            plt.xlim(x_lim)
        if y_lim:
            plt.ylim(y_lim)

        if save:
            plt.savefig(f"{plot_dir}/sfr_{radii}.png")
            plt.close()
        else:
            plt.show()


    def sfr_output(self, run_name):
        """
        Produces an output file where each row represents one timestep and contains 4 columns: time (yrs), total mass
        of stars (solar masses), total mass of gas (solar masses), and the sfr (solar masses per year)
        
        Parameters
        ----------
        run_name : str
            The name of the run. Used to give the right name to the file.
        """
        times, m_stars, m_gas = [], [], []

        for t in self.timesteps:

            times.append(t.time*1e6)
            m_stars.append(np.sum(t.particle_masses['stars']))

            try:
                m_gas.append(np.sum(t.particle_masses['gas']))
            except:
                # If we want to output sfr files but don't have gas particles files.
                m_gas.append(np.nan)

        data = np.array([times, m_stars, m_gas, self.get_sfr(galaxy='all', interval=1, max_dist=np.inf)])

        np.savetxt(f'{run_name}_sfr', data.T, fmt=['%.3e','%.5e', '%.5e', '%.5e'])


    def plot_distance(self):
        """
        Plot the distance between the two largest galaxies over time.

        Returns
        -------
        list
            A list of distances between the two largest galaxies for each timestep.
        """
        fig = plt.figure(figsize=(6, 4), dpi=400)
        dist = []
        for t in self.timesteps:
            dist.append(np.linalg.norm(t.galaxies_centers[0] - t.galaxies_centers[1]))
        plt.plot(self.dump_times, dist)
        plt.xlabel('Time [Myrs]')
        plt.ylabel('Distance [kpc]')
        plt.show()
        return dist
    

    def get_distance_minima(self):
        """
        Finds the minima of the distance between the two galaxies and the time when they occur.
        
        Returns
        -------
        array
            An array where the first column is the distance at the minima and the other is the time.
        """
        dist = []

        for t in self.timesteps:
            dist.append(np.linalg.norm(t.galaxies_centers[0] - t.galaxies_centers[1]))
        
        dist_list = []
        time_list = []

        # The first and last timesteps are excluded.
        for i in range(len(dist[1:-1])):
            if dist[i+1] < dist[i] and dist[i+1] < dist[i+2]:
                dist_list.append(dist[i+1])
                time_list.append(self.timesteps[i+1].time)
        
        return np.stack((np.array(dist_list), np.array(time_list)), axis=-1)


    def plot_12_plus_log_x(self, elements, gas=True, stars=False, plot_dir=None, save=False, galaxy="all",
                           fig_size=(6, 4), interval=1, x_lim=None, y_lim=None, dpi=400):
        """
        Plots the evolution of 12 + log(X/H) abundance ratios for specified elements over time.

        Parameters
        ----------
        elements : list of str
            List of element symbols to plot. "He", "N", "O", and "Fe" are supported, though I don't know why you would
            want to plot log(He/H).
        gas : bool, optional
            If True, include gas abundances (default: True).
        stars : bool, optional
            If True, include stellar abundances (default: False).
        plot_dir : str or None, optional
            Directory path to save the plot if `save` is True (default: None).
        save : bool, optional
            If True, saves the plot to `plot_dir`; otherwise, displays the plot (default: False).
        galaxy : str, optional
            Name of the galaxy to analyze, or "all" for all galaxies (default: "all").
        fig_size : tuple, optional
            Figure size in inches (width, height) (default: (6, 4)).
        interval : int, optional
            Number of timesteps to average over for each plotted point (default: 1).
        x_lim : tuple or None, optional
            Limits for the x-axis as (xmin, xmax) (default: None).
        y_lim : tuple or None, optional
            Limits for the y-axis as (ymin, ymax) (default: None).
        dpi : int, optional
            Resolution of the plot in dots per inch.
        """

        plt.clf()
        fig = plt.figure(figsize=fig_size, dpi=dpi)

        if gas == False and stars == False:
            print("You need to use at least one of the following:\n" \
            "gas, stars")

        abbs = []
        if gas:
            abbs.append("gas")
        if stars:
            abbs.append("stars")

        # Create a list of the times around which we take at the average abundance.
        times = []
        temp_times = []
        for time in self.dump_times:
            temp_times.append(time)

            if len(temp_times) == interval:
                times.append(np.mean(temp_times))
                temp_times = []

            if time == (self.len * self.intervals / 1e6) and temp_times:
                times.append(np.mean(temp_times))

        # Create a dictionnary to store our abundances over time.
        dict_12_plus_log = {}
        dict_12_plus_log_temp = {}

        # Initialise the dictionnary keys.
        for element in elements:
            dict_12_plus_log[element], dict_12_plus_log_temp[element] = [], []

        # We create a list of elements to give to the get_element_mass method.
        elements_plus_H = elements + ['H']

        for t in self.timesteps:
            mass_dict = t.get_mean_element_mass(elements_plus_H, gas=gas, stars=stars, galaxy=galaxy)
            aH = mass_dict['H']

            # For every element studied, we take the abundance by dividing the total mass with the atomic mass,
            # and we then take the 12 + log(x/H).
            for element in elements:
                aElement = mass_dict[element] / self.atomic_mass[element]
                dict_12_plus_log_temp[element].append(aElement/aH)

                if len(dict_12_plus_log_temp[element]) == interval:
                    dict_12_plus_log[element].append(12 + np.log10(np.mean(dict_12_plus_log_temp[element])))
                    dict_12_plus_log_temp[element] = []

                if t.idnum == self.idnums[-1] and dict_12_plus_log_temp[element]:
                    dict_12_plus_log[element].append(12 + np.log10(np.mean(dict_12_plus_log_temp[element])))

        for element in elements:
            plt.plot(times, dict_12_plus_log[element], label=f"{element}")

        plt.ylabel("12 + log(X/H) (dex)")
        plt.xlabel("Time (Myrs)")
        plt.legend()

        if x_lim:
            plt.xlim(x_lim)
        if y_lim:
            plt.ylim(y_lim)

        if save:
            plt.savefig(f"{plot_dir}/abondance_over_time_{elements}.png")
            plt.close()
        else:
            plt.show()


    def find_split(self, abb, split, x_lim, y_lim, idnum=0):
        """
        Provides a way of visualy testing which number is being used to split the galaxies for a given abb (different
        for each abb).

        When the initial conditions of a simulation with two galaxies are made, particles from galaxy A will have ids
        under 'x' while particles from galaxy B will have ids over 'x'. This 'x' value is our 'split'. While the GCD+
        code runs, it keeps track of which galaxy a new particle comes from, and it assigns it an id that corresponds
        with its parent galaxy (or at least it does it's best). This is not perfect, but by knowing the 'split', it
        allows us to differentiate between the two galaxies.
        
        Parameters
        ----------
        abb : str
            The abb for which we want to find the split.
        split : int
            A test value for our split.
        x_lim, y_lim : tuples
            The limits for the plot.
        idnum : int, optional
            The idnum of the timstep we would like to use, 0 by default.
        """
        t = self.timesteps[idnum]

        x = t.particle_centered_positions[abb][:,0]
        y = t.particle_centered_positions[abb][:,1]

        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)

        ax1 = axes[0]
        ax1.set_aspect('equal')
        ax1.plot(x[t.particle_ids[abb] < split], y[t.particle_ids[abb] < split], marker='o',
                 markersize=0.5, linestyle='none')
        ax1.set_xlim(x_lim)
        ax1.set_ylim(y_lim)


        ax2 = axes[1]
        ax2.set_aspect('equal')
        ax2.plot(x[t.particle_ids[abb] >= split], y[t.particle_ids[abb] >= split], marker='o',
                 markersize=0.5, linestyle='none')
        ax2.set_xlim(x_lim)
        ax2.set_ylim(y_lim)

        plt.show()


    def assign_stars_from_gas(self, mask1, mask2):

        gas_mask1 = self.get_mask(galaxy='gal1', abb='gas')
        gas_mask2 = self.get_mask(galaxy='gal2', abb='gas')

        star_pos = self.particle_positions['stars']
        gas_pos  = self.particle_positions['gas']

        unassigned = ~(mask1 | mask2)

        tree = cKDTree(gas_pos)
        _, idx = tree.query(star_pos[unassigned])

        mask1[unassigned] = gas_mask1[idx]
        mask2[unassigned] = gas_mask2[idx]

        return mask1, mask2


    def split_all_timesteps(self, output_dir_1, output_dir_2, abb):
        """splits the particles of all timesteps into three different files, one for each galaxy, based on get_mask(). 
        The output files are saved in the specified directories. The only columns that are saved into these new files are teh ones used in read_data in the timestep objet,
          if read_data is changed, this code must be changed as well, otherwise running read_data won't work on the new files."""
        os.makedirs(output_dir_1, exist_ok=True)# output for galaxy 1
        os.makedirs(output_dir_2, exist_ok=True)# output for galaxy 2

        for i in tqdm(range(len(self.timesteps))):
            self.read_one_data(i)
            ts = self.timesteps[i]
            if abb == "stars" or abb == "s":
                name = f"s{ts.idnum:06d}"
            
            if abb == "gas" or abb == "g":
                name = f"g{ts.idnum:06d}"
            
            if abb == "dark" or abb == "d":
                name = f"d{ts.idnum:06d}"   
            # if for some reason you want to split feedback i guess
            if abb == "feed" or abb == "f" or abb == "feedback" or abb == "fdg":
                name = f"fdg{ts.idnum:06d}"

            mask1 = ts.get_mask(galaxy='gal1', abb=abb)
            mask2 = ts.get_mask(galaxy='gal2', abb=abb)

            datasplit = np.column_stack((
                ts.particle_positions[abb][:, 0]/100,  # x
                ts.particle_positions[abb][:, 1]/100,  # y
                ts.particle_positions[abb][:, 2]/100,  # z
                ts.particle_velocities[abb][:, 0]/207.4 , # vx
                ts.particle_velocities[abb][:, 1]/207.4, # vy
                ts.particle_velocities[abb][:, 2]/207.4, # vz
                ts.particle_masses[abb]/1e12,          # mass

            ))
            if abb == "stars":
                mask1, mask2 = ts.assign_stars_from_gas(mask1, mask2)
                datasplit = np.column_stack((datasplit, ts.particle_masses[abb]/1e12, ts.particle_mHe[abb], ts.particle_mC[abb], ts.particle_mN[abb],
                                        ts.particle_mO[abb], ts.particle_mNe[abb], ts.particle_mMg[abb],
                                        ts.particle_mSi[abb], ts.particle_mFe[abb], ts.particle_mZ[abb], np.zeros_like(ts.particle_ids[abb]), np.zeros_like(ts.particle_ids[abb]), ts.particle_ids[abb]))
                
            if abb == "gas":
                datasplit = np.column_stack((datasplit, np.zeros_like(ts.particle_ids[abb]), np.zeros_like(ts.particle_ids[abb]), ts.particle_mHe[abb], ts.particle_mC[abb], ts.particle_mN[abb],
                                        ts.particle_mO[abb], ts.particle_mNe[abb], ts.particle_mMg[abb],
                                        ts.particle_mSi[abb], ts.particle_mFe[abb], ts.particle_mZ[abb], ts.particle_ids[abb]))

            gal1 = datasplit[mask1]
            gal2 = datasplit[mask2]

            np.savetxt(os.path.join(output_dir_1, f"{name}"), gal1)
            np.savetxt(os.path.join(output_dir_2, f"{name}"), gal2)

class Timestep(Simulation):
    def __init__(self, path, idnum, timestamp, n_galaxies, split, abbs, fill, dt=5e6):
        self.path = path
        self.idnum = idnum
        self.time = timestamp
        self.n_galaxies = n_galaxies
        self.split = split
        self.abbs = abbs
        self.dt = dt

        self.data_files = []
        for abb in self.abbs:
            self.data_files.append(f"{self.path}/{abb[0]}{str(self.idnum).zfill(fill)}")

        self.particle_ids = None
        self.particle_labels = None

        self.particle_positions = None # [kpc]
        self.particle_velocities = None # [km/s]
        self.particle_masses = None # [solar mass]

        self.particle_mHe = None
        self.particle_mN = None
        self.particle_mO = None
        self.particle_mFe = None
        self.particle_mZ = None

        self.sfr = None

        self.galaxies_centers = None
        self.galaxies_velocities = None
        self.particle_centered_positions = None
        self.particle_centered_velocities = None


    def __eq__(self, other):
        if isinstance(other, Timestep) or isinstance(other, InitialConditions):
            return self.idnum == other.idnum
        return False


    def read_data(self, initial_conditions=None):
        """
        Read particle data from simulation files.

        Parameters
        ----------
        initial_conditions : InitialConditions, optional
            The initial conditions of the simulation, used for labeling particles.
        """
        # Initialize dictionaries to hold particle positions, masses, and velocities
        self.particle_positions = {}
        self.particle_velocities = {}
        self.particle_masses = {}
        self.particle_ids = {}

        # Initialize dictionaries to hold particle composition data
        self.particle_mHe = {} # 7
        self.particle_mC = {} # 8
        self.particle_mN = {} # 9
        self.particle_mO = {} # 10
        self.particle_mNe = {} # 11
        self.particle_mMg = {} # 12
        self.particle_mSi = {} # 13
        self.particle_mFe = {} # 14
        self.particle_mZ = {} # 15

        # Read the data file
        for i, input_file in enumerate(self.data_files):

            # We get a string for the file name (e.g. s007).
            input_name = input_file.split("/")[-1]

            if input_name[0] == "s":
                abb = "stars"
            elif input_name[0] == "g":
                abb = "gas"
            elif input_name[0] == "d":
                abb = "dark"
            elif input_name[0] == "f":
                abb = "feed"
            
            # If we have new GCD+ output files.
            if len(input_file.split("/")[-1]) < 5:
                # Since the dark matter files have a different structure, we cannot proceed the same way for dark
                # matter and baryons.
                if abb != "dark":
                    # To save time and memory, we only read the columns we need (see documentation)
                    # Therefore the indices of data we call do not correspond to the indices in the file.
                    try:
                        data = np.loadtxt(input_file, unpack=True, dtype=float,
                                        usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 25))
                    except FileNotFoundError:
                        # In case we're looking at detilted data.
                        data = np.loadtxt(input_file+"r", unpack=True, dtype=float,
                                        usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 25))
                    self.particle_positions[abb] = np.array(list(zip(data[0], data[1], data[2])))
                    self.particle_velocities[abb] = np.array(list(zip(data[3], data[4], data[5])))
                    self.particle_masses[abb] = data[6]
                    self.particle_ids[abb] = data[16]

                    self.particle_mHe[abb] = data[7]
                    self.particle_mC[abb] = data[8]
                    self.particle_mN[abb] = data[9]
                    self.particle_mO[abb] = data[10]
                    self.particle_mNe[abb] = data[11]
                    self.particle_mMg[abb] = data[12]
                    self.particle_mSi[abb] = data[13]
                    self.particle_mFe[abb] = data[14]
                    self.particle_mZ[abb] = data[15]
                else:
                    # To save time and memory, we only read the columns we need (see documentation)
                    # Therefore the indices of data we call do not correspond to the indices in the file.
                    data = np.loadtxt(input_file, unpack=True, dtype=float,
                                    usecols=(0, 1, 2, 3, 4, 5, 6, 10))
                    self.particle_positions[abb] = np.array(list(zip(data[0], data[1], data[2])))
                    self.particle_velocities[abb] = np.array(list(zip(data[3], data[4], data[5])))
                    self.particle_masses[abb] = data[6]
                    self.particle_ids[abb] = data[7]

            # If we have old GCD+ imput files, we must use wildly different settings for how we read the file.
            else:
                # different particles behave differently in old GCD+
                if abb == "stars":
                    # To save time and memory, we only read the columns we need (ask Hugo if you need to see
                    # documentation). Therefore the indices of data we call do not correspond to the indices in the
                    # file.
                    try:
                        data = np.loadtxt(input_file, unpack=True, dtype=float,
                                        usecols=(0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19))
                    except FileNotFoundError:
                        # In case we're looking at detilted data.
                        data = np.loadtxt(input_file+"r", unpack=True, dtype=float,
                                        usecols=(0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19))
                    # For some reason, old GCD+ files uses units of 100 kpc instead of straight kpc.
                    self.particle_positions[abb] = np.array(list(zip(data[0], data[1], data[2]))) * 100
                    # Old GCD+ uses unites of 207.4 km/S
                    self.particle_velocities[abb] = np.array(list(zip(data[3], data[4], data[5]))) * 207.4
                    # For some FUCKING reason, old GCD+ uses units of 1e12 solar masses, only for total mass.
                    self.particle_masses[abb] = data[6] * 1e12
                    self.particle_ids[abb] = data[16]

                    self.particle_mHe[abb] = data[7]
                    self.particle_mC[abb] = data[8]
                    self.particle_mN[abb] = data[9]
                    self.particle_mO[abb] = data[10]
                    self.particle_mNe[abb] = data[11]
                    self.particle_mMg[abb] = data[12]
                    self.particle_mSi[abb] = data[13]
                    self.particle_mFe[abb] = data[14]
                    self.particle_mZ[abb] = data[15]

                elif abb == "gas":
                    try:
                        data = np.loadtxt(input_file, unpack=True, dtype=float,
                                        usecols=(0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18))
                    except FileNotFoundError:
                        # In case we're looking at detilted data.
                        data = np.loadtxt(input_file+"r", unpack=True, dtype=float,
                                        usecols=(0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18))
                    # For some reason, old GCD+ files uses units of 100 kpc instead of straight kpc.
                    self.particle_positions[abb] = np.array(list(zip(data[0], data[1], data[2]))) * 100
                    # Old GCD+ uses unites of 207.4 km/S
                    self.particle_velocities[abb] = np.array(list(zip(data[3], data[4], data[5]))) * 207.4
                    # For some FUCKING reason, old GCD+ uses units of 1e12 solar masses, only for total mass.
                    self.particle_masses[abb] = data[6] * 1e12
                    self.particle_ids[abb] = data[16]

                    self.particle_mHe[abb] = data[7]
                    self.particle_mC[abb] = data[8]
                    self.particle_mN[abb] = data[9]
                    self.particle_mO[abb] = data[10]
                    self.particle_mNe[abb] = data[11]
                    self.particle_mMg[abb] = data[12]
                    self.particle_mSi[abb] = data[13]
                    self.particle_mFe[abb] = data[14]
                    self.particle_mZ[abb] = data[15]

                # I am unsure of how old output files are made, this might not work.
                elif abb == 'dark':
                    # To save time and memory, we only read the columns we need (ask Hugo if you need documentation)
                    # Therefore the indices of data we call do not correspond to the indices in the file.
                    data = np.loadtxt(input_file, unpack=True, dtype=float,
                                    usecols=(0, 1, 2, 3, 4, 5, 6, 10))
                    self.particle_positions[abb] = np.array(list(zip(data[0], data[1], data[2]))) * 100
                    self.particle_velocities[abb] = np.array(list(zip(data[3], data[4], data[5]))) * 207.4
                    self.particle_masses[abb] = data[6] * 1e12
                    self.particle_ids[abb] = data[7]

        # Get the particle labels
        if initial_conditions is not None:
            self.get_particle_labels(initial_conditions.label_map)
            # Adjusts the origin (center largest galaxy)
            self.adjust_center()
        elif self.split is not None:
            self.get_particle_labels()
            # Adjusts the origin (center largest galaxy)
            self.adjust_center()


    def rotate_particles(self, axis, angle_degrees):
        """
        Rotate particle positions and velocities around a specified axis.
        DOES NOT WORK FOR FEEDBACK.

        Parameters
        ----------
        axis : str
            The axis to rotate around ('x', 'y', or 'z').
        angle_degrees : float
            The angle of rotation in degrees.
        """
        # Convert angle to radians
        angle = np.radians(angle_degrees)

        # Define the rotation matrix based on the specified axis
        if axis == 'x':
            R = np.array([[1, 0, 0],
                          [0, np.cos(angle), -np.sin(angle)],
                          [0, np.sin(angle), np.cos(angle)]])
        elif axis == 'y':
            R = np.array([[np.cos(angle), 0, np.sin(angle)],
                          [0, 1, 0],
                          [-np.sin(angle), 0, np.cos(angle)]])
        elif axis == 'z':
            R = np.array([[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]])
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")

        for abb in self.abbs:
            # Apply rotation to particle positions and velocities
            self.particle_positions[abb] = np.dot(self.particle_positions[abb], R)
            self.particle_velocities[abb] = np.dot(self.particle_velocities[abb], R)

            # Apply rotation to centered positions and velocities
            if self.particle_centered_positions is not None and self.particle_centered_velocities is not None:
                self.particle_centered_positions[abb] = np.dot(self.particle_centered_positions[abb], R)
                self.particle_centered_velocities[abb] = np.dot(self.particle_centered_velocities[abb], R)


    def get_particle_labels(self, initial_label_map=None):
        """
        Assign labels to particles based either on the initial conditions or on the split numbers.

        Parameters
        ----------
        initial_label_map : dict, optional
            A mapping of particle IDs to their labels.
            If this is None, the labels will be assigned using split numbers.

        Returns
        -------
        dict
            A dictionary of particle labels for each particle type.
        """
        if initial_label_map is not None:
            self.particle_labels = {}
            for abb in self.abbs:
                if abb == "feed":
                    continue
                self.particle_labels[abb] = np.array([initial_label_map[abb].get(id, 0) for id in self.particle_ids[abb]])
            return self.particle_labels

        else:
            self.particle_labels = {}
            for abb in self.abbs:
                self.particle_labels[abb] = np.ones(self.particle_ids[abb].shape)
                self.particle_labels[abb][self.particle_ids[abb] >= self.split[abb]] = 2
                self.particle_labels[abb] = self.particle_labels[abb].astype(int)
            return self.particle_labels


    def get_mask(self, galaxy, abb, max_dist=np.inf):
        """
        Generate a mask for selecting particles based on their labels and distance.

        Parameters
        ----------
        option : str
            The selection option ('gal1', 'gal2', 'all', or 'noise').
        abb : str
            The particle type ('stars', 'gas', 'dark' or 'feed').
        max_dist : float, optional
            The maximum distance from the galaxy center to include particles.

        Returns
        -------
        np.ndarray
            A boolean mask for the selected particles.
        """
        if galaxy == 'gal1':
            center = self.galaxies_centers[0]
        elif galaxy == 'gal2':
            center = self.galaxies_centers[1]
        elif galaxy == 'all':
            center = self.galaxies_centers[0]
        elif galaxy == "noise":
            center = self.galaxies_centers[0]
        else:
            raise ValueError("Option must be 'gal1', 'gal2', 'all' or 'noise'.")

        if abb != 'feed':
            if galaxy == 'gal1':
                mask = self.particle_labels[abb] == 1
            elif galaxy == 'gal2':
                mask = self.particle_labels[abb] == 2
            elif galaxy == 'all':
                mask = self.particle_labels[abb] >= 0
            elif galaxy == "noise":
                mask = self.particle_labels[abb] == 0

            sqr_dist = np.sum((self.particle_positions[abb] - center)**2, axis=1)
            return mask & (sqr_dist <= max_dist**2)
        else:
            sqr_dist = np.sum((self.particle_positions[abb] - center)**2, axis=1)
            return (sqr_dist <= max_dist**2)


    def adjust_center(self, galaxy='gal1', abb='stars'):
        """
        Adjust the origin of the simulation to the center of a specified galaxy.

        Parameters
        ----------
        galaxy : str, optional
            The galaxy to center on ('gal1', 'gal2', or 'all').
        abb : str, optional
            The particle type used to determine the center of the galaxy.
        """
        if galaxy == 'all':
            galaxy = 'gal1'
        galaxy = int(galaxy[-1]) - 1

        if abb not in self.abbs:
            abb = self.abbs[0]

        # Get the centers of galaxies
        self.galaxies_centers = []
        self.galaxies_velocities = []
        for n in range(self.n_galaxies):
            self.galaxies_centers.append(np.mean(self.particle_positions[abb][self.particle_labels[abb] == n + 1],
                                                 axis=0))
            self.galaxies_velocities.append(np.mean(self.particle_velocities[abb][self.particle_labels[abb] == n + 1],
                                                    axis=0))

        # Re-center all particle positions so that the largest cluster is at the origin.

        # Using deepcopy makes a copy of the dictionnary AND of the elements in it.
        self.particle_centered_positions = copy.deepcopy(self.particle_positions)
        self.particle_centered_velocities = copy.deepcopy(self.particle_velocities)

        for abb in self.abbs:
            # This is to deal with the stupid feedback particles that don't always exist so the array is sometimes of
            # length 1 and it ruins everything else.
            if len(self.particle_centered_positions[abb].shape) == 1:
                pass
            else:
                self.particle_centered_positions[abb][:, 0] -= self.galaxies_centers[galaxy][0]
                self.particle_centered_positions[abb][:, 1] -= self.galaxies_centers[galaxy][1]
                self.particle_centered_positions[abb][:, 2] -= self.galaxies_centers[galaxy][2]

                self.particle_centered_velocities[abb][:, 0] -= self.galaxies_velocities[galaxy][0]
                self.particle_centered_velocities[abb][:, 1] -= self.galaxies_velocities[galaxy][1]
                self.particle_centered_velocities[abb][:, 2] -= self.galaxies_velocities[galaxy][2]


    def plot_scatter_density(self, abb, plot_dir=None, save=False, fig_size=(6, 6), x_lim=(-30, 30), y_lim=(-30, 30),
                             z_lim=(-30, 30), dpi=400, galaxy='all', plan='xy', max_dist=np.inf):
        """
        Plots a map of particle positions.

        Parameters
        ----------
        abb : str
            The particle type ('stars' or 'gas').
        plot_dir : str, optional
            Directory to save the plot if `save` is True.
        save : bool, optional
            If True, save the plot to a file. Otherwise, display it.
        fig_size : tuple, optional
            Size of the figure in inches (width, height).
        x_lim, y_lim, z_lim : tuple, optional
            Limits for the spatial axes in kiloparsecs.
        dpi : int, optional
            Resolution of the plot in dots per inch.
        galaxy : str, optional
            The galaxy to focus on ('gal1', 'gal2', or 'all').
        plan : {'xy', 'zy', 'xz'}, optional
            Projection plane for the SFR map.
        max_dist : float, optional
            Maximum distance from the galaxy center to include particles.
        """
        if abb in self.abbs:
            pass
        elif abb == 's':
            abb = 'stars'
        elif abb == 'g':
            abb = 'gas'
        elif abb == 'd':
            abb = 'dark'
        else:
            raise ValueError("abb must be 's', 'stars', 'g', 'gas', 'd' or 'dark'.")
        
        mask = self.get_mask(galaxy, abb, max_dist=max_dist)
        self.adjust_center(galaxy)

        # Depending on the plane, it'll plot different things. We also change the definitions of the limits if needed.
        if plan == 'xy':
            zero, one = 0, 1
        elif plan == 'zy':
            zero, one = 2, 1
            x_lim = z_lim
        elif plan == 'xz':
            zero, one = 0, 2
            y_lim = z_lim
        else:
            raise ValueError("'plan' must be either 'xy', 'zy' or 'xz'")

        # Extracting particle positions and making a binary array of particle positions.
        x = self.particle_centered_positions[abb][mask][:, zero]
        y = self.particle_centered_positions[abb][mask][:, one]

        # Plotting.
        plt.clf()
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        grid = ImageGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0.3, cbar_mode=None)

        fig.suptitle(f'T = {int(self.time)} Myrs')

        ax1 = grid[0]
        ax1.set_aspect('equal')
        ax1.set_xlim(x_lim)
        ax1.set_ylim(y_lim)
        im1 = ax1.plot(x, y, marker='o', linestyle='none', color='red', markersize=0.5)
        ax1.set_xlabel(f'{plan[0]} [kpc]')
        ax1.set_ylabel(f'{plan[1]} [kpc]')

        if save:
            plt.savefig(f"{plot_dir}/{abb[0]}_{str(self.idnum).zfill(3)}_pos.png")
            plt.close()
        else:
            plt.show()


    def plot_density_distribution(self, abb, plot_dir=None, save=False, cmap='hot', nbins=200, colorbar=True,
                                  fig_size=(4, 6), x_lim=(-30, 30), y_lim=(-30, 30), z_lim=(-30, 30), vmin=1, vmax=3, dpi=400,
                                  galaxy='all', max_dist=np.inf, plan='xy', multiplot=False, make_plot=True):
        """
        Plot the density distribution of particles from the stored positions and masses.

        Parameters
        ----------
        abb : str
            The particle type ('stars' or 'gas').
        plot_dir : str, optional
            Directory to save the plot if 'save' is True.
        save : bool, optional
            If True, save the plot to a file. Otherwise, display it.
        cmap : str, optional
            The colormap to use for the plot.
        nbins : int, optional
            Number of bins for the density calculation.
        colorbar : bool, optional
            If True, include a colorbar in the plot.
        fig_size : tuple, optional
            Size of the figure in inches (width, height).
        x_lim, y_lim, z_lim : tuple, optional
            Limits for the spatial axes in kiloparsecs.
        vmin : float, optional
            Minimum value for the color scale.
        vmax : float, optional
            Maximum value for the color scale.
        dpi : int, optional
            Resolution of the plot in dots per inch.
        galaxy : str, optional
            The galaxy to focus on ('gal1', 'gal2', or 'all').
        max_dist : float, optional
            Maximum distance from the galaxy center to include particles.
        plan : {'xy', 'zy', 'xz'}, optional
            Projection plane for the SFR map.
        multiplot : bool, optional
            Will plot the view from 3 planes if True, will only plot the view from the xy plane if False.
        make_plot : bool, optional
            If True, generates and displays/saves the plot. If False, returns the histogram data.

        Returns
        -------
        dict
            The density and the bin edges for all three planes (if "make_plot" is false).
        """

        if abb in self.abbs:
            pass
        elif abb == 's':
            abb = 'stars'
        elif abb == 'g':
            abb = 'gas'
        elif abb == 'd':
            abb = 'dark'
        elif abb == 'f' or abb == 'feed':
            raise ValueError("This method does not work for feedback. Please use plot_feedback_density instead.")
        else:
            raise ValueError("abb must be 's', 'stars', 'g', 'gas', 'd', 'dark', 'f' or 'feed'.")

        mask = self.get_mask(galaxy, abb, max_dist=max_dist)

        # Extracting particle positions
        x = self.particle_centered_positions[abb][mask][:, 0]
        y = self.particle_centered_positions[abb][mask][:, 1]
        z = self.particle_centered_positions[abb][mask][:, 2]
        #x = self.particle_positions[abb][mask][:, 0]
        #y = self.particle_positions[abb][mask][:, 1]
        #z = self.particle_positions[abb][mask][:, 2]
        
        # Define bins for density plots
        x_len = x_lim[1] - x_lim[0]
        y_len = y_lim[1] - y_lim[0]
        z_len = z_lim[1] - z_lim[0]
        axe_max = min(x_len, y_len, z_len)

        xy_bins = [int((x_len/axe_max)*nbins), int((y_len/axe_max)*nbins)]
        zy_bins = [int((z_len/axe_max)*nbins), int((y_len/axe_max)*nbins)]
        xz_bins = [int((x_len/axe_max)*nbins), int((z_len/axe_max)*nbins)]

        # Calculate the area of a pixel in parsec^2.
        resolution_element_area = (1.e3 * (axe_max) / nbins)**2

        # Compute histogram data for density plots
        mass_in_solar = self.particle_masses[abb][0]
        rho_xy, xedges_xy, yedges_xy = np.histogram2d(x, y, bins=xy_bins, range=[x_lim, y_lim])
        rho_zy, zedges_zy, yedges_zy = np.histogram2d(z, y, bins=zy_bins, range=[z_lim, y_lim])
        rho_xz, xedges_xz, zedges_xz = np.histogram2d(x, z, bins=xz_bins, range=[x_lim, z_lim])

        # If we don't plot things directly, we simply return the histogram data in a dictionnary.
        if not make_plot:
            return {'xy': (np.log10(rho_xy.T * mass_in_solar / resolution_element_area), xedges_xy, yedges_xy),
                    'zy': (np.log10(rho_zy.T * mass_in_solar / resolution_element_area), zedges_zy, yedges_zy),
                    'xz': (np.log10(rho_xz.T * mass_in_solar / resolution_element_area), xedges_xz, zedges_xz)}

        # Plotting all 3 planes.
        if multiplot:
            # Plotting the results
            plt.clf()
            fig = plt.figure(figsize=fig_size, dpi=dpi)
            if colorbar:
                grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.3, cbar_location="bottom",
                            cbar_mode="single", cbar_size="5%", cbar_pad=0.4)
            else:
                grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.3, cbar_mode=None)

            fig.suptitle(f'T = {int(self.time)} Myrs')

            # Density plot xy-plane
            ax1 = grid[2]
            im1 = ax1.pcolormesh(xedges_xy, yedges_xy, np.log10(rho_xy.T * mass_in_solar / resolution_element_area),
                                cmap=cmap, vmin=vmin, vmax=vmax)
            ax1.set_aspect('equal')
            ax1.set_ylabel('y [kpc]')
            ax1.set_xlabel('x [kpc]')
            ax1.set_xlim(x_lim)
            ax1.set_ylim(y_lim)
            ax1.tick_params(axis='both', direction='in', left=True, right=True, bottom=True, top=True)

            # Density plot xz-plane
            ax3 = grid[0]
            im3 = ax3.pcolormesh(xedges_xz, zedges_xz, np.log10(rho_xz.T * mass_in_solar / resolution_element_area),
                                cmap=cmap, vmin=vmin, vmax=vmax)
            ax3.set_aspect('equal')
            ax3.set_ylim(z_lim)
            ax3.set_xlabel('x [kpc]')
            ax3.set_ylabel('z [kpc]')
            ax3.tick_params(axis='both', direction='in', left=True, right=True, bottom=True, top=True)

            # Empty subplot
            ax4 = grid[1]
            ax4.set_aspect('equal')
            ax4.set_xlim(z_lim)
            ax4.set_ylim(z_lim)
            ax4.axis('off')

            # Density plot zy-plane
            ax2 = grid[3]
            im2 = ax2.pcolormesh(zedges_zy, yedges_zy, np.log10(rho_zy.T * mass_in_solar / resolution_element_area),
                                cmap=cmap, vmin=vmin, vmax=vmax)
            ax2.set_aspect('equal')
            ax2.set_xlabel('z [kpc]')
            ax2.set_ylabel('y [kpc]')
            ax2.tick_params(axis='both', direction='in', left=True, right=True, bottom=True, top=True)

            # Color bar
            if colorbar:
                cbar = grid.cbar_axes[0].colorbar(im1)
                cbar.set_label(r'log density [M$_{\odot}$/pc$^2$]')

            plt.tight_layout()
        
        # Plotting the chosen plane.
        else:
            if plan == 'xy':
                rho_plot = rho_xy
                xedges_plot, yedges_plot = xedges_xy, yedges_xy
            elif plan == 'zy':
                rho_plot = rho_zy
                xedges_plot, yedges_plot = zedges_zy, yedges_zy
            elif plan == 'xz':
                rho_plot = rho_xz
                xedges_plot, yedges_plot = xedges_xz, zedges_xz

            # Plotting.
            plt.clf()
            fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi, ncols=1)
            ax.set_aspect('equal')
            ax.set_title(f'T = {int(self.time)} Myrs', x=0.02, y=0.98, ha='left', va='top', pad=-2)

            # Chosen plane.
            ax1 = ax
            im1 = ax1.pcolormesh(xedges_plot, yedges_plot, np.log10(rho_plot.T * mass_in_solar / resolution_element_area),
                                 cmap=cmap, vmin=vmin, vmax=vmax)
            ax1.set_xlabel(f'{plan[0]} [kpc]')
            ax1.set_ylabel(f'{plan[1]} [kpc]')
            
            # Colorbar.
            if colorbar:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(im1, cax=cax, orientation='vertical', pad=0.2)
                cbar.set_label(r'log density [M$_{\odot}$/pc$^2$]')

            plt.tight_layout()

        if save:
            plt.savefig(f"{plot_dir}/{abb[0]}_{str(self.idnum).zfill(3)}.png")
            plt.close()
        else:
            plt.show()


    def plot_3d(self, abb, plot_dir=None, nbins=250, jitter_strength=0.2, strength=2, x_lim=(-15, 15), y_lim=(-15, 15),
                z_lim=(-15, 15), cmap='hot', save=True):
        """
        Plots the 3 dimensional position of abb with an interactive plot. To run, it requires the plotly module, which
        itself requires pandas.
        It saves the interactive plots as HTML files that can be viewed in a browser. Saving the files uses a lot of
        storage (~8 Mb/file in my experience).

        Parameters
        ----------
        abb : str
            The abb of the particule we want to look at ('s', 'stars', 'g', 'gas', 'd', 'dark', 'f' or 'feed').
            UNTESTED FOR FEEDBACK
        plot_dir : str, opt
            The directory we want to use to save our figure. It needs to be a string that starts from the very
            beginning of the path. (ex. C:\\User\\Clément ...) The double backslashes are needed unless in an r string.
            I reccomend to use os.getcwd() to get the path of the current directory and to complete the rest by hand.
        nbins : int, opt
            The number of bins we want to use.
        jitter_strength : float, opt
            How much jitter (randomness in the position) we add to the particles in order to break up visual artifacts.
        strength : str, opt
            How much we increase the size of the points.
        x_lim : tuple, opt
            The limits of the x axis.
        y_lim : tuple, opt
            The limits of the y axis.
        z_lim : tuple, opt
            The limits of the z axis.
        cmap : str, opt
            The colormap we want to use.
        save : bool, opt
            Wether we save the file or not. Will display the file if False (untested outside of a Jupyter notebook).
        """
        if abb in self.abbs:
            pass
        elif abb == 's':
            abb = 'stars'
        elif abb == 'g':
            abb = 'g'
        elif abb == 'd':
            abb = 'dark'
        elif abb == 'f':
            abb = 'feed'
        else:
            raise ValueError("abb must be 's', 'stars', 'g', 'gas', 'd', 'dark' 'f' or 'feed'.")

        x = self.particle_centered_positions[abb][:, 0]
        y = self.particle_centered_positions[abb][:, 1]
        z = self.particle_centered_positions[abb][:, 2]

        pos_arr = np.dstack((x, y, z))
        pos_arr = pos_arr.squeeze()

        xyz, binedges, binnumber = stats.binned_statistic_dd(pos_arr, values=None, statistic='count', bins=nbins)

        # Define bins for density plots
        x_len = x_lim[1] - x_lim[0]
        y_len = y_lim[1] - y_lim[0]
        z_len = z_lim[1] - z_lim[0]
        axe_max = min(x_len, y_len, z_len)
        resolution_element_area = (1.e3 * (axe_max) / nbins)**2

        # Convert bin indices to coordinates (bin centers)
        x_edges, y_edges, z_edges = binedges
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2

        # Get 3D coordinates of non-empty bins
        X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
        occupied_coords = np.stack([X[xyz > 0], Y[xyz > 0], Z[xyz > 0]], axis=-1)

        x, y, z = occupied_coords[:, 0], occupied_coords[:, 1], occupied_coords[:, 2]

        # Add a bit of randomness to the particle position to avoid visual artifacts.
        # This only happens since we are binning the data.
        x += np.random.uniform(-jitter_strength, jitter_strength, size=x.shape)
        y += np.random.uniform(-jitter_strength, jitter_strength, size=y.shape)
        z += np.random.uniform(-jitter_strength, jitter_strength, size=z.shape)

        mask = xyz > 0
        counts = xyz[mask]
        # We make the density data logarithmic.
        log_counts = np.log10(counts / resolution_element_area)

        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                           marker=dict(size=counts*strength,
                                                       color=log_counts,
                                                       colorscale=cmap,
                                                       opacity=1))])
        fig.update_layout(
            scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=1, z=1),
                       xaxis_title = 'x', xaxis=dict(range=[x_lim[0], x_lim[1]]),
                       yaxis_title = 'y', yaxis=dict(range=[y_lim[0], y_lim[1]]),
                       zaxis_title = 'z', zaxis=dict(range=[z_lim[0], z_lim[1]])),
            title=f'{abb.capitalize()} position at {self.idnum*self.dt / 1e6} Myrs', margin=dict(l=0, r=0, b=0, t=30))

        if save:
            file_name = f"\\3d_{abb}_position_{str(self.idnum).zfill(3)}.html"
            path = plot_dir + file_name
            fig.write_html(path)
        else:
            fig.show()


    def plot_feedback_density(self, plot_dir=None, save=False, cmap='hot', nbins=500, colorbar=True, x_lim=(-30, 30),
                              y_lim=(-30, 30), dpi=400, vmin=1, vmax=3):
        """
        Plots the density of feedback particles. It has to be a seperate function because feedback is weird and breaks
        everything and I just wanted to see if I could make a map of the feedback density.
        I don't think this method is useful.

        Parameters
        ----------
        plot_dir : str, opt
            The directory where we save the plot.
        save : bool, opt
            Whether we save the figure or not.
        cmap : str, opt
            The colormap we want to use.
        nbins : int, opt
            The number of bins to use in our plot.
        colorbar : bool, opt
            Whether we show the colorbar or not.
        x_lim : tuple, opt
            The limits of the x axis.
        y_lim : tuple, opt
            The limits of the y axis.
        dpi : int, opt
            The resolution of our plot in dots per inches.
        vmin : float, opt
            The minimum value of the colorbar.
        vmax : float, opt
            The maximum value of the colorbar.
        """
        x = self.particle_centered_positions['feed'][:, 0]
        y = self.particle_centered_positions['feed'][:, 1]
        z = self.particle_centered_positions['feed'][:, 2]

        # Define bins for density plots
        x_len = x_lim[1] - x_lim[0]
        y_len = y_lim[1] - y_lim[0]
        axe_max = min(x_len, y_len)
        
        xy_bins = [int((x_len/axe_max)*nbins), int((y_len/axe_max)*nbins)]

        resolution_element_area = (1.e3 * (axe_max) / nbins)**2

        mass_in_solar = self.particle_masses['feed'][0]
        rho_xy, xedges_xy, yedges_xy = np.histogram2d(x, y, bins=xy_bins, range=[x_lim, y_lim])

        plt.clf()
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi, ncols=1)
        ax.set_aspect('equal')

        ax.set_title(f'T = {int(self.time)} Myrs', x=0.02, y=0.98, ha='left', va='top', pad=-2)

        # XY plane
        ax1 = ax
        im1 = ax1.pcolormesh(xedges_xy, yedges_xy, np.log10(rho_xy.T * mass_in_solar / resolution_element_area),
                                cmap=cmap, vmin=vmin, vmax=vmax)
        ax1.set_xlabel('x [kpc]')
        ax1.set_ylabel('y [kpc]')

        # Color bar
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im1, cax=cax, orientation='vertical', pad=0.2)
            cbar.set_label(r'log density [M$_{\odot}$/pc$^2$]')

        plt.tight_layout()

        if save:
            plt.savefig(f"{plot_dir}/f_{str(self.idnum).zfill(3)}.png")
            plt.close()
        else:
            plt.show()


    def plot_velocity(self, abb, vel_type, plot_dir=None, v_sys=0, save=False, cmap='turbo', dpi=400, x_lim=(-55, 55),
                      y_lim=(-55, 55), z_lim=(-55, 55), vmin=-100, vmax=100, nbins=400, colorbar=True,
                      log_density_threshold=None, galaxy='all', max_dist=np.inf, plan='xy', make_plot=False):
        """
        Plots the different velocities (line of sight, radial and circular) of particles.

        Parameters
        ----------
        abb : str
            The particle type ('stars' or 'gas').
        vels : str
            The velocity to plot ('los', 'rad', 'circ' or 'total').
        plot_dir : str, optional
            Directory to save the plot if `save` is True.
        v_sys : float, optional
            Systemic velocity to add to the line-of-sight velocity.
        save : bool, optional
            If True, save the plot to a file. Otherwise, display it.
        cmap : str, optional
            The colormap to use for the plot.
        dpi : int, optional
            Resolution of the plot in dots per inch.
        x_lim, y_lim, z_lim : tuple, optional
            Limits for the spatial axes in kiloparsecs.
        vmin : float, optional
            Minimum value for the color scale.
        vmax : float, optional
            Maximum value for the color scale.
        nbins : int, optional
            Number of bins for the density calculation.
        colorbar : bool, optional
            If True, include a colorbar in the plot.
        log_density_threshold : float, optional
            Threshold for masking low-density regions.
        galaxy : str, optional
            The galaxy to focus on ('gal1', 'gal2', or 'all').
        max_dist : float, optional
            Maximum distance from the galaxy center to include particles.
        plan : {'xy', 'zy', 'xz'}, optional
            Projection plane for the SFR map.
        make_plot : bool, optional
            If True, generates and displays/saves the plot. If False, returns the histogram data.

        Returns
        -------
        h_vel : ndarray
            2D array of velocities (if "make_plot" is False).
        xedges : ndarray
            Bin edges along the x-axis (if "make_plot" is False).
        yedges : ndarray
            Bin edges along the y-axis (if "make_plot" is False).
        """
        if abb in self.abbs:
            pass
        elif abb == 's':
            abb = 'stars'
        elif abb == 'g':
            abb = 'gas'
        elif abb == 'd':
            abb = 'dark'
        elif abb == 'f':
            abb = 'feed'
        else:
            raise ValueError("abb must be 's', 'stars', 'g', 'gas', 'd', 'dark' 'f' or 'feed'.")

        velocities_dict = {'los': 'LOS', 'rad': 'Radial', 'circ': 'Circular', 'total': 'Total'}

        try:
            _ = velocities_dict[vel_type]
        except KeyError:
            raise KeyError("vel_type must be: 'los' and/or 'rad' and/or 'circ'.")

        # Get the mask for the specified galaxy.
        mask = self.get_mask(galaxy, abb, max_dist=max_dist)
        self.adjust_center(galaxy)

        if plan == 'xy':
            zero, one, two = 0, 1, 2
        elif plan == 'zy':
            zero, one, two = 2, 1, 0
            x_lim = z_lim
        elif plan == 'xz':
            zero, one, two = 0, 2, 1
            y_lim = z_lim

        # Extracting particle positions and velocities depending on the plane.
        x = self.particle_centered_positions[abb][mask][:, zero]
        y = self.particle_centered_positions[abb][mask][:, one]
        vx = self.particle_centered_velocities[abb][mask][:, zero]
        vy = self.particle_centered_velocities[abb][mask][:, one]
        vz = self.particle_centered_velocities[abb][mask][:, two] + v_sys

        # Define bins for density plots.
        x_len = x_lim[1] - x_lim[0]
        y_len = y_lim[1] - y_lim[0]
        axe_max = min(x_len, y_len)

        resolution_element_area = (1.e3 * (axe_max) / nbins)**2

        # We get the histogram for the right type of velocity.
        if vel_type == 'los':
            # Calculate 2D histogram with mean velocities
            h_vel, xedges, yedges, _ = stats.binned_statistic_2d(x, y, vz, statistic='mean', bins=nbins,
                                                                range=[x_lim, y_lim])

        if vel_type == 'rad':
            # We find the radial velocity of all particles (source: trust me bro).
            rad_vel = (x * vx + y * vy) / np.sqrt(x**2 + y**2)

            # Making a histogram of the radial velocities.
            h_vel, xedges, yedges, _ = stats.binned_statistic_2d(x, y, rad_vel, statistic='mean', bins=nbins,
                                                            range=[x_lim, y_lim])

        if vel_type == 'circ':
            # We find the circular velocity of all particles (source: trust me bro).
            circ_vel = (x * vy - y * vx) / np.sqrt(x**2 + y**2)

            # Making a histogram of the radial velocities.
            h_vel, xedges, yedges, _ = stats.binned_statistic_2d(x, y, circ_vel, statistic='mean', bins=nbins,
                                                            range=[x_lim, y_lim])

        if vel_type == 'total':
            # Get the total
            v_total = np.sqrt(vx**2 + vy**2 + vz**2)

            # Making a histogram of the radial velocities.
            h_vel, xedges, yedges, _ = stats.binned_statistic_2d(x, y, v_total, statistic='mean', bins=nbins,
                                                            range=[x_lim, y_lim])

        # Masking low density regions
        if log_density_threshold is not None:
            h_counts, _, _, _ = plt.hist2d(x, y, bins=nbins, range=[x_lim, y_lim])
            mask = np.log10(h_counts * self.particle_masses[abb][0] / resolution_element_area) < log_density_threshold
            h_vel[mask] = np.nan

        if make_plot:
            return h_vel, xedges, yedges

        # Plotting
        plt.clf()
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi, ncols=1)
        ax.set_aspect('equal')

        ax.set_title(f'T = {int(self.time)} Myrs', x=0.02, y=0.98, ha='left', va='top', pad=-2)

        # XY plane
        ax1 = ax
        im1 = ax1.pcolormesh(xedges, yedges, h_vel.T, cmap=cmap, vmin=vmin, vmax=vmax)
        ax1.set_xlabel(f'{plan[0]} [kpc]')
        ax1.set_ylabel(f'{plan[1]} [kpc]')

        # Color bar
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im1, cax=cax, orientation='vertical', pad=0.2)
            cbar.set_label(f'{velocities_dict[vel_type]}' + r' velocitiy [km/s]')

        plt.tight_layout()
        if save:
            plt.savefig(f"{plot_dir}/{abb[0]}_{str(self.idnum).zfill(3)}_{vel_type}_vel.png")
            plt.close()
        else:
            plt.show()


    def plot_velocity_dispersion(self, abb, vel_type, plot_dir=None, save=False, cmap='turbo', dpi=400,
                                 x_lim=(-55, 55), y_lim=(-55, 55), z_lim=(-55, 55), vmin=0, vmax=100, nbins=400,
                                 colorbar=True, log_density_threshold=None, galaxy='all', max_dist=np.inf, plan='xy',
                                 make_plot=False):
        """
        Plot the velocity dispersion of particles.

        Parameters
        ----------
        abb : str
            The particle type ('stars' or 'gas').
        vel_type : str
            The velocity for which the dispersion is taken ('los', 'rad', 'circ', 'total').
        plot_dir : str, optional
            Directory to save the plot if `save` is True.
        save : bool, optional
            If True, save the plot to a file. Otherwise, display it.
        cmap : str, optional
            The colormap to use for the plot.
        dpi : int, optional
            Resolution of the plot in dots per inch.
        x_lim, y_lim, z_lim : tuple, optional
            Limits for the spatial axes in kiloparsecs.
        vmin : float, optional
            Minimum value for the color scale. 0 should be used in almost all cases (velocity dispersion starts at 0).
        vmax : float, optional
            Maximum value for the color scale.
        nbins : int, optional
            Number of bins for the density calculation.
        colorbar : bool, optional
            If True, include a colorbar in the plot.
        log_density_threshold : float, optional
            Threshold for masking low-density regions.
        galaxy : str, optional
            The galaxy to focus on ('gal1', 'gal2', or 'all').
        max_dist : float, optional
            Maximum distance from the galaxy center to include particles.
        plan : {'xy', 'zy', 'xz'}, optional
            Projection plane for the SFR map.
        make_plot : bool, optional
            If True, generates and displays/saves the plot. If False, returns the histogram data.

        Returns
        -------
        h_disp : ndarray
            2D array of velocity disperion (if "make_plot" is False).
        xedges : ndarray
            Bin edges along the x-axis (if "make_plot" is False).
        yedges : ndarray
            Bin edges along the y-axis (if "make_plot" is False).
        """
        if abb in self.abbs:
            pass
        elif abb == 's':
            abb = 'stars'
        elif abb == 'g':
            abb = 'gas'
        elif abb == 'd':
            abb = 'dark'
        elif abb == 'f':
            abb = 'feed'
        else:
            raise ValueError("abb must be 's', 'stars', 'g', 'gas', 'd', 'dark' 'f' or 'feed'.")

        velocities_dict = {'los': 'LOS', 'rad': 'Radial', 'circ': 'Circular', 'total': 'Total'}

        try:
            velocities_dict[vel_type]
        except KeyError:
            raise KeyError("vel_type must be: 'los' and/or 'rad' and/or 'circ' and/or 'total'.")

        # Get the mask for the specified galaxy
        mask = self.get_mask(galaxy, abb, max_dist=max_dist)
        self.adjust_center(galaxy)

        if plan == 'xy':
            zero, one, two = 0, 1, 2
        elif plan == 'zy':
            zero, one, two = 2, 1, 0
            x_lim = z_lim
        elif plan == 'xz':
            zero, one, two = 0, 2, 1
            y_lim = z_lim
        
        # Extracting particle positions and velocities
        x = self.particle_centered_positions[abb][mask][:, zero]
        y = self.particle_centered_positions[abb][mask][:, one]
        vx = self.particle_centered_velocities[abb][mask][:, zero]
        vy = self.particle_centered_velocities[abb][mask][:, one]
        vz = self.particle_centered_velocities[abb][mask][:, two]

        # Define bins for density plots
        x_len = x_lim[1] - x_lim[0]
        y_len = y_lim[1] - y_lim[0]
        axe_max = min(x_len, y_len)

        resolution_element_area = (1.e3 * (axe_max) / nbins)**2

        # We get the histogram for the right type of velocity.
        if vel_type == 'los':
            # Calculate 2D histogram with mean velocities
            h_disp, xedges, yedges, _ = stats.binned_statistic_2d(x, y, vz, statistic='std', bins=nbins,
                                                                range=[x_lim, y_lim])

        if vel_type == 'rad':
            # We find the radial velocity of all particles (source: trust me bro).
            rad_vel = (x * vx + y * vy) / np.sqrt(x**2 + y**2)

            # Making a histogram of the radial velocities.
            h_disp, xedges, yedges, _ = stats.binned_statistic_2d(x, y, rad_vel, statistic='std', bins=nbins,
                                                            range=[x_lim, y_lim])

        if vel_type == 'circ':
            # We find the circular velocity of all particles (source: trust me bro).
            circ_vel = (x * vy - y * vx) / np.sqrt(x**2 + y**2)

            # Making a histogram of the radial velocities.
            h_disp, xedges, yedges, _ = stats.binned_statistic_2d(x, y, circ_vel, statistic='std', bins=nbins,
                                                            range=[x_lim, y_lim])

        if vel_type == 'total':
            # Get the total
            v_total = np.sqrt(vx**2 + vy**2 + vz**2)

            # Making a histogram of the radial velocities.
            h_disp, xedges, yedges, _ = stats.binned_statistic_2d(x, y, v_total, statistic='std', bins=nbins,
                                                            range=[x_lim, y_lim])


        # Masking low density regions
        if log_density_threshold is not None:
            h_counts, _, _, _ = plt.hist2d(x, y, bins=nbins, range=[x_lim, y_lim])
            mask = np.log10(h_counts * self.particle_masses[abb][0] / resolution_element_area) < log_density_threshold
            h_disp[mask] = np.nan

        if make_plot:
            return h_disp, xedges, yedges

        # Plotting
        plt.clf()
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi, ncols=1)
        ax.set_aspect('equal')

        ax.set_title(f'T = {int(self.time)} Myrs', x=0.02, y=0.98, ha='left', va='top', pad=-2)

        # XY plane
        ax1 = ax
        im1 = ax1.pcolormesh(xedges, yedges, h_disp.T, cmap=cmap, vmin=vmin, vmax=vmax)
        ax1.set_xlabel(f'{plan[0]} [kpc]')
        ax1.set_ylabel(f'{plan[1]} [kpc]')

        # Color bar
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im1, cax=cax, orientation='vertical', pad=0.2)
            cbar.set_label(f'{velocities_dict[vel_type]}' + r' velocitiy dispersion [km/s]')

        plt.tight_layout()
        if save:
            plt.savefig(f"{plot_dir}/{vel_type}/{abb[0]}_{str(self.idnum).zfill(3)}_{vel_type}_vel_disp.png")
            plt.close()
        else:
            plt.show()


    def plot_log_N(self, abb, plot_dir=None, save=False, cmap='plasma', nbins=400, colorbar=True, x_lim=(-55, 55),
                   y_lim=(-55, 55), z_lim=(-55, 55), vmin=-1.02, vmax=-0.89, galaxy='all', max_dist=np.inf,
                   log_density_threshold=None, rings=False, plan='xy', make_plot=False):
        """
        Plot the logarithmic nitrogen-to-oxygen ratio of particles.

        Parameters
        ----------
        abb : str
            The particle type ('stars' or 'gas').
        plot_dir : str, optional
            Directory to save the plot if `save` is True.
        save : bool, optional
            If True, save the plot to a file. Otherwise, display it.
        cmap : str, optional
            The colormap to use for the plot.
        nbins : int, optional
            Number of bins for the density calculation.
        colorbar : bool, optional
            If True, include a colorbar in the plot.
        x_lim, y_lim, z_lim : tuple, optional
            Limits for the spatial axes in kiloparsecs.
        vmin : float, optional
            Minimum value for the color scale.
        vmax : float, optional
            Maximum value for the color scale.
        galaxy : str, optional
            The galaxy to focus on ('gal1', 'gal2', or 'all').
        max_dist : float, optional
            Maximum distance from the galaxy center to include particles.
        log_density_threshold : float, optional
            Threshold for masking low-density regions.
        rings : bool, opt
            Whether we plot rings around the center to help visualize the distance from the center.
        plan : {'xy', 'zy', 'xz'}, optional
            Projection plane for the SFR map.
        make_plot : bool, optional
            If True, generates and displays/saves the plot. If False, returns the histogram data.

        Returns
        -------
        mean_abundances : ndarray
            2D array of the mean abundance in each bin (if "make_plot" is False).
        xedges : ndarray
            Bin edges along the x-axis (if "make_plot" is False).
        yedges : ndarray
            Bin edges along the y-axis (if "make_plot" is False).
            
        """
        if abb in self.abbs:
            pass
        elif abb == 's':
            abb = 'stars'
        elif abb == 'g':
            abb = 'gas'
        else:
            raise ValueError("abb must be 's', 'stars', 'g' or 'gas'.")

        # Get the mask for the specified galaxy
        mask = self.get_mask(galaxy, abb, max_dist=max_dist)
        self.adjust_center(galaxy)

        if plan == 'xy':
            zero, one = 0, 1
        elif plan == 'zy':
            zero, one = 2, 1
            x_lim = z_lim
        elif plan == 'xz':
            zero, one = 0, 2
            y_lim = z_lim

        # Extracting particle positions and velocities
        x = self.particle_centered_positions[abb][mask][:, zero]
        y = self.particle_centered_positions[abb][mask][:, one]

        # Define bins for density plots
        x_len = x_lim[1] - x_lim[0]
        y_len = y_lim[1] - y_lim[0]
        axe_max = min(x_len, y_len)

        xy_bins = [int((x_len/axe_max)*nbins), int((y_len/axe_max)*nbins)]

        resolution_element_area = (1.e3 * (axe_max) / nbins)**2

        n = self.particle_mN[abb][mask] / 14
        o = self.particle_mO[abb][mask] / 16

        abondance = np.log10(n/o)

        # Calculate 2D histograms with mean velocities
        h_xy, xedges, yedges, _ = plt.hist2d(x, y, bins=xy_bins, range=[x_lim, y_lim], weights=abondance)
        h_counts, _, _, _ = plt.hist2d(x, y, bins=nbins, range=[x_lim, y_lim])
        mean_abundances = h_xy / h_counts

        # Masking low density regions
        if log_density_threshold is not None:
            mask = np.log10(h_counts * self.particle_masses[abb][0] / resolution_element_area) < log_density_threshold
            mean_abundances[mask] = np.nan

        # Return the histogram data if that's what we want.
        if make_plot:
            return mean_abundances, xedges, yedges

        # Plotting
        plt.clf()
        fig, ax = plt.subplots(figsize=(4, 4), dpi=400, ncols=1)
        ax.set_aspect('equal')

        ax.set_title(f'T = {int(self.time)} Myrs', x=0.02, y=0.98, ha='left', va='top', pad=-2)

        # XY plane
        ax1 = ax

        if rings:
            circ1 = plt.Circle((0, 0), np.sqrt(axe_max*0.3403), fill=False)
            ax1.add_artist(circ1)
            circ2 = plt.Circle((0, 0), np.sqrt(axe_max*1.3611), fill=False)
            ax1.add_artist(circ2)
            circ3 = plt.Circle((0, 0), np.sqrt(axe_max*3.0625), fill=False)
            ax1.add_artist(circ3)

        im1 = ax1.pcolormesh(xedges, yedges, mean_abundances.T, cmap=cmap, vmin=vmin, vmax=vmax)
        ax1.set_xlabel(f'{plan[0]} [kpc]')
        ax1.set_ylabel(f'{plan[1]} [kpc]')

        # Color bar
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im1, cax=cax, orientation='vertical', pad=0.1)
            cbar.set_label("log(N/O) (dex)")

        plt.tight_layout()
        if save:
            plt.savefig(f"{plot_dir}/{abb[0]}_{str(self.idnum).zfill(3)}_logN.png")
            plt.close()
        else:
            plt.show()


    def plot_12_plus_log_O(self, abb, plot_dir=None, save=False, cmap='plasma', nbins=400, colorbar=True,
                           x_lim=(-55, 55), y_lim=(-55, 55), z_lim=(-55, 55), vmin=8.54, vmax=8.82, galaxy='all',
                           max_dist=np.inf, log_density_threshold=None, rings=False, plan='xy', make_plot=False):
        """
        Plot the 12 + log(O/H) abundance of particles.

        Parameters
        ----------
        abb : str
            The particle type ('stars' or 'gas').
        plot_dir : str, optional
            Directory to save the plot if `save` is True.
        save : bool, optional
            If True, save the plot to a file. Otherwise, display it.
        cmap : str, optional
            The colormap to use for the plot.
        nbins : int, optional
            Number of bins for the density calculation.
        colorbar : bool, optional
            If True, include a colorbar in the plot.
        x_lim, y_lim, z_lim : tuple, optional
            Limits for the spatial axes in kiloparsecs.
        vmin : float, optional
            Minimum value for the color scale.
        vmax : float, optional
            Maximum value for the color scale.
        galaxy : str, optional
            The galaxy to focus on ('gal1', 'gal2', or 'all').
        max_dist : float, optional
            Maximum distance from the galaxy center to include particles.
        log_density_threshold : float, optional
            Threshold for masking low-density regions.
        rings : bool, opt
            Whether we plot rings around the center to help visualize the distance from the center.
        plan : {'xy', 'zy', 'xz'}, optional
            Projection plane for the SFR map.
        make_plot : bool, optional
            If True, generates and displays/saves the plot. If False, returns the histogram data.

        Returns
        -------
        mean_abundances : ndarray
            2D array of the mean abundance in each bin (if "make_plot" is False).
        xedges : ndarray
            Bin edges along the x-axis (if "make_plot" is False).
        yedges : ndarray
            Bin edges along the y-axis (if "make_plot" is False).
        """
        if abb in self.abbs:
            pass
        elif abb == 's':
            abb = 'stars'
        elif abb == 'g':
            abb = 'gas'
        else:
            raise ValueError("abb must be 's', 'stars', 'g' or 'gas'.")

        # Get the mask for the specified galaxy
        mask = self.get_mask(galaxy, abb, max_dist=max_dist)
        self.adjust_center(galaxy)

        if plan == 'xy':
            zero, one = 0, 1
        elif plan == 'zy':
            zero, one = 2, 1
            x_lim = z_lim
        elif plan == 'xz':
            zero, one = 0, 2
            y_lim = z_lim

        # Extracting particle positions and velocities
        x = self.particle_centered_positions[abb][mask][:, zero]
        y = self.particle_centered_positions[abb][mask][:, one]

        # Define bins for density plots
        x_len = x_lim[1] - x_lim[0]
        y_len = y_lim[1] - y_lim[0]
        axe_max = min(x_len, y_len)

        xy_bins = [int((x_len/axe_max)*nbins), int((y_len/axe_max)*nbins)]

        resolution_element_area = (1.e3 * (axe_max) / nbins)**2

        h = self.particle_masses[abb][mask] - self.particle_mHe[abb][mask] - self.particle_mZ[abb][mask]
        o = self.particle_mO[abb][mask] / 16

        abondance = 12 + np.log10(o/h)

        # Calculate 2D histograms with mean velocities
        h_xy, xedges, yedges, _ = plt.hist2d(x, y, bins=xy_bins, range=[x_lim, y_lim], weights=abondance)
        h_counts, _, _, _ = plt.hist2d(x, y, bins=nbins, range=[x_lim, y_lim])
        mean_abundances = h_xy / h_counts

        # Masking low density regions
        if log_density_threshold is not None:
            mask = np.log10(h_counts * self.particle_masses[abb][0] / resolution_element_area) < log_density_threshold
            mean_abundances[mask] = np.nan

        # Return the histograms if that's what we want.
        if make_plot:
            return mean_abundances, xedges, yedges

        # Plotting
        plt.clf()
        fig, ax = plt.subplots(figsize=(4, 4), dpi=400, ncols=1)
        ax.set_aspect('equal')

        ax.set_title(f'T = {int(self.time)} Myrs', x=0.02, y=0.98, ha='left', va='top', pad=-2)

        # XY plane
        ax1 = ax

        if rings:
            circ1 = plt.Circle((0, 0), np.sqrt(axe_max*0.3403), fill=False)
            ax1.add_artist(circ1)
            circ2 = plt.Circle((0, 0), np.sqrt(axe_max*1.3611), fill=False)
            ax1.add_artist(circ2)
            circ3 = plt.Circle((0, 0), np.sqrt(axe_max*3.0625), fill=False)
            ax1.add_artist(circ3)

        im1 = ax1.pcolormesh(xedges, yedges, mean_abundances.T, cmap=cmap, vmin=vmin, vmax=vmax)
        ax1.set_xlabel(f'{plan[0]} [kpc]')
        ax1.set_ylabel(f'{plan[1]} [kpc]')

        # Color bar
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im1, cax=cax, orientation='vertical', pad=0.1)
            cbar.set_label("12+log(O/H) (dex)")

        plt.tight_layout()
        if save:
            plt.savefig(f"{plot_dir}/{abb[0]}_{str(self.idnum).zfill(3)}_12+logO.png")
            plt.close()
        else:
            plt.show()


    def list_12_plus_log_x_radii(self, abb, element, points, interval, galaxy="all"):
        """
        Returns the list of the 12 + log(x/H) at different radii for the timestep.
        UNTESTED IN ITS CURRENT FORM

        Parameters
        ----------
        abb : str
            The particle type ('stars' or 'gas').
        element : str
            The element we want to study ('He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si' or 'Fe').
        points : int
            The number of points we want in our graph.
        interval : float
            How far from each radius we look in both directions.
        galaxy : str, optional
            The galaxy to focus on ('gal1', 'gal2', or 'all').
        """

        self.adjust_center(galaxy)

        log_list = []

        # Retrieve the atomic mass for the element while checking if the argument for element is correct.

        atomic_mass = {'He': 4, 'C':12, 'N': 14, 'O':16, 'Ne':20, 'Mg':24, 'Si':28, 'Fe':56}

        try:
            atomic_mass = atomic_mass[element]
        except KeyError:
            raise KeyError("element must be 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si' or 'Fe'.")

        for point in range(points):
            # Get the abundance for the different points
            if point == 0:
                mask = self.get_mask(galaxy, abb, max_dist=interval)

                # We find the abundance for a given radius
                aH = self.particle_masses[abb][mask] - self.particle_mHe[abb][mask] - self.particle_mZ[abb][mask]
                
                # Find the total mass of the element
                name = f"particle_m{element}"
                mElement = getattr(self, name)[abb][mask]

                # Find the abundance of the element.
                aElement = mElement / atomic_mass

                abondance = 12 + np.log10(aElement/aH)
                log_list.append(np.mean(abondance))
            else:
                # We make a mask for everything a ±interval distance away from the radius
                mask_inner = ~self.get_mask(galaxy, abb, max_dist=(2*point - 1)*interval)
                mask_outer = self.get_mask(galaxy, abb, max_dist=(2*point + 1)*interval)
                mask = mask_inner * mask_outer

                # We find the abundance for a given radius
                aH = self.particle_masses[abb][mask] - self.particle_mHe[abb][mask] - self.particle_mZ[abb][mask]

                # Find the total mass of the element
                name = f"particle_m{element}"
                mElement = getattr(self, name)[abb][mask]

                # Find the abundance of the element.
                aElement = mElement / atomic_mass

                abondance = 12 + np.log10(aElement/aH)
                log_list.append(np.mean(abondance))
                
        return log_list


    def get_mean_element_mass(self, elements, gas=True, stars=False, galaxy="all"):
        """
        Find the mean mass of specified elements in the simulation, separated by gas and/or stars.

        Parameters
        ----------
        elements : list of str
            List of element symbols for which to compute the mean mass.
        gas : bool, optional
            If True, include gas particles in the calculation (default is True).
        stars : bool, optional
            If True, include star particles in the calculation (default is False).
        galaxy : str, optional
            Specifies which galaxy to consider. Default is "all".
        Returns
        -------
        mass_dict : dict
            Dictionary mapping each element symbol to its mean mass.
        """

        abbs=[]
        if gas:
            abbs.append("gas")
        if stars:
            abbs.append("stars")

        mass_dict = {}

        # Initialise the dictionnary keys we need.
        for element in elements:
            mass_dict[element] = 0

        # Get the mass of a given element for each abb.
        for abb in abbs:
            for element in mass_dict.keys():
                if element == "H":
                    mass_dict[element] += np.mean(self.particle_masses[abb] - self.particle_mHe[abb] - self.particle_mZ[abb])
                else:
                    name = f"particle_m{element}"
                    mass_dict[element] = np.mean(getattr(self, name)[abb])

        return mass_dict


    def calculate_star_formation_rate(self, dt=5e6):
        """
        Calculate the star formation rate (SFR) over a specified time period.

        Parameters
        ----------
        dt : float, optional
            Time interval during which new stars are formed, in years.

        Returns
        -------
        float
            The star formation rate in solar masses per year.
        """
        print("Use Simulation.get_sfr() instead of Timestep.calculate_star_formation_rate()")
        # Iterate over each star's formation age and corresponding mass
        new_stars_mask =  self.particle_formation_yr['stars'] - np.linspace(0, 1e9, 201)[self.idnum] < dt
        current_mass = np.sum(self.particle_masses['stars'][new_stars_mask])
        
        # Calculate the star formation rate in [Msol/yr]
        star_formation_rate = current_mass / dt
        # print('Taux de formation:', star_formation_rate, '\n')
        return star_formation_rate


    def split_mask(self, abb, split):
        """
        Makes a mask of all particles above a certain ID. Can be used to separate two galaxies.
        To look at the particles with an ID below or equal, simply add a "~" before the mask.
        
        Parameters
        ----------
        abb : str
            The particle type we want to analyse.
        split : int
            The ID where we want to split.
        
        Returns
        -------
            A mask of the all the particles with an ID above split."""
        
        try:
            mask = self.particle_ids[abb] > split
        except KeyError:
            raise ValueError("abb must be 's', 'stars', 'g', 'gas', 'd', 'dark', 'f' or 'feed'.")
        
        return mask

class InitialConditions(Timestep):
    def __init__(self, path, n_galaxies=1, abbs=["stars", "gas",'dark'], fill=3):
        """
        Initializes an InitialConditions object.

        The InitialConditions class uses a clustering tool (fit_predict from HDBSCAN) during the first timestep to
        assign a label to each particle depending on which galaxy it comes from. This only works for the particles
        present in the first timestep.

        Parameters
        ----------
        path : str
            The path of the data files for the simulation.
        n_galaxies : int
            The amount of galaxies.
        abbs : list, opt
            A list of the abbs we want to use (not loading in useless abbs increases performance).
        fill : int, opt
            The amount of numbers in the data files. Recent GCD+ output files have 3, but older ones have 6.
        """
        super().__init__(path, idnum=0, timestamp=0, n_galaxies=n_galaxies, split=None, abbs=abbs, fill=fill)
        self.read_data()
        self.particle_labels = self.get_labels()
        self.label_map = self.get_label_map()
        self.adjust_center()


    def get_labels(self):
        """
        Assign labels to particles based on clustering.
        Does not take into account feedback particles (since there are none at the beginning).

        Returns
        -------
        dict
            A dictionary of particle labels for each particle type.
        """

        # Get the total number of particles, except for feedback particles.
        labels_len = 0
        for abb in self.abbs:
            if abb == "feed":
                continue
            labels_len += self.particle_positions[abb].shape[0]

        # Make a variable for the amount of particles in the previous abbs (starts at 0).
        previous_len = 0
        # Create a dictionnary for the different masks we will generate.
        mask_dict = {}
        # Make a position array (for the HDBSCAN clusterer).
        pos = np.array([])

        for abb in self.abbs:
            if abb != "feed":
                # Get the amount of particles in this abb, and then make a mask using the amount of particles before,
                # in the, and after the abb.
                abb_len = self.particle_positions[abb].shape[0]
                mask_dict[abb] = np.concatenate([np.zeros(previous_len),
                                                 np.ones(abb_len),
                                                 np.zeros(labels_len - (previous_len + abb_len))])

                # Update the amount of previous particles.
                previous_len += abb_len

                # Add the positions of particles in the abb to the position array.
                if pos.shape == (0,):
                    pos = self.particle_positions[abb]
                else:
                    pos = np.concatenate([pos, self.particle_positions[abb]])

        clusterer = HDBSCAN(min_cluster_size=20, min_samples=100, cluster_selection_epsilon=10)
        fit = clusterer.fit_predict(pos)

        # Make a dictionnary for the labels of the different particles.
        labels = {}
        for abb in self.abbs:
            if abb == "feed":
                continue
            labels[abb] = fit[mask_dict[abb].astype(bool)]

        # Identify the cluster with the most particles and the cluster with the second most particles.
        unique_labels, counts = np.unique(labels[list(labels.keys())[0]], return_counts=True)
        max = np.argmax(counts)
        min = np.argmin(counts)
        gal1_label = unique_labels[max]
        gal2_label = unique_labels[(unique_labels != gal1_label)]

        new_labels = {}
        for abb in self.abbs:
            if abb == "feed":
                continue
            new_labels[abb] = np.zeros_like(labels[abb], dtype=int)

        for abb in self.abbs:
            if abb == "feed":
                continue
            new_labels[abb][labels[abb] == gal1_label] = 1
            # If there is more than one unique label.
            if len(unique_labels) > 1:
                try:
                    # If there are two unique_labels.
                    new_labels[abb][labels[abb] == gal2_label] = 2
                except ValueError:
                    # If there are three unique_labels.
                    gal2_label = unique_labels[(unique_labels != gal1_label) & (unique_labels != unique_labels[min])]
                    new_labels[abb][labels[abb] == gal2_label] = 2

        return new_labels


    def get_label_map(self):
        """
        Create a mapping of particle IDs to their labels.

        Returns
        -------
        dict
            A dictionary mapping particle IDs to their labels.
        """
        label_map = {}
        for abb in self.abbs:
            if abb == "feed":
                continue
            label_map[abb] = {}
            for id, label in zip(self.particle_ids[abb], self.particle_labels[abb]):
                label_map[abb][id] = label
        return label_map
