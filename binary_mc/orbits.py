# Import necessary libraries
import re, ssl, numpy as np, matplotlib.pyplot as plt, astropy.units as u
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord

# Configure HTTPS context for legacy Python environments
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Override default context for environments that do support HTTPS
    ssl._create_default_https_context = _create_unverified_https_context
    
def kepler(Marr, eccarr):
    """
    Solves Kepler's Equation for eccentric anomaly using Newton-Raphson iteration.

    Args:
        Marr (array): Mean anomalies
        eccarr (array): Eccentricities

    Returns:
        array: Eccentric anomalies
    """
    conv = 1.0e-12  # convergence threshold
    k = 0.85  # initial guess factor

    Earr = Marr + np.sign(np.sin(Marr)) * k * eccarr
    fiarr = Earr - eccarr * np.sin(Earr) - Marr
    convd = np.where(np.abs(fiarr) > conv)[0]
    nd = len(convd)

    while nd > 0:

        M = Marr[convd]
        ecc = eccarr[convd]
        E = Earr[convd]

        fi = E - ecc * np.sin(E) - M
        fip = 1 - ecc * np.cos(E)
        fipp = ecc * np.sin(E)
        fippp = 1 - fip

        d1 = -fi / fip
        d2 = -fi / (fip + d1 * fipp / 2.0)
        d3 = -fi / (fip + d2 * fipp / 2.0 + d2 * d2 * fippp / 6.0)

        E += d3
        Earr[convd] = E
        fiarr = Earr - eccarr * np.sin(Earr) - Marr
        convd = np.abs(fiarr) > conv
        nd = np.sum(convd is True)

    return Earr if Earr.size > 1 else Earr[0]

# Function to return an array of stellar mass samples for a given star
def estimate_mass(idx, ids, M_guesses, N, default=(1.10, 0.05)):
    """
    Estimates the stellar mass for a given source, using (in priority order):
    1. User-supplied mass guesses
    2. The `gorp_mass` relation
    3. A default mass (with Gaussian scatter)

    Args:
        idx (int): Index of the star in the input list (ids)
        ids (list of str): Gaia DR3 source IDs
        M_guesses (list of tuples or None): Optional (mass, mass_err) for each star
        N (int): Number of Monte Carlo samples to generate
        default (tuple): Fallback (mean, std) for mass in solar units if no other info is available

    Returns:
        np.ndarray: Array of N mass samples (in solar masses)
    """

    # Option 1: Use the user-provided mass guess, if available
    if M_guesses is not None and M_guesses[idx][0] is not None:
        return np.random.normal(M_guesses[idx][0], M_guesses[idx][1], N)

    # Option 2: Use the gorp_mass.gaia_posterior tool to estimate mass
    try:
        from gorp_mass import gaia_posterior
    except ImportError:
        print("Unable to use `gorp_mass` photometric estimator. Please install via https://github.com/markgiovinazzi/gorp_masses")
        return np.random.normal(*default, N)
    try:
        posts = gaia_posterior(ids[idx])

        mass = posts[0]['mass']
        mass_err = posts[0]['mass_err']

        if np.isnan(mass) or np.isnan(mass_err):
            print(f"gaia_posterior returned nan values for Gaia DR3 {ids[idx]}. "
                  f"Check the Gaia photometry to see if this source is missing entries. "
                  f"Defaulting to M = N({default[0]}, {default[1]})")
            return np.random.normal(*default, N)

        return np.random.normal(mass, mass_err, N)
    except Exception:
        # Option 3: Fallback to default mass if no guess or posterior available
        print(f"No secondary mass given, and source is invalid for the `gorp_mass` relation. "
              f"Defaulting to M = N({default[0]}, {default[1]}) for Gaia DR3 {ids[idx]}")
        return np.random.normal(*default, N)

def dynamic_log_bins(data, num_bins=1000):
    """
    Returns log-spaced bins from floor(log10(min)) to ceil(log10(max)).
    """
    # Flatten nested lists if needed
    if isinstance(data[0], (list, np.ndarray)):
        data = np.concatenate([d for d in data])
    else:
        data = np.array(data)

    data = data[np.isfinite(data) & (data > 0)]  # filter out bad values

    if data.size == 0:
        raise ValueError("No valid (positive, finite) values in data.")

    min_val = data.min()
    max_val = data.max()

    log_min = np.floor(np.log10(min_val))
    log_max = np.ceil(np.log10(max_val))

    return np.logspace(log_min, log_max, num_bins)
    
def print_binary_diagnostics(periods, rv_slopes, period_bins, slope_bins, gaia_ids=None):
    print("\n--- Binary Orbit Diagnostics ---")

    n = len(periods)
    for i in range(n):
        print()
        if gaia_ids is not None:
            print(f"[Gaia DR3 {gaia_ids[i]}]")

        Ps_array, _ = periods[i]
        slope_array, _ = rv_slopes[i]
        this_periods = np.ravel(Ps_array)
        this_slopes = np.ravel(slope_array)

        # Period histogram
        hist_P, bins_P = np.histogram(this_periods, bins=period_bins)
        most_likely_period = 0.5 * (bins_P[np.argmax(hist_P)] + bins_P[np.argmax(hist_P)+1])
        min_period = np.min(this_periods)

        print(f"Most likely period (approx.): {most_likely_period:.2f} years")
        print(f"Minimum period: {min_period:.2f} years")

        # RV slope histogram
        hist_S, bins_S = np.histogram(this_slopes, bins=slope_bins)
        most_likely_slope = 0.5 * (bins_S[np.argmax(hist_S)] + bins_S[np.argmax(hist_S)+1])
        
        # if most likely RV slope ≥ 1 m/s/yr, report 10th and 1st percentiles (for strong slopes)
        if most_likely_slope >= 1:
        
            rv_10 = np.percentile(this_slopes, 10)
            rv_01 = np.percentile(this_slopes, 1)

            print(f"Most likely RV slope (approx.): {most_likely_slope:.4f} m/s/yr")
            print(f"90% of orbits have RV slope > {rv_10:.4f} m/s/yr")
            print(f"99% of orbits have RV slope > {rv_01:.4f} m/s/yr")
        
        # else, reporting 90th and 99th percentiles will make most sense
        else:
        
            rv_90 = np.percentile(this_slopes, 90)
            rv_99 = np.percentile(this_slopes, 99)

            print(f"Most likely RV slope (approx.): {most_likely_slope:.4f} m/s/yr")
            print(f"90% of orbits have RV slope < {rv_90:.4f} m/s/yr")
            print(f"99% of orbits have RV slope < {rv_99:.4f} m/s/yr")

    print("\n--------------------------------\n")

def binary_mc(ids, N = 10000000, M_guesses = None, save_plot = True):
    """
    Monte Carlo simulation of orbital properties for a primary and its companions using Gaia data.

    Args:
        ids (list or str): List of Gaia DR3 source IDs (first is primary)
        N (int): Number of Monte Carlo samples
        M_guesses (list of tuples): Optional list of (mass, error) for each source
    """
    # Validate ID input
    if isinstance(ids, list):
        N_ids = len(ids)
        ids_str = ', '.join(map(str, ids))
    elif isinstance(ids, str):
        N_ids = 1
        ids_str = ids
    else:
        raise ValueError('Input "ids" must be either \'str\' or list/tuple of \'str\'')

    # Construct Gaia SQL query
    query = f"""
        SELECT g.source_id, g.phot_g_mean_mag, g.parallax, g.parallax_error, g.bp_rp, g.ra, g.dec, g.ra_error, g.dec_error, g.pmra, g.pmdec, g.pmra_error, g.pmdec_error
        FROM gaiadr3.gaia_source as g
        WHERE g.source_id IN ({ids_str})
    """
    
    print('Querying the GAIA Archive...')
    job = Gaia.launch_job(query)
    results = job.get_results()

    if len(results) < N_ids:
        raise ValueError('Not all Gaia IDs were found in the query results.')

    primary = results[0]
    secondaries = results[1:]

    colors = ['darkblue', 'darkorange', 'darkgreen', 'darkred', 'black']
    all_Ps, all_Ks, all_semimajors, pm_plot_data = [], [], [], []
    
    # get mass of the primary before iterating through companion(s)
    M1 = estimate_mass(0, ids, M_guesses, N)

    for idx, secondary in enumerate(secondaries, start = 1):

        # Extract coordinates and proper motion data for primary and secondary
        ra1, ra2 = primary['ra'], secondary['ra']
        dec1, dec2 = primary['dec'], secondary['dec']
        ra_error1, ra_error2 = primary['ra_error'], secondary['ra_error']
        dec_error1, dec_error2 = primary['dec_error'], secondary['dec_error']
        pmra1, pmra2 = primary['pmra'], secondary['pmra']
        pmdec1, pmdec2 = primary['pmdec'], secondary['pmdec']
        pmra_error1, pmra_error2 = primary['pmra_error'], secondary['pmra_error']
        pmdec_error1, pmdec_error2 = primary['pmdec_error'], secondary['pmdec_error']
        plx = primary['parallax']

        # Simulate proper motion differences; 1.37 inflation factor comes from Brandt (2021): The Hipparcos–Gaia Catalog of Accelerations: Gaia EDR3 Edition
        dpmra = np.random.normal(pmra2, pmra_error2 * 1.37, N) - np.random.normal(pmra1, pmra_error1 * 1.37, N)
        dpmdec = np.random.normal(pmdec2, pmdec_error2 * 1.37, N) - np.random.normal(pmdec1, pmdec_error1 * 1.37, N)
        pm_plot_data.append((ra2, dec2, dpmra, dpmdec, idx))

        # Calculate projected separation in AU
        c1 = SkyCoord(ra1 * u.deg, dec1 * u.deg)
        c2 = SkyCoord(ra2 * u.deg, dec2 * u.deg)
        sep = c1.separation(c2).arcsec
        sep_au = sep * 1000. / plx

        # Sample orbital elements from thermal distribution
        m = 2
        e = (np.sqrt(8 * m * np.random.uniform(0, 1, N) + (m - 2)**2) + m - 2) / (2. * m)
        i = np.arccos(np.random.uniform(0, 1, N))
        omega = np.random.uniform(0, 2 * np.pi, N)
        ma = np.random.uniform(0, 2 * np.pi, N)
        E = kepler(ma, e)
        f = 2.0 * np.arctan2((1 + e)**0.5 * np.sin(E / 2.), (1 - e)**0.5 * np.cos(E / 2.))

        # Compute semi-major axis in AU
        r = sep_au / np.sqrt(np.cos(omega + f)**2 + np.sin(omega + f)**2 * np.cos(i)**2)
        semimajors = r * (1 + e * np.cos(f)) / (1 - e**2)

        # get mass of the current companion, or "secondary"
        M2 = estimate_mass(idx, ids, M_guesses, N)

        # Calculate orbital period (P) and RV slope (K)
        Ps = np.sqrt(semimajors**3 / (M1 + M2))
        Ks = 28.435 / Ps**(1. / 3) * (M2 * 1047.47 * np.sin(i)) / (M1 + M2)**(2. / 3) / np.sqrt(1 - e**2)

        # Store results for plotting
        all_Ps.append((Ps, colors[(idx - 1) % len(colors)]))
        all_Ks.append((Ks, colors[(idx - 1) % len(colors)]))
        all_semimajors.append((semimajors, colors[(idx - 1) % len(colors)]))

    # Plot proper motion vectors for all companions
    plt.figure(figsize = (6, 4))
    plt.scatter(0, 0, marker='*', color='red', label = 'Gaia DR3 ' + str(results['source_id'][0]))
    ra1, dec1 = primary['ra'], primary['dec']
    for ra2, dec2, dpmra, dpmdec, idx in pm_plot_data:
        l = 'Gaia DR3 ' + str(results['source_id'][idx])
        delta_ra = (ra2 - ra1) * 3600
        delta_dec = (dec2 - dec1) * 3600
        plt.scatter(delta_ra, delta_dec, marker='*', color=colors[(idx - 1) % len(colors)], label = l)
        ls = np.random.choice(range(N), 1000)
        for i in ls:
            plt.arrow(delta_ra, delta_dec, dpmra[i], dpmdec[i], alpha=0.1, color = colors[(idx - 1) % len(colors)])
    plt.grid(alpha=0.2, zorder=-10000)
    plt.legend()
    plt.xlabel(r'$\Delta\alpha$ [arcsec]')
    plt.ylabel(r'$\Delta\delta$ [arcsec]')
    plt.title('Proper Motion Vectors Relative to Gaia DR3 ' + str(primary['source_id']))
    if save_plot: plt.savefig('rel_PMs_GaiaDR3' + str(primary['source_id']), dpi=500, bbox_inches='tight')
    else: plt.show()

    def plot_histogram(data_list, bins, xlabel = '', ylabel = '', title = '', logx = False, cumulative = False, save_plot = False):
        """
        Plots histogram(s) for lists of data, with optional log scale and cumulative density.
        """
        plt.figure(figsize = (6, 4))
        for count, (data, color) in enumerate(data_list, start=1):
            l = 'Gaia DR3 ' + str(results['source_id'][count])
            if not cumulative:
                plt.hist(data, bins=bins, histtype='step', lw=2, color=color, label=l)
                cstr = ''
            else:
                plt.hist(data, bins=bins, histtype='step', lw=2, color=color, density=True,
                         cumulative=True, linestyle='--', label=l)
                plt.semilogy()
                cstr = 'Cumulative '

        if logx:
            plt.xscale('log')
            plt.grid(which='both', alpha=0.2)
        else:
            plt.grid(alpha=0.2)

        if xlabel: plt.xlabel(xlabel)
        if ylabel: plt.ylabel(ylabel)
        plt.title(cstr + title)
        plt.legend()
        if save_plot:
            base = re.sub(r'\W+', '_', title.strip().lower()) or 'histogram'
            filename = cstr + f"{base}.png"
            plt.savefig(filename, dpi=500, bbox_inches='tight')
        else:
            plt.show()

    # Plot period distribution
    period_bins = dynamic_log_bins([d for d, _ in all_Ps], num_bins=int(np.sqrt(N)))
    plot_histogram(all_Ps, bins=period_bins, xlabel='P [yr]', title='Periods (Gaia DR3 ' + str(primary['source_id']) + ')', logx=True, save_plot = save_plot)

    # Plot RV slope distributions, one cumulative and one not
    slope_data = [(Ks / (Ps / 4.), color) for (Ks, color), (Ps, _) in zip(all_Ks, all_Ps)]
    slope_bins = dynamic_log_bins([Ks / (Ps / 4.) for (Ks, _), (Ps, _) in zip(all_Ks, all_Ps)], num_bins=int(np.sqrt(N)))
    plot_histogram(slope_data, bins=slope_bins, xlabel=r'$\dot{K}~\left[\mathrm{m~s^{-1}~yr^{-1}}\right]$',
                   title='RV Slopes (Gaia DR3 ' + str(primary['source_id']) + ')', logx=True, save_plot = save_plot)
    plot_histogram(slope_data, bins=slope_bins, xlabel=r'$\dot{K}~\left[\mathrm{m~s^{-1}~yr^{-1}}\right]$',
                   title='RV Slopes (Gaia DR3 ' + str(primary['source_id']) + ')', logx=True, cumulative=True, save_plot = save_plot)
                   
    print_binary_diagnostics(all_Ps, slope_data, period_bins, slope_bins, gaia_ids = secondaries['source_id'])

