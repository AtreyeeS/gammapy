# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Simulate observations"""
import numpy as np
from astropy.table import Table
import astropy.units as u
from gammapy.cube import (
    MapDataset,
    PSFMap,
    make_map_background_irf,
    make_map_exposure_true_energy,
)
from gammapy.data import EventList
from gammapy.maps import WcsNDMap
from gammapy.modeling.models import BackgroundModel
from gammapy.modeling.models import ConstantTemporalModel

from gammapy.utils.random import get_random_state

__all__ = ["simulate_dataset", "MapDatasetEventSampler"]


def simulate_dataset(
    skymodel,
    geom,
    pointing,
    irfs,
    livetime=1 * u.h,
    offset=1 * u.deg,
    max_radius=0.8 * u.deg,
    random_state="random-seed",
):
    """Simulate a 3D dataset.

    Simulate a source defined with a sky model for a given pointing,
    geometry and irfs for a given exposure time.
    This will return a dataset object which includes the counts cube,
    the exposure cube, the psf cube, the background model and the sky model.

    Parameters
    ----------
    skymodel : `~gammapy.modeling.models.SkyModel`
        Background model map
    geom : `~gammapy.maps.WcsGeom`
        Geometry object for the observation
    pointing : `~astropy.coordinates.SkyCoord`
        Pointing position
    irfs : dict
        Irfs used for simulating the observation
    livetime : `~astropy.units.Quantity`
        Livetime exposure of the simulated observation
    offset : `~astropy.units.Quantity`
        Offset from the center of the pointing position.
        This is used for the PSF and Edisp estimation
    max_radius : `~astropy.coordinates.Angle`
        The maximum radius of the PSF kernel.
    random_state: {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.

    Returns
    -------
    dataset : `~gammapy.cube.MapDataset`
        A dataset of the simulated observation.
    """
    background = make_map_background_irf(
        pointing=pointing, ontime=livetime, bkg=irfs["bkg"], geom=geom
    )

    background_model = BackgroundModel(background)

    psf = irfs["psf"].to_energy_dependent_table_psf(theta=offset)
    psf_map = PSFMap.from_energy_dependent_table_psf(psf)

    exposure = make_map_exposure_true_energy(
        pointing=pointing, livetime=livetime, aeff=irfs["aeff"], geom=geom
    )

    if "edisp" in irfs:
        energy = geom.axes[0].edges
        edisp = irfs["edisp"].to_energy_dispersion(offset, e_reco=energy, e_true=energy)
    else:
        edisp = None

    dataset = MapDataset(
        model=skymodel,
        exposure=exposure,
        background_model=background_model,
        psf=psf_map,
        edisp=edisp,
    )

    npred_map = dataset.npred()
    rng = get_random_state(random_state)
    counts = rng.poisson(npred_map.data)
    dataset.counts = WcsNDMap(geom, counts)

    return dataset


class MapDatasetEventSampler:
    """Sample events from a map dataset

    Parameters
    ----------
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.
    """

    def __init__(self, random_state="random-seed"):
        self.random_state = get_random_state(random_state)

    def sample_background(self, dataset):
        """Sample background

        Parameters
        ----------
        dataset : `MapDataset`
            Map dataset.

        Returns
        -------
        events : `EventList`
            Background events
        """
        table = Table()

        background = dataset.background_model.evaluate()
        n_events = self.random_state.poisson(np.sum(background.data))

        # sample position
        coords = background.sample_coord(n_events, self.random_state)
        table["ENERGY"] = coords["energy"]
        table["RA"] = coords.skycoord.icrs.ra.deg
        table["DEC"] = coords.skycoord.icrs.dec.deg
        table["MC_ID"] = 0

        # sample time
        time_start, time_stop, time_ref = (
            dataset.gti.time_start,
            dataset.gti.time_stop,
            dataset.gti.time_ref,
        )
        model = ConstantTemporalModel()
        time = model.sample_time(n_events, time_start, time_stop, self.random_state)
        table["TIME"] = u.Quantity(((time.mjd - time_ref.mjd) * u.day).to(u.s)).value

        return EventList(table)
