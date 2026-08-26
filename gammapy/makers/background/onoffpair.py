# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""On-Off background estimation for dedicated On Off Pair runs"""

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

from gammapy.data import EventList
from gammapy.datasets import (
    MapDataset,
    MapDatasetOnOff,
    SpectrumDataset,
    SpectrumDatasetOnOff,
)
from ..core import Maker
from gammapy.maps import Map
from gammapy.utils.coordinates import fov_to_sky, sky_to_fov


class OnOffBackgroundMaker(Maker):
    """Estimate OFF background from a paired OFF observation.

    Turns a reduced ``MapDataset`` / ``SpectrumDataset``
    into its OnOff counterpart by mirroring the OFF run's events into
    the ON pointing FoV frame and setting acceptances from livetime.

    On-off pairs should be created beforehand by the user.

    Parameters
    ----------
    acceptance_method : str, optional
        Allowed option are {"livetime_ratio"}
        Default is "livetime"
        Strategy for the acceptance calculation
        "livetime_ratio" takes the ratio of livetimes, flat acceptance
    """

    tag = "OnOffBackgroundMaker"

    available_acceptance_methods = ["livetime_ratio"]

    def __init__(self, acceptance_method="livetime_ratio"):
        if acceptance_method not in self.available_acceptance_methods:
            raise ValueError(
                f"Unknown acceptance_method {acceptance_method!r}, "
                f"choose from {self.available_acceptance_methods}"
            )
        self.acceptance_method = acceptance_method

    def _mirror_off_events(self, on_observation, off_observation):
        """Reproject OFF events into the ON pointing FoV frame (aligned with RA/DEC)."""
        events = off_observation.events
        off_pointing = off_observation.get_pointing_icrs(off_observation.tmid)
        on_pointing = on_observation.get_pointing_icrs(on_observation.tmid)

        fov_lon, fov_lat = sky_to_fov(
            events.radec.ra, events.radec.dec, off_pointing.ra, off_pointing.dec
        )
        off_ra, off_dec = fov_to_sky(fov_lon, fov_lat, on_pointing.ra, on_pointing.dec)
        table = events.table.copy()
        off_coord = SkyCoord(off_ra, off_dec, frame="icrs")
        table["RA"] = off_coord.ra
        table["DEC"] = off_coord.dec
        return EventList(table)

    def _make_acceptance(self, dataset, on_observation, off_observation):
        geom = dataset.counts.geom
        t_on = on_observation.observation_live_time_duration.to_value("s")
        t_off = off_observation.observation_live_time_duration.to_value("s")

        if self.acceptance_method == "livetime_ratio":
            acceptance = Map.from_geom(geom, unit="", data=t_on)
            acceptance_off = Map.from_geom(geom, unit="", data=t_off)
            return acceptance, acceptance_off

        raise ValueError(f"Unhandled acceptance_method {self.acceptance_method!r}")

    def run(self, dataset, on_observation, off_observation):
        """Create background using dedicated off pointing.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.SpectrumDataset`
            Reduced dataset for the ON observation
        on_observation : `~gammapy.data.Observation`
            The ON observation.
        off_observation : `~gammapy.data.Observation`
            The paired OFF observation.

        Returns
        -------
        dataset : `~gammapy.datasets.MapDatasetOnOff` or `~gammapy.datasets.SpectrumDatasetOnOff`
        """
        geom = dataset.counts.geom
        counts_off = Map.from_geom(geom, unit="")
        counts_off.fill_events(self._mirror_off_events(on_observation, off_observation))

        acceptance, acceptance_off = self._make_acceptance(
            dataset, off_observation, on_observation
        )

        if isinstance(dataset, SpectrumDataset):
            return SpectrumDatasetOnOff.from_spectrum_dataset(
                dataset=dataset,
                acceptance=acceptance,
                acceptance_off=acceptance_off,
                counts_off=counts_off,
                name=dataset.name,
            )
        elif isinstance(dataset, MapDataset):
            return MapDatasetOnOff.from_map_dataset(
                dataset=dataset,
                acceptance=acceptance,
                acceptance_off=acceptance_off,
                counts_off=counts_off,
                name=dataset.name,
            )
        raise TypeError(
            f"Expected MapDataset or SpectrumDataset, got {type(dataset).__name__}"
        )
