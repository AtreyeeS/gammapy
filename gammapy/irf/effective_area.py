# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import quantity_support
from gammapy.maps import MapAxis, MapAxes, RegionGeom, RegionNDMap
from gammapy.utils.scripts import make_path
from .core import IRF

__all__ = ["EffectiveAreaTable", "EffectiveAreaTable2D"]


class EffectiveAreaTable(IRF):
    """Effective area table.

    TODO: Document

    Parameters
    ----------
    energy_axis_true : `MapAxis`
        Energy axis
    data : `~astropy.units.Quantity`
        Effective area
    meta : dict
        Meta data

    Examples
    --------
    Plot parametrized effective area for HESS, HESS2 and CTA.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        from gammapy.irf import EffectiveAreaTable

        energy = np.logspace(-3, 3, 100) * u.TeV

        for instrument in ['HESS', 'HESS2', 'CTA']:
            aeff = EffectiveAreaTable.from_parametrization(energy, instrument)
            ax = aeff.plot(label=instrument)

        ax.set_yscale('log')
        ax.set_xlim([1e-3, 1e3])
        ax.set_ylim([1e3, 1e12])
        plt.legend(loc='best')
        plt.show()

    Find energy where the effective area is at 10% of its maximum value

    >>> import numpy as np
    >>> import astropy.units as u
    >>> from gammapy.irf import EffectiveAreaTable
    >>> energy = np.logspace(-1, 2) * u.TeV
    >>> aeff_max = aeff.max_area # doctest: +SKIP
    >>> print(aeff_max).to('m2') # doctest: +SKIP
    156909.413371 m2
    >>> energy_threshold = aeff.find_energy(0.1 * aeff_max) # doctest: +SKIP
    >>> print(energy_threshold) # doctest: +SKIP
    0.185368478744 TeV
    """
    required_axes = ["energy_true"]

    def plot(self, ax=None, show_energy=None, **kwargs):
        """Plot effective area.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        energy : `~astropy.units.Quantity`
            Energy nodes
        show_energy : `~astropy.units.Quantity`, optional
            Show energy, e.g. threshold, as vertical line

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        kwargs.setdefault("lw", 2)

        energy_axis = self.axes["energy_true"]

        with quantity_support():
            ax.errorbar(energy_axis.center, self.quantity, xerr=energy_axis.as_xerr, **kwargs)

        if show_energy is not None:
            ener_val = u.Quantity(show_energy).to_value(energy_axis.unit)
            ax.vlines(ener_val, 0, 1.1 * self.max_area, linestyles="dashed")

        ax.set_xscale("log")
        ax.set_xlabel(f"Energy [{energy_axis.unit}]")
        ax.set_ylabel(f"Effective Area [{self.unit}]")
        return ax

    @classmethod
    def from_parametrization(cls, energy, instrument="HESS"):
        r"""Create parametrized effective area.

        Parametrizations of the effective areas of different Cherenkov
        telescopes taken from Appendix B of Abramowski et al. (2010), see
        https://ui.adsabs.harvard.edu/abs/2010MNRAS.402.1342A .

        .. math::
            A_{eff}(E) = g_1 \left(\frac{E}{\mathrm{MeV}}\right)^{-g_2}\exp{\left(-\frac{g_3}{E}\right)}

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy binning, analytic function is evaluated at log centers
        instrument : {'HESS', 'HESS2', 'CTA'}
            Instrument name
        """
        energy = u.Quantity(energy)
        # Put the parameters g in a dictionary.
        # Units: g1 (cm^2), g2 (), g3 (MeV)
        # Note that whereas in the paper the parameter index is 1-based,
        # here it is 0-based
        pars = {
            "HESS": [6.85e9, 0.0891, 5e5],
            "HESS2": [2.05e9, 0.0891, 1e5],
            "CTA": [1.71e11, 0.0891, 1e5],
        }

        if instrument not in pars.keys():
            ss = f"Unknown instrument: {instrument}\n"
            ss += "Valid instruments: HESS, HESS2, CTA"
            raise ValueError(ss)

        energy_axis_true = MapAxis.from_edges(energy, interp="log", name="energy_true")

        g1 = pars[instrument][0]
        g2 = pars[instrument][1]
        g3 = -pars[instrument][2]

        energy = energy_axis_true.center.to_value("MeV")
        data = g1 * energy ** (-g2) * np.exp(g3 / energy)
        return cls(axes=[energy_axis_true], data=data, unit="cm2")

    @classmethod
    def from_constant(cls, energy, value):
        """Create constant value effective area.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy binning, analytic function is evaluated at log centers
        value : `~astropy.units.Quantity`
            Effective area
        """
        value = u.Quantity(value)
        energy_axis_true = MapAxis.from_energy_edges(energy, name="energy_true")
        return cls(axes=[energy_axis_true], data=value.value, unit=value.unit)

    @classmethod
    def from_table(cls, table):
        """Create from `~astropy.table.Table` in ARF format.

        Data format specification: :ref:`gadf:ogip-arf`
        """
        axes = MapAxes.from_table(table, format="ogip-arf")[cls.required_axes]
        data = table["SPECRESP"].quantity
        return cls(axes=axes, data=data.value, unit=data.unit)

    @classmethod
    def from_hdulist(cls, hdulist, hdu="SPECRESP"):
        """Create from `~astropy.io.fits.HDUList`."""
        return cls.from_table(Table.read(hdulist[hdu]))

    @classmethod
    def read(cls, filename, hdu="SPECRESP"):
        """Read from file."""
        filename = str(make_path(filename))
        with fits.open(filename, memmap=False) as hdulist:
            try:
                return cls.from_hdulist(hdulist, hdu=hdu)
            except KeyError:
                raise ValueError(
                    f"File {filename} contains no HDU {hdu!r}\n"
                    f"Available: {[_.name for _ in hdulist]}"
                )

    def to_table(self):
        """Convert to `~astropy.table.Table` in ARF format.

        Data format specification: :ref:`gadf:ogip-arf`
        """
        table = self.axes.to_table(format="ogip-arf")
        table.meta = {
            "EXTNAME": "SPECRESP",
            "hduclass": "OGIP",
            "hduclas1": "RESPONSE",
            "hduclas2": "SPECRESP",
        }
        table["SPECRESP"] = self.evaluate_fill_nan()
        return table

    def to_region_map(self, region=None):
        """"""
        geom = RegionGeom(region=region, axes=self.axes)
        return RegionNDMap.from_geom(
            geom=geom, data=self.data, unit=self.unit
        )

    def to_hdulist(self, name=None, use_sherpa=False):
        """Convert to `~astropy.io.fits.HDUList`."""
        table = self.to_table()

        if use_sherpa:
            table["ENERG_HI"] = table["ENERG_HI"].quantity.to("keV")
            table["ENERG_LO"] = table["ENERG_LO"].quantity.to("keV")
            table["SPECRESP"] = table["SPECRESP"].quantity.to("cm2")

        return fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(table, name=name)])

    def write(self, filename, use_sherpa=False, **kwargs):
        """Write to file."""
        filename = str(make_path(filename))
        self.to_hdulist(use_sherpa=use_sherpa).writeto(filename, **kwargs)

    def evaluate_fill_nan(self, **kwargs):
        """Modified evaluate function.

        Replaces possible nan values. Below the finite range the effective area is set
        to zero and above to value of the last valid note. This is needed since
        other codes, e.g. sherpa, don't like nan values in FITS files. Make
        sure that the replacement happens outside of the energy range, where
        the `~gammapy.irf.EffectiveAreaTable` is used.
        """
        retval = self.evaluate(**kwargs)
        idx = np.where(np.isfinite(retval))[0]
        retval[np.arange(idx[0])] = 0
        retval[np.arange(idx[-1], len(retval))] = retval[idx[-1]]
        return retval

    @property
    def max_area(self):
        """Maximum effective area."""
        cleaned_data = self.quantity[np.where(~np.isnan(self.quantity))]
        return cleaned_data.max()

    def find_energy(self, aeff, energy_min=None, energy_max=None):
        """Find energy for a given effective area.

        In case the solution is not unique, provide the `energy_min` or `energy_max` arguments
        to limit the solution to the given range. By default the peak energy of the
        effective area is chosen as `energy_max`.

        Parameters
        ----------
        aeff : `~astropy.units.Quantity`
            Effective area value
        energy_min : `~astropy.units.Quantity`
            Lower bracket value in case solution is not unique.
        energy_max : `~astropy.units.Quantity`
            Upper bracket value in case solution is not unique.

        Returns
        -------
        energy : `~astropy.units.Quantity`
            Energy corresponding to the given aeff.
        """
        from gammapy.modeling.models import TemplateSpectralModel

        energy = self.axes["energy_true"].center

        if energy_min is None:
            energy_min = energy[0]
        if energy_max is None:
            # use the peak effective area as a default for the energy maximum
            energy_max = energy[np.argmax(self.data.data)]

        aeff_spectrum = TemplateSpectralModel(energy, self.quantity)
        return aeff_spectrum.inverse(aeff, energy_min=energy_min, energy_max=energy_max)


class EffectiveAreaTable2D(IRF):
    """2D effective area table.

    Data format specification: :ref:`gadf:aeff_2d`

    Parameters
    ----------
    energy_axis_true : `MapAxis`
        True energy axis
    offset_axis : `MapAxis`
        Field of view offset axis.
    data : `~astropy.units.Quantity`
        Effective area
    meta : dict
        Meta data

    Examples
    --------
    Here's an example you can use to learn about this class:

    >>> from gammapy.irf import EffectiveAreaTable2D
    >>> filename = '$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits'
    >>> aeff = EffectiveAreaTable2D.read(filename, hdu='EFFECTIVE AREA')
    >>> print(aeff)
    EffectiveAreaTable2D
    --------------------
    <BLANKLINE>
      axes  : ['energy_true', 'offset']
      shape : (42, 6)
      ndim  : 2
      unit  : m2
      dtype : >f4
    <BLANKLINE>

    Here's another one, created from scratch, without reading a file:

    >>> from gammapy.irf import EffectiveAreaTable2D
    >>> from gammapy.maps import MapAxis
    >>> energy_axis_true = MapAxis.from_energy_bounds("0.1 TeV", "100 TeV", nbin=30, name="energy_true")
    >>> offset_axis = MapAxis.from_bounds(0, 5, nbin=4, name="offset")
    >>> aeff = EffectiveAreaTable2D(axes=[energy_axis_true, offset_axis], data=1e10, unit="cm2")
    >>> print(aeff)
    EffectiveAreaTable2D
    --------------------
    <BLANKLINE>
      axes  : ['energy_true', 'offset']
      shape : (30, 4)
      ndim  : 2
      unit  : cm2
      dtype : float64
    <BLANKLINE>

    """

    tag = "aeff_2d"
    required_axes = ["energy_true", "offset"]

    @property
    def low_threshold(self):
        """Low energy threshold"""
        return self.meta["LO_THRES"] * u.TeV

    @property
    def high_threshold(self):
        """High energy threshold"""
        return self.meta["HI_THRES"] * u.TeV

    def to_effective_area_table(self, offset, energy=None):
        """Evaluate at a given offset and return `~gammapy.irf.EffectiveAreaTable`.

        Parameters
        ----------
        offset : `~astropy.coordinates.Angle`
            Offset
        energy : `~astropy.units.Quantity`
            Energy axis bin edges
        """
        if energy is None:
            energy_axis_true = self.axes["energy_true"]
        else:
            energy_axis_true = MapAxis.from_energy_edges(energy, name="energy_true")

        area = self.evaluate(offset=offset, energy_true=energy_axis_true.center)

        return EffectiveAreaTable(axes=[energy_axis_true], data=area.value, unit=area.unit)

    def plot_energy_dependence(self, ax=None, offset=None, **kwargs):
        """Plot effective area versus energy for a given offset.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        offset : `~astropy.coordinates.Angle`
            Offset
        kwargs : dict
            Forwarded tp plt.plot()

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        if offset is None:
            off_min, off_max = self.axes["offset"].center[[0, -1]]
            offset = np.linspace(off_min.value, off_max.value, 4) * off_min.unit

        energy = self.axes["energy_true"].center

        for off in offset:
            area = self.evaluate(offset=off, energy_true=energy)
            kwargs.setdefault("label", f"offset = {off:.1f}")
            ax.plot(energy, area.value, **kwargs)

        ax.set_xscale("log")
        ax.set_xlabel(f"Energy [{energy.unit}]")
        ax.set_ylabel(f"Effective Area [{self.unit}]")
        ax.set_xlim(min(energy.value), max(energy.value))
        return ax

    def plot_offset_dependence(self, ax=None, offset=None, energy=None, **kwargs):
        """Plot effective area versus offset for a given energy.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        offset : `~astropy.coordinates.Angle`
            Offset axis
        energy : `~astropy.units.Quantity`
            Energy

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        if energy is None:
            energy_axis = self.axes["energy_true"]
            e_min, e_max = energy_axis.center[[0, -1]]
            energy = np.geomspace(e_min, e_max, 4)

        if offset is None:
            offset = self.axes["offset"].center

        for ee in energy:
            area = self.evaluate(offset=offset, energy_true=ee)
            area /= np.nanmax(area)
            if np.isnan(area).all():
                continue
            label = f"energy = {ee:.1f}"
            ax.plot(offset, area, label=label, **kwargs)

        ax.set_ylim(0, 1.1)
        ax.set_xlabel(f"Offset ({self.axes['offset'].unit})")
        ax.set_ylabel("Relative Effective Area")
        ax.legend(loc="best")

        return ax

    def plot(self, ax=None, add_cbar=True, **kwargs):
        """Plot effective area image."""
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        energy = self.axes["energy_true"].edges
        offset = self.axes["offset"].edges
        aeff = self.evaluate(offset=offset, energy_true=energy[:, np.newaxis])

        vmin, vmax = np.nanmin(aeff.value), np.nanmax(aeff.value)

        kwargs.setdefault("cmap", "GnBu")
        kwargs.setdefault("edgecolors", "face")
        kwargs.setdefault("vmin", vmin)
        kwargs.setdefault("vmax", vmax)

        caxes = ax.pcolormesh(energy.value, offset.value, aeff.value.T, **kwargs)

        ax.set_xscale("log")
        ax.set_ylabel(f"Offset ({offset.unit})")
        ax.set_xlabel(f"Energy ({energy.unit})")

        xmin, xmax = energy.value.min(), energy.value.max()
        ax.set_xlim(xmin, xmax)

        if add_cbar:
            label = f"Effective Area ({aeff.unit})"
            ax.figure.colorbar(caxes, ax=ax, label=label)

        return ax

    def peek(self, figsize=(15, 5)):
        """Quick-look summary plots."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        self.plot(ax=axes[2])
        self.plot_energy_dependence(ax=axes[0])
        self.plot_offset_dependence(ax=axes[1])
        plt.tight_layout()
