import pytest
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy import units as u
from regions import CircleSkyRegion
from numpy.testing import assert_allclose
from gammapy.modeling import Covariance
from gammapy.modeling.models import (
    BackgroundModel,
    GaussianSpatialModel,
    Models,
    PointSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
    FoVBackgroundModel,
)
from gammapy.maps import Map, MapAxis, WcsGeom


@pytest.fixture(scope="session")
def backgrounds():
    axis = MapAxis.from_edges(np.logspace(-1, 1, 3), unit=u.TeV, name="energy")
    geom = WcsGeom.create(skydir=(0, 0), npix=(5, 4), frame="galactic", axes=[axis])
    m = Map.from_geom(geom)
    m.quantity = np.ones(geom.data_shape) * 1e-7
    background1 = BackgroundModel(m, name="bkg1", datasets_names="dataset-1")
    background2 = BackgroundModel(m, name="bkg2", datasets_names=["dataset-2"])
    backgrounds = [background1, background2]
    return backgrounds


@pytest.fixture(scope="session")
def models(backgrounds):
    spatial_model = GaussianSpatialModel(
        lon_0="3 deg", lat_0="4 deg", sigma="3 deg", frame="galactic"
    )
    spectral_model = PowerLawSpectralModel(
        index=2, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    model1 = SkyModel(
        spatial_model=spatial_model, spectral_model=spectral_model, name="source-1",
    )

    model2 = model1.copy(name="source-2")
    model2.datasets_names = ["dataset-1"]
    model3 = model1.copy(name="source-3")
    model3.datasets_names = "dataset-2"
    model3.spatial_model = PointSpatialModel(frame="galactic")
    model3.parameters.freeze_all()
    models = Models([model1, model2, model3] + backgrounds)
    return models


def test_select_region(models):
    center_sky = SkyCoord(3, 4, unit="deg", frame="galactic")
    circle_sky_12 = CircleSkyRegion(center=center_sky, radius=1 * u.deg)
    selected = models.select_region([circle_sky_12])
    assert selected.names == ["source-1", "source-2"]

    center_sky = SkyCoord(0, 0.5, unit="deg", frame="galactic")
    circle_sky_3 = CircleSkyRegion(center=center_sky, radius=1 * u.deg)
    selected = models.select_region([circle_sky_3])
    assert selected.names == ["source-3"]

    selected = models.select_region([circle_sky_3, circle_sky_12])
    assert selected.names == ["source-1", "source-2", "source-3"]

    axis = MapAxis.from_edges(np.logspace(-1, 1, 3), unit=u.TeV, name="energy")
    geom = WcsGeom.create(skydir=(3, 4), npix=(5, 4), frame="galactic", axes=[axis])
    mask = geom.region_mask([circle_sky_12])
    contribute = models.select_mask(mask, margin=None, use_evaluation_region=True)
    inside = models.select_mask(mask, margin=None, use_evaluation_region=False)
    assert contribute.names == ["source-1", "source-2"]
    assert inside.names == ["source-1", "source-2"]

    axis = MapAxis.from_edges(np.logspace(-1, 1, 3), unit=u.TeV, name="energy")
    geom = WcsGeom.create(
        skydir=(3, 4), binsz=0.1, npix=(15, 15), frame="galactic", axes=[axis]
    )
    mask = geom.region_mask([circle_sky_12])
    contribute = models.select_mask(
        mask, margin=4.1 * u.deg, use_evaluation_region=True
    )
    assert contribute.names == ["source-1", "source-2", "source-3"]

    spatial_model = GaussianSpatialModel(
        lon_0="0 deg", lat_0="0 deg", sigma="0.9 deg", frame="galactic"
    )
    assert spatial_model.evaluation_region.height == 2 * spatial_model.evaluation_radius
    model4 = SkyModel(
        spatial_model=spatial_model,
        spectral_model=PowerLawSpectralModel(),
        name="source-4",
    )
    assert model4.contributes(mask, margin=None, use_evaluation_region=True)


def test_select(models):
    conditions = [
        {"datasets_names": "dataset-1"},
        {"datasets_names": "dataset-2"},
        {"datasets_names": ["dataset-1", "dataset-2"]},
        {"datasets_names": None},
        {"tag": "BackgroundModel"},
        {"tag": ["SkyModel", "BackgroundModel"]},
        {"tag": "point", "model_type": "spatial"},
        {"tag": ["point", "gauss"], "model_type": "spatial"},
        {"tag": "pl", "model_type": "spectral"},
        {"tag": ["pl", "pl-norm"], "model_type": "spectral"},
        {"name_substring": "bkg"},
        {"frozen": True},
        {"frozen": False},
        {"datasets_names": "dataset-1", "tag": "pl", "model_type": "spectral"},
    ]

    expected = [
        ["source-1", "source-2", "bkg1"],
        ["source-1", "source-3", "bkg2"],
        ["source-1", "source-2", "source-3", "bkg1", "bkg2"],
        ["source-1", "source-2", "source-3", "bkg1", "bkg2"],
        ["bkg1", "bkg2"],
        ["source-1", "source-2", "source-3", "bkg1", "bkg2"],
        ["source-3"],
        ["source-1", "source-2", "source-3"],
        ["source-1", "source-2", "source-3"],
        ["source-1", "source-2", "source-3", "bkg1", "bkg2"],
        ["bkg1", "bkg2"],
        ["source-3"],
        ["source-1", "source-2", "bkg1", "bkg2"],
        ["source-1", "source-2"],
    ]
    for cdt, xp in zip(conditions, expected):
        selected = models.select(**cdt)
        print(selected.names)
        assert selected.names == xp

    mask = models.selection_mask(**conditions[4]) | models.selection_mask(
        **conditions[6]
    )
    selected = models[mask]
    assert selected.names == ["source-3", "bkg1", "bkg2"]


def test_restore_status(models):
    model = models[1].spectral_model
    covariance_data = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    # the covariance is resest for frozen parameters
    # because of from_factor_matrix (used by the optimizer)
    # so if amplitude if frozen we get
    covariance_frozen = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    model.covariance = Covariance.from_factor_matrix(model.parameters, np.ones((2, 2)))
    assert_allclose(model.covariance.data, covariance_data)
    with models.restore_status(restore_values=True):
        model.amplitude.value = 0
        model.amplitude.frozen = True
        model.covariance = Covariance.from_factor_matrix(
            model.parameters, np.ones((1, 1))
        )
        assert_allclose(model.covariance.data, covariance_frozen)
        assert model.amplitude.value == 0
        assert model.amplitude.error == 0
    assert_allclose(model.amplitude.value, 1e-11)
    assert model.amplitude.frozen == False
    assert isinstance(models.covariance, Covariance)
    assert_allclose(model.covariance.data, covariance_data)
    assert model.amplitude.error == 1

    with models.parameters.restore_status(restore_values=True):
        model.amplitude.value = 0
        model.amplitude.frozen = True
        assert model.amplitude.value == 0
        assert model.amplitude.frozen == True
    assert_allclose(model.amplitude.value, 1e-11)
    assert model.amplitude.frozen == False

    with models.parameters.restore_status(restore_values=False):
        model.amplitude.value = 0
        model.amplitude.frozen = True
    assert model.amplitude.value == 0


def test_bounds(models):

    models.set_parameters_bounds(
        tag="pl",
        model_type="spectral",
        parameters_names="index",
        min=0,
        max=5,
        value=2.4,
    )
    pl_mask = models.selection_mask(tag="pl", model_type="spectral")
    assert np.all([m.spectral_model.index.value == 2.4 for m in models[pl_mask]])
    assert np.all([m.spectral_model.index.min == 0 for m in models[pl_mask]])
    assert np.all([m.spectral_model.index.max == 5 for m in models[pl_mask]])

    models.set_parameters_bounds(
        tag=["pl", "pl-norm"],
        model_type="spectral",
        parameters_names=["norm", "amplitude"],
        min=0,
        max=None,
    )
    bkg_mask = models.selection_mask(tag="BackgroundModel")
    assert np.all([m.spectral_model.amplitude.min == 0 for m in models[pl_mask]])
    assert np.all([m._spectral_model.norm.min == 0 for m in models[bkg_mask]])


def test_freeze(models):
    models.freeze()
    assert np.all([p.frozen for p in models.parameters])
    models.unfreeze()

    assert not models["source-1"].spatial_model.lon_0.frozen
    assert models["source-1"].spectral_model.reference.frozen
    assert not models["source-3"].spatial_model.lon_0.frozen
    assert models["bkg1"].spectral_model.tilt.frozen
    assert not models["bkg1"].spectral_model.norm.frozen

    models.freeze("spatial")
    assert models["source-1"].spatial_model.lon_0.frozen
    assert models["source-3"].spatial_model.lon_0.frozen
    assert not models["bkg1"].spectral_model.norm.frozen

    models.unfreeze("spatial")
    assert not models["source-1"].spatial_model.lon_0.frozen
    assert models["source-1"].spectral_model.reference.frozen
    assert not models["source-3"].spatial_model.lon_0.frozen

    models.freeze("spectral")
    assert models["bkg1"].spectral_model.tilt.frozen
    assert models["bkg1"].spectral_model.norm.frozen
    assert models["source-1"].spectral_model.index.frozen
    assert not models["source-3"].spatial_model.lon_0.frozen

    models.unfreeze("spectral")
    assert models["bkg1"].spectral_model.tilt.frozen
    assert not models["bkg1"].spectral_model.norm.frozen
    assert not models["source-1"].spectral_model.index.frozen


def test_parameters(models):
    pars = models.parameters.select(frozen=False)
    pars.freeze_all()

    assert np.all([p.frozen for p in pars])
    assert len(pars.select(frozen=True)) == len(pars)

    pars.unfreeze_all()
    assert np.all([not p.frozen for p in pars])
    assert len(pars.min) == len(pars)
    assert len(pars.max) == len(pars)

    with pytest.raises(TypeError):
        pars.min = 1

    with pytest.raises(ValueError):
        pars.min = [1]

    with pytest.raises(ValueError):
        pars.max = [2]


def test_fov_bkg_models():
    models = Models(
        [FoVBackgroundModel(dataset_name=name) for name in ["test-1", "test-2"]]
    )
    models.freeze()
    assert models.frozen

    models.parameters.select(name="tilt").unfreeze_all()

    assert not models["test-1-bkg"].spectral_model.tilt.frozen
    assert not models["test-2-bkg"].spectral_model.tilt.frozen

    models.parameters.select(name=["tilt", "norm"]).freeze_all()

    assert models["test-1-bkg"].spectral_model.tilt.frozen
    assert models["test-1-bkg"].spectral_model.norm.frozen


def test_reassign_dataset(models):
    ref = models.select(datasets_names="dataset-2")
    models = models.reassign("dataset-2", "dataset-2-copy")
    assert len(models.select(datasets_names="dataset-2")) == np.sum(
        [m.datasets_names == None for m in models]
    )
    new = models.select(datasets_names="dataset-2-copy")
    assert len(new) == len(ref)
    assert new["source-1"].datasets_names == None
    assert new["source-3"].datasets_names == ["dataset-2-copy"]
