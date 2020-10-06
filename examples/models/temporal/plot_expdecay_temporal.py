r"""
.. _constant-temporal-model:

ExpDecay Temporal Model
=======================

This model parametrises an ExpDecay time model.

.. math::
                F(t) = exp(t - t_{ref})/t0


"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from gammapy.modeling.models import Models, ExpDecayTemporalModel, SkyModel

t0= "5 h"
t_ref = Time("2020-10-01")
time_range = [t_ref, t_ref + 1 * u.d]
expdecay_model = ExpDecayTemporalModel(t_ref = t_ref.mjd * u.d, t0 = t0)
expdecay_model.plot(time_range)
plt.grid(which="both")



# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:
from gammapy.modeling.models import PowerLawSpectralModel
model = SkyModel(spectral_model=PowerLawSpectralModel(), temporal_model= expdecay_model, name="expdecay_model")
models = Models([model])

print(models.to_yaml())





