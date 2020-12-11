Background and Purpose
======================

OCEANSAR (Ocean SAR simulator) provides a number of tools to simulate SAR (and other radar) observations of ocean surfaces. It provides:

* Routines to calculate a number of directional wave spectra.
* Routines to calculate time-varying (Lagrangian) ocean surfaces applying linear wave theory.
* Code to calculate the instantaneous NRCS for *Bragg scatterting* and for *specular scattering*.
* Code to calculate time varying complex scattering coefficients (and their temporal evolution) considering the various scattering mechanisms considered.
* Code to simulate single-channel or multi-channel radar (SAR) data. Currently more or less arbitrary combinations of cross-track and along-track baselines are supported, but restricted to a quasi-monostatic geometries.

The code is highly flexible and can be customized to particular radar concepts. For example, we have recently added code to simulate radar signals emulating SKIM's configuration.

A bit of history
================

The code evolved from a simulator of marine radar images, under the name Wavesim, implemented in IDL by Paco Lopez-Dekker as part of a project for Prosensing Inc., in 2003. The code was extended and improved by Gordon Farquharson to simulate FOPAIR acquisitions, an later, still in IDL, futher extended to be able to simulate SAR images. 

In 2011, the code was ported to Python by Gerhard Marull Paretas, who also added:

* MPI support in order to speed-up the wave simulation.
* Various PO scattering models.

The Python code was used as a base for OASIS, a software tool to simulate along-track-interferometric acquisitions with
the future generation European C-band SAR systems (post Sentinel-1) in a  project carried out at the Microwaves and
Radar Institute (DLR) for the European Space Agency (ESA).

Since then, Paco Lopez Dekker added partial support for polarimetric acquisitions, the simulation of formation flying concepts, and derived a version to simulate SKIM.

Usage
=====
Right now probably the best way to run OCEANSAR is:

python path-to-oceansar/oceansar/radarsim/oceansar_sarsim.py config_file.cfg

This runs a single-process (i.e. no MPI) **OCEANSAR** simulation which relies heavily on numba to speed things up. The main script runs four modules:

1. The raw data generator, which produces multi-channel raw data.
2. A sar processing module that generates the corresponding SLCs.
3. A L1b InSAR module that corregisters, removes the flat-earth phase (optionally), and computes the multi-looked interferograms (or in fact all the required elements of the covariance matrix for all the channels).
4. A prototype ATI post-processor. 

Given that the simulator can handle arbitrary combinations of baselines, it is difficult to write a generic ATI or eventually XTI processor, so it is expected that these components would be customized by users for their particular application or configuration.


General Roadmap
===============

Aside from extending OCEANSAR to RAR concepts, such as **SKIM** (currently in progress), current emphasis is on extending simulation capabilities to the bistatic case, with the **Harmony** mission concept as reference scenario.

Things to be done:

* Include models for breaking waves and/or white capping.
* Generalize simulator to bistatic geometries (some of the needed components are already partially implemented).
