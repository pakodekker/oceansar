Background and Purpose
======================

OCEANSAR (Ocean SAR simulator) provides a number of tools to do just that. It provides:

* Routines to calculate a number of directional wave spectra.
* Routines to calculate time-varying (Lagrangian) ocean surfaces applying linear wave theory.
* Code to calculate the instantaneous NRCS for *Bragg scatterting* and for *specular scattering*.
* Code to calculate time varying complex scattering coefficients (and their temporal evolution) considering the
various scattering mechanisms considered.
* Code to simulate raw and processed multi-channel SAR data.

The code is highly flexible and can be customized to particular radar concepts. For example, we have recently added code to simulate radar signals emulating SKIM's configuration.

A bit of history
================

A first iteration the code, under the name Wavesim, implemented in IDL, was implemented by Paco Lopez-Dekker as part of a project for
Prosensing Inc., in 2003. Gordon Farquharson fixed some bugs and added support for Elfouhaily's spectrum in order to use
the code to simulate FOPAIR acquisitions (note the ironiy that FOPAIR was used to study the surf-zone, where neither
the spectral models implemented nor the deep-water linear theory wave model as applicable).

In 2011, the code was ported to Python by Gerhard Marull Paretas, who also added:

* MPI support in order to speed-up the wave simulation.
* Various PO scattering models.

The Python code was used as a base for OASIS, a software tool to simulate along-track-interferometric acquisitions with
the future generation European C-band SAR systems (post Sentinel-1) in a  project carried out at the Microwaves and
Radar Institute (DLR) for the European Space Agency (ESA).

Since then, Paco Lopez Dekker added support for polarimetric acquisitions, and separated OCEANSAR from the radar-specific
code in order to be able to open-source the first part.

General Roadmap
===============

Aside from extending OCEANSAR to RAR concepts, such as **SKIM** (currently in progress), current emphasis is on extending simulation capabilities to the bistatic case, with the **STEREOID** mission concept as reference scenario.

Things to be done:
* Include models for breaking waves and/or white capping.
* Implement routines to compute expected statistics of radar returns.
