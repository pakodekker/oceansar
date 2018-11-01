
from mpi4py import MPI
import numpy as np

from oceansar import utils


class OceanSurfaceBalancer(object):
    """ Ocean Surface Balancer class

        This class is used to access a surface from
        different MPI processes so that each one is
        assigned an azimuth (y) portion of the surface and
        also gives access to common properties

        :param surface: Full ocean surface (only owned by root process)
        :param dt: Interpolation differential
        :param t0: Initialization time
        :param root: Rank number of surface owner

    """

    def __init__(self, surface, dt, t0=0., root=0):

        # MPI
        self.comm = MPI.COMM_WORLD
        self.size, self.rank = self.comm.Get_size(), self.comm.Get_rank()
        self.root = root

        # Surface
        if self.rank == self.root:
            if not surface:
                raise ValueError('Surface is needed by root process')

            self.surface = surface

            # Prepare surface properties for broadcasting
            surface_prop = {'Lx': self.surface.Lx,
                            'Ly': self.surface.Ly,
                            'dx': self.surface.dx,
                            'dy': self.surface.dy,
                            'Nx': self.surface.Nx,
                            'Ny': self.surface.Ny,
                            'x': self.surface.x,
                            'y': self.surface.y,
                            'wind_dir': self.surface.wind_dir,
                            'wind_dir_eff': self.surface.wind_dir_eff,
                            'wind_fetch': self.surface.wind_fetch,
                            'wind_U': self.surface.wind_U,
                            'wind_U_eff': self.surface.wind_U_eff,
                            'current_mag': self.surface.current_mag,
                            'current_dir': self.surface.current_dir,
                            'compute': self.surface.compute}
        else:
            surface_prop = None

        # Broadcast & save properties
        surface_prop = self.comm.bcast(surface_prop, root=self.root)
        self.Lx = surface_prop['Lx']
        self.Ly = surface_prop['Ly']
        self.dx = surface_prop['dx']
        self.dy = surface_prop['dy']
        self.Nx = surface_prop['Nx']
        self.Ny_full = surface_prop['Ny']
        self.x = surface_prop['x']
        self.y_full = surface_prop['y']
        self.wind_dir = surface_prop['wind_dir']
        self.wind_dir_eff = surface_prop['wind_dir_eff']
        self.wind_fetch = surface_prop['wind_fetch']
        self.wind_U = surface_prop['wind_U']
        self.wind_U_eff = surface_prop['wind_U_eff']
        self.current_mag = surface_prop['current_mag']
        self.current_dir = surface_prop['current_dir']
        self.compute = surface_prop['compute']

        # Setup balancing (counts, displacements) for 2-D matrixes [Ny,Nx]
        self.counts, self.displ = utils.balance_elements(
            self.Ny_full, self.size)
        self.counts *= self.Nx
        self.displ *= self.Nx

        # Process-dependent properties
        self.Ny = np.int(self.counts[self.rank] / self.Nx)
        self.y = np.empty(self.Ny, dtype=np.float32)
        if self.rank == self.root:
            y = (np.ascontiguousarray(surface.y),
                 (self.counts / self.Nx, self.displ / self.Nx), MPI.FLOAT)
        else:
            y = None
        self.comm.Scatterv(y, (self.y, MPI.FLOAT), root=self.root)

        # INITIALIZE SURFACE
        # Memory allocation (LOW (0) / HIGH (1) dt values)
        if 'D' in self.compute:
            self._Dx = (np.empty(2 * int(self.counts[self.rank]), dtype=np.float32).
                        reshape(2, int(self.counts[self.rank] / self.Nx), int(self.Nx)))
            self._Dy = np.empty_like(self._Dx, dtype=np.float32)
            self._Dz = np.empty_like(self._Dx, dtype=np.float32)
        if 'Diff' in self.compute:
            self._Diffx = (np.empty(2 * int(self.counts[self.rank]), dtype=np.float32).
                           reshape(2, int(self.counts[self.rank] / self.Nx), int(self.Nx)))
            self._Diffy = np.empty_like(self._Diffx)
        if 'Diff2' in self.compute:
            self._Diffxx = (np.empty(2 * int(self.counts[self.rank]), dtype=np.float32).
                            reshape(2, int(self.counts[self.rank] / self.Nx), int(self.Nx)))
            self._Diffyy = np.empty_like(self._Diffxx)
            self._Diffxy = np.empty_like(self._Diffxx)
        if 'V' in self.compute:
            self._Vx = (np.empty(2 * int(self.counts[self.rank]), dtype=np.float32).
                        reshape(2, int(self.counts[self.rank] / self.Nx), int(self.Nx)))
            self._Vy = np.empty_like(self._Vx)
            self._Vz = np.empty_like(self._Vx)
        if 'A' in self.compute:
            self._Ax = (np.empty(2 * int(self.counts[self.rank]), dtype=np.float32).
                        reshape(2, int(self.counts[self.rank] / self.Nx), int(self.Nx)))
            self._Ay = np.empty_like(self._Ax)
            self._Az = np.empty_like(self._Ax)
        if 'hMTF' in self.compute:
            self._hMTF = (np.empty(2 * int(self.counts[self.rank]), dtype=np.float32).
                          reshape(2, int(self.counts[self.rank] / self.Nx), int(self.Nx)))

        self.dt = dt
        self.t_l_last = -1.
        self.t_h_last = -1.
        self.t = t0

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):

        self._t = np.float32(value)

        # Update low/high times
        t_l = np.float32(np.floor(self._t / self.dt) * self.dt)
        t_h = t_l + self.dt

        if (t_l != self.t_l_last) or (t_h != self.t_h_last):
            # Only update t_h if 'going up'
            if t_l == self.t_h_last:
                if 'D' in self.compute:
                    self._Dx[0] = self._Dx[1]
                    self._Dy[0] = self._Dy[1]
                    self._Dz[0] = self._Dz[1]
                if 'Diff' in self.compute:
                    self._Diffx[0] = self._Diffx[1]
                    self._Diffy[0] = self._Diffy[1]
                if 'Diff2' in self.compute:
                    self._Diffxx[0] = self._Diffxx[1]
                    self._Diffyy[0] = self._Diffyy[1]
                    self._Diffxy[0] = self._Diffxy[1]
                if 'V' in self.compute:
                    self._Vx[0] = self._Vx[1]
                    self._Vy[0] = self._Vy[1]
                    self._Vz[0] = self._Vz[1]
                if 'A' in self.compute:
                    self._Ax[0] = self._Ax[1]
                    self._Ay[0] = self._Ay[1]
                    self._Az[0] = self._Az[1]
                if 'hMTF' in self.compute:
                    self._hMTF[0] = self._hMTF[1]

                t_update = np.array([[1, t_h]])
            else:
                t_update = np.array([[0, t_l], [1, t_h]])

            # Initialize surface properties
            for t_i in t_update:
                if self.rank == self.root:
                    self.surface.t = t_i[1]
                    if 'D' in self.compute:
                        Dx_f = (self.surface.Dx,
                                (self.counts, self.displ), MPI.FLOAT)
                        Dy_f = (self.surface.Dy,
                                (self.counts, self.displ), MPI.FLOAT)
                        Dz_f = (self.surface.Dz,
                                (self.counts, self.displ), MPI.FLOAT)
                    if 'Diff' in self.compute:
                        Diffx_f = (self.surface.Diffx,
                                   (self.counts, self.displ), MPI.FLOAT)
                        Diffy_f = (self.surface.Diffy,
                                   (self.counts, self.displ), MPI.FLOAT)
                    if 'Diff2' in self.compute:
                        Diffxx_f = (self.surface.Diffxx,
                                    (self.counts, self.displ), MPI.FLOAT)
                        Diffyy_f = (self.surface.Diffyy,
                                    (self.counts, self.displ), MPI.FLOAT)
                        Diffxy_f = (self.surface.Diffxy,
                                    (self.counts, self.displ), MPI.FLOAT)
                    if 'V' in self.compute:
                        Vx_f = (self.surface.Vx,
                                (self.counts, self.displ), MPI.FLOAT)
                        Vy_f = (self.surface.Vy,
                                (self.counts, self.displ), MPI.FLOAT)
                        Vz_f = (self.surface.Vz,
                                (self.counts, self.displ), MPI.FLOAT)
                    if 'A' in self.compute:
                        Ax_f = (self.surface.Ax,
                                (self.counts, self.displ), MPI.FLOAT)
                        Ay_f = (self.surface.Ay,
                                (self.counts, self.displ), MPI.FLOAT)
                        Az_f = (self.surface.Az,
                                (self.counts, self.displ), MPI.FLOAT)
                    if 'hMTF' in self.compute:
                        hMTF_f = (self.surface.hMTF,
                                  (self.counts, self.displ), MPI.FLOAT)

                else:
                    if 'D' in self.compute:
                        Dx_f = None
                        Dy_f = None
                        Dz_f = None
                    if 'Diff' in self.compute:
                        Diffx_f = None
                        Diffy_f = None
                    if 'Diff2' in self.compute:
                        Diffxx_f = None
                        Diffyy_f = None
                        Diffxy_f = None
                    if 'V' in self.compute:
                        Vx_f = None
                        Vy_f = None
                        Vz_f = None
                    if 'A' in self.compute:
                        Ax_f = None
                        Ay_f = None
                        Az_f = None
                    if 'hMTF' in self.compute:
                        hMTF_f = None

                if 'D' in self.compute:
                    self.comm.Scatterv(
                        Dx_f, (self._Dx[int(t_i[0])], MPI.FLOAT), root=self.root)
                    self.comm.Scatterv(
                        Dy_f, (self._Dy[int(t_i[0])], MPI.FLOAT), root=self.root)
                    self.comm.Scatterv(
                        Dz_f, (self._Dz[int(t_i[0])], MPI.FLOAT), root=self.root)
                if 'Diff' in self.compute:
                    self.comm.Scatterv(
                        Diffx_f, (self._Diffx[int(t_i[0])], MPI.FLOAT), root=self.root)
                    self.comm.Scatterv(
                        Diffy_f, (self._Diffy[int(t_i[0])], MPI.FLOAT), root=self.root)
                if 'Diff2' in self.compute:
                    self.comm.Scatterv(
                        Diffxx_f, (self._Diffxx[int(t_i[0])], MPI.FLOAT), root=self.root)
                    self.comm.Scatterv(
                        Diffyy_f, (self._Diffyy[int(t_i[0])], MPI.FLOAT), root=self.root)
                    self.comm.Scatterv(
                        Diffxy_f, (self._Diffxy[int(t_i[0])], MPI.FLOAT), root=self.root)
                if 'V' in self.compute:
                    self.comm.Scatterv(
                        Vx_f, (self._Vx[int(t_i[0])], MPI.FLOAT), root=self.root)
                    self.comm.Scatterv(
                        Vy_f, (self._Vy[int(t_i[0])], MPI.FLOAT), root=self.root)
                    self.comm.Scatterv(
                        Vz_f, (self._Vz[int(t_i[0])], MPI.FLOAT), root=self.root)
                if 'A' in self.compute:
                    self.comm.Scatterv(
                        Ax_f, (self._Ax[int(t_i[0])], MPI.FLOAT), root=self.root)
                    self.comm.Scatterv(
                        Ay_f, (self._Ay[int(t_i[0])], MPI.FLOAT), root=self.root)
                    self.comm.Scatterv(
                        Az_f, (self._Az[int(t_i[0])], MPI.FLOAT), root=self.root)
                if 'hMTF' in self.compute:
                    self.comm.Scatterv(
                        hMTF_f, (self._hMTF[int(t_i[0])], MPI.FLOAT), root=self.root)

        self.t_l_last = t_l
        self.t_h_last = t_h

        # Apply linear interpolation
        w_h = np.float32((self._t - t_l) / self.dt)
        w_l = np.float32(1. - w_h)

        if 'D' in self.compute:
            self.Dx = w_l * self._Dx[0] + w_h * self._Dx[1]
            self.Dy = w_l * self._Dy[0] + w_h * self._Dy[1]
            self.Dz = w_l * self._Dz[0] + w_h * self._Dz[1]
        if 'Diff' in self.compute:
            self.Diffx = w_l * self._Diffx[0] + w_h * self._Diffx[1]
            self.Diffy = w_l * self._Diffy[0] + w_h * self._Diffy[1]
        if 'Diff2' in self.compute:
            self.Diffxx = w_l * self._Diffxx[0] + w_h * self._Diffxx[1]
            self.Diffyy = w_l * self._Diffyy[0] + w_h * self._Diffyy[1]
            self.Diffxy = w_l * self._Diffxy[0] + w_h * self._Diffxy[1]
        if 'V' in self.compute:
            self.Vx = w_l * self._Vx[0] + w_h * self._Vx[1]
            self.Vy = w_l * self._Vy[0] + w_h * self._Vy[1]
            self.Vz = w_l * self._Vz[0] + w_h * self._Vz[1]
        if 'A' in self.compute:
            self.Ax = w_l * self._Ax[0] + w_h * self._Ax[1]
            self.Ay = w_l * self._Ay[0] + w_h * self._Ay[1]
            self.Az = w_l * self._Az[0] + w_h * self._Az[1]
        if 'hMTF' in self.compute:
            self.hMTF = w_l * self._hMTF[0] + w_h * self._hMTF[1]
