import numpy as np
cimport numpy as np


cdef class BaseScaler:
    cdef readonly bint fitted

    cpdef BaseScaler fit(self, np.ndarray data) except *
    cpdef np.ndarray transform(self, np.ndarray data) except *
    cpdef np.ndarray fit_transform(self, np.ndarray data) except *


cdef class StandardScaler(BaseScaler):
    cdef double _mean
    cdef double _std

    cpdef BaseScaler fit(self, np.ndarray data) except *
    cpdef np.ndarray transform(self, np.ndarray data) except *
    cpdef np.ndarray fit_transform(self, np.ndarray data) except *


cdef class MinMaxScaler(BaseScaler):
    cdef double _min
    cdef double _max

    cpdef BaseScaler fit(self, np.ndarray data) except *
    cpdef np.ndarray transform(self, np.ndarray data) except *
    cpdef np.ndarray fit_transform(self, np.ndarray data) except *