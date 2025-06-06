      subroutine pexcf (coef, i, mxstat, u)
c
c
c calculates coefficients for determining the far-field radiation pattern of p waves from the moment-rate tensor components of a
c point source in an infinite, homogeneous, elastic medium.  the radiation pattern is normalized; to obtain particle amplitudes,
c multiply by
c
c    1.0/(4.0*pi*rho*(v**3)*r),
c
c     where:
c          rho is the density in the source region,
c          v is the p-wave speed in the source region, and
c          r is the geometric spreading factor
c            (for a homogeneous medium, this is equal to the distance
c            to the observation point.)
c
c reference:
c         aki, keiiti, and paul g. richards, quantitative seismology,
c         freeman, san francisco, 1980, equation 49.1, page 118.
c
c written by bruce julian
c
      integer           mxstat                          
c							! (input) maximum # of stations permitted
      real              coef(mxstat, 6)                 
c							! (output) excitation coefficients
      integer           i                               
c							! (input) index of station
      real              u(3)                            
c							! (input) unit vector in ray direction
c
      coef(i, 1) = u(1)*u(1)
      coef(i, 2) = 2.*u(1)*u(2)
      coef(i, 3) = u(2)*u(2)
      coef(i, 4) = 2.*u(3)*u(1)
      coef(i, 5) = 2.*u(2)*u(3)
      coef(i, 6) = u(3)*u(3)
c
      return
      end
