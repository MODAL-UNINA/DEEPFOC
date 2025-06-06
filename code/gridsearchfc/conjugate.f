
      subroutine auxpln (dd1, da1, sa1, dd2, da2, sa2)
c
c    Calculate the auxilliary plane of a double couple fault plane solution, given the principle plane.
c
c    written by Paul Reasenberg, June, 1984, from class notes by Dave Boore, (both at the U.S.G.S., Menlo Park.)
c    angle variables phi, del and lam are as defined in Aki and Richards, (1980), p.114.
c
      real              da1                             
c                                                       ! (input)  dip angle in degrees of priciple plane
      real              dd1                             
c                                                       ! (input)  dip directions in degrees of priciple plane
      real              sa1                             
c                                                       ! (input)  slip angle in degrees of priciple plane
      real              da2                             
c                                                       ! (output)  dip angle in degrees of auxilliary plane
      real              dd2                             
c                                                       ! (output)  dip directions in degrees of auxilliary plane
      real              sa2                             
c                                                       ! (output)  slip angle in degrees of auxilliary plane
c

      double precision  bot                             
c                                                       ! scratch variable
      double precision  del1                            
c                                                       ! dip angle of principal plane in radians
      logical           first                           
c                                                       ! test: true if first time into routine
      double precision  phi1                            
c                                                       ! fault plane strike of principal plane
      double precision  phi2                            
c                                                       ! strike of auxilliary plane in radians
      double precision  rad                             
c                                                       ! conversion factor from degrees to radian
      double precision  sgn                             
c                                                       ! saves principal plane slip angle for assigning proper sign to auxilliary
      double precision  top                             
c                                                       ! scratch variable
      double precision  xlam1                           
c                                                       ! slip angle of principal plane in radians
      double precision  xlam2                           
c                                                       ! slip angle of auxilliary plane
c
      data first /.true./
      save first, rad
c
      if (first) then
        first = .false.
        rad = datan(1.0d0)/45.0d0
      end if
c
      phi1 = dd1 - 90.0d0
      if (phi1 .lt. 0.0d0) phi1 = phi1 + 360.0d0
      phi1 = phi1*rad
      del1 = da1*rad
      sgn = sa1
      xlam1 = sa1*rad
c
      top = dcos(xlam1)*dsin(phi1) - dcos(del1)*dsin(xlam1)*dcos(phi1)
      bot = dcos(xlam1)*dcos(phi1) + dcos(del1)*dsin(xlam1)*dsin(phi1)
      dd2 = datan2(top, bot)/rad
      phi2 = (dd2 - 90.0d0)*rad
      if (sa1 .lt. 0.0d0) dd2 = dd2 - 180.0d0
      if (dd2 .lt. 0.0d0) dd2 = dd2 + 360.0d0
      if (dd2. gt. 360.0d0) dd2 = dd2 - 360.0d0
c
      da2 = dacos(dsin(dabs(xlam1))*dsin(del1))/rad
      xlam2 = -dcos(phi2)*dsin(del1)*dsin(phi1) +
     & dsin(phi2)*dsin(del1)*dcos(phi1)
c
c machine accuracy problem
c
      if (dabs(xlam2) .gt. 1.0d0) then
        xlam2 = dsign(1.0d0, xlam2)
      end if
      xlam2 = dsign(dacos(xlam2), sgn)
      sa2 = xlam2/rad
c
      return
      end
