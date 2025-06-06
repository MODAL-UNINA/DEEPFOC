      subroutine tandp (ainp, aint, azp, azt, da1, da2, dd1, dd2, sa1,
     & sa2, pi, rad)
c
c     given two planes compute az and angle of incidence of p & t axes
c
      real              ainp                            
c                                                       ! angle of incidence of p axis
      real              aint                            
c                                                       ! angle of incidence of t axis
      real              azp                             
c                                                       ! azimuth of p axis
      real              azt                             
c                                                       ! azimuth of t axis
      real              da1                             
c                                                       ! dip angle of priniciple plane
      real              da2                             
c                                                       ! dip angle of auxilliary plane
      real              dd1                             
c                                                       ! dip direction of principle plane
      real              dd2                             
c                                                       ! dip direction of auxilliary plane
      real              sa1                             
c                                                       ! rake of principal plane
      real              sa2                             
c                                                       ! rake of auxilliary plane
      real              pi                              
c                                                       ! pi
      real              rad                             
c                                                       ! pi/180
c
      real              ain1                            
c                                                       ! angle of incidence of p/t axis
      real              ain2                            
c                                                       ! angle of incidence of t/p axis
      real              alat1                           
c                                                       ! dip angle in radians of principle plane measured from vertical
      real              alat2                           
c                                                       ! dip angle in radians of auxilliary plane measured from vertical
      real              alon1                           
c                                                       ! dd1 in radians
      real              alon2                           
c                                                       ! dd2 in radians
      real              azimth                          
c                                                       ! azimuth in radians of pole ??
      real              az0                             
c                                                       ! azimuth from pole of auxilliary plane to pole of principle ??
      real              az1                             
c                                                       ! azimuth of p/t axis
      real              az2                             
c                                                       ! azimuth of t/p axis
      real              bazm                            
c                                                       ! not used
      real              delta                           
c                                                       ! not used
      real              plunge                          
c                                                       ! plunge in radians of pole ??
      real              shift                           
c                                                       ! azimuthal shift from pole of plane to p to t axis (= 45 degrees)??
      real              xpos                            
c                                                       ! not used
      real              ypos                            
c                                                       ! not used
c
      parameter (shift = 0.7853981)
c
      alat1 = (90. - da1)*rad
      alon1 = dd1*rad
      alat2 = (90. - da2)*rad
      alon2 = dd2*rad
      call refpt (alat2, alon2)
      call delaz (alat1, alon1, delta, az0, bazm, xpos, ypos)
      call back (shift, az0, plunge, azimth)
      if (abs(azimth) .gt. pi) azimth = azimth - sign(2.0*pi, azimth)
      az1 = azimth/rad
      ain1 = plunge/rad + 90.
      az0 = az0 + pi
      call back (shift, az0, plunge, azimth)
      if (abs(azimth) .gt. pi) azimth = azimth - sign(2.0*pi, azimth)
      az2 = azimth/rad
      ain2 = plunge/rad + 90.
      if (sa1 .ge. 0.) then
        ainp = ain2
	aint = ain1
	azp = az2
	azt = az1
      else
        ainp = ain1
	aint = ain2
	azp = az1
	azt = az2
      end if	
c
c map axes to lower hemisphere
c
	if (ainp .gt. 90.) then
	  ainp = 180. - ainp
	  azp = 180. + azp
	end if
	if (aint .gt. 90.) then
	  aint = 180. - aint
	  azt = 180. + azt
	end if
	if (azp .lt. 0.) azp = azp + 360.
	if (azt .lt. 0.) azt = azt + 360.
      return
      end
