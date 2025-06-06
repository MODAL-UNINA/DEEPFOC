c     geocen
c
c      geocen - calculate geocentric postitions, distances, and azimuths (bruce julian, usgs menlo park, ca)
c
c      the geocentric distance delta and azimuth az0 from point (lat0, lon0) to point (lat1, lon1) are calculted from
c            cos(delta) = cos(lat0')*cos(lat1')*cos(lon1 - lon0) + sin(lat0')*sin(lat1')
c            sin(delta) = sqrt(a*a + b*b)
c            tan(az0) = a/b
c
c      where
c            a = cos(lat1')*sin(lon1-lon0)
c            b = cos(latn0')*sin(lat1') - sin(lat0')*cos(lat1')*cos(lon1 - lon0)
c            lat0', lat1' = geocentric latitudes of points
c            lon0, lon1 = longitudes of points
c
c      the geocentric latitude lat' is gotten from the geographic latitude lat by tan(lat') = (1 - alpha)*(1 - alpha)*tan(lat),
c      where alpha is the flattening of the ellipsoid.  see function ggtogc for conversion.
c      the back azimuth is calculated by the same formulas with (lat0', lon0) and (lat1', lon1) interchanged.
c      azimuth is measured clockwise from north thru east.
c
      subroutine refpt (olat, olon)
c
      real              olat                            
c							! origin latitude in radians
      real              olon                            
c							! origin longitude in radians
      real              ct0                             
c							! sine of reference point latitude
      real              olatsv                            
c							! reference secondary point latitude
      real              olonsv                            
c							! reference secondary point longitude
      real              st0                             
c							! cosine of reference point latitude
      common /geocen/ st0, ct0, olonsv, olatsv
c
c refpt - store the geocentric coordinates of the refeernce point
c
c
      st0 = cos(olat)
      ct0 = sin(olat)
      olonsv = olon
      olatsv = olat
      return
      end
      subroutine delaz(lat, lon, delta, az0, az1, x, y)
c
c delaz - calculate the geocentric distance, azimuths
c
      real              az0                             
c							! azimuth from reference point to secondary point in radians
      real              az1                             
c							! azimuth from secondary point to reference point in radians
      real              cdelt                           
c							! sine of delta to secondary point
      real              cdlon                           
c							! cosine of difference of secondary point, reference longitude
      real              colat                           
c							! average colatitude of station
      real              ct0                             
c							! sine of reference point latitude
      real              ct1                             
c							! sine of secondary point latitude
      real              delta                           
c							! geocentric distance in degrees
      real              erad                            
c							! equatorial radius (chovitz, 1981, eos, 62, 65-67)
      real              flat                            
c							! earth flattening constant (chovitz, 1981, eos, 62, 65-67)
      real              lambda                          
c							! dummy variable
      real              lat                             
c							! latitude in radians
      real              lon                             
c							! longitude in radians
      real              olatsv                            
c							! origin latitude in radians
      real              olonsv                            
c							! reference secondary point longitude
      real              pi                              
c							! 3.14159...
      real              radius                          
c							! earth radius at colat
      real              sdelt                           
c							! cosine of delta to secondary point
      real              sdlon                           
c							! sine of difference of secondary point, reference longitude
      real              st0                             
c							! cosine of reference point latitude
      real              st1                             
c							! cosine of secondary point latitude
      real              twopi                           
c							! 2*pi
      real              x                               
c							! east-west distance (km)
      real              y                               
c							! north-south distance (km)
c
      parameter (pi = 3.1415926535897, twopi = 2.*pi)
      parameter (flat = 1./298.257, erad = 6378.137)
      parameter (lambda = flat*(2. - flat)/(1. - flat)**2)
c
      common /geocen/ st0, ct0, olonsv, olatsv
c
      ct1 = sin(lat)
      st1 = cos(lat)
      if ((ct1 - ct0) .eq. 0. .and. (lon - olonsv) .eq. 0.) then
        delta = 0.
        az0 = 0.
        az1 = 0.
      else
        sdlon = sin(lon - olonsv)
        cdlon = cos(lon - olonsv)
        cdelt = st0*st1*cdlon + ct0*ct1
        call cvrtop (st0*ct1 - st1*ct0*cdlon, st1*sdlon, sdelt, az0)
        delta = atan2(sdelt, cdelt)
        call cvrtop (st1*ct0 - st0*ct1*cdlon, -sdlon*st0, sdelt, az1)
        if (az0 .lt. 0.0) az0 = az0 + twopi
        if (az1 .lt. 0.0) az1 = az1 + twopi
      end if
      colat = pi/2. - (lat + olatsv)/2.
      radius = erad/sqrt(1.0 + lambda*cos(colat)**2)
      y = radius*delta*cos(az0)
      x = radius*delta*sin(az0)
      return
      end
       subroutine back (delta, az0, lat, lon)
c
c back - calculate geocentric coordinates of secondary point from delta, az
c
      real              az0                             
c							! azimuth from reference point to secondary point in radians
      real              cdelt                           
c							! sine of delta to secondary point
      real              ct0
c							! sine of reference point latitude
      real              ct1                             
c							! sine of secondary point latitude
      real              cz0                             
c							! cosine of azimuth to secondary point
      real              delta                           
c							! geocentric distance in degrees
      real              dlon                            
c							! azimuth in polar coordinates to secondary point ?
      real              lat                             
c							! latitude in radians
      real              lon                             
c							! longitude in radians
      real              olatsv                            
c							! reference secondary point latitude
      real              olonsv                            
c							! reference secondary point longitude
      real              pi                              
c							! 3.14159...
      real              sdelt                           
c							! cosine of delta to secondary point
      real              st0                             
c							! cosine of reference point latitude
      real              st1                             
c							! cosine of secondary point latitude
      real              twopi                           
c							! 2*pi
c
      parameter (pi = 3.1415926535897, twopi = 2.*pi)
c
      common /geocen/ st0, ct0, olonsv, olatsv
c
      sdelt = sin(delta)
      cdelt = cos(delta)
      cz0 = cos(az0)
      ct1 = st0*sdelt*cz0 + ct0*cdelt
      call cvrtop (st0*cdelt - ct0*sdelt*cz0, sdelt*sin(az0), st1, dlon)
      lat = atan2(ct1, st1)
      lon = olonsv + dlon
      if (abs(lon) .gt. pi) lon = lon - sign(twopi, lon)
c
      return
      end
      subroutine cvrtop(x, y, r, theta)
c
c cvrtop - convert from rectangular to polar coordinates (bruce julian, usgs menlo park, ca)
c
      real              x,y                             
c							! x,y rectangular coordinates
      real              r, theta                        
c							! radius, azimuth in polar coordinates
c
      r = sqrt(x*x + y*y)
      theta = atan2(y, x)
      return
      end
	real function ggtogc(lat)

c	convert from geographic to geocentric latitude (bruce r. julian, usgs menlo park, ca     13 sept 1983)

	real			lat			
c							! latitude
	real			c1			
c							! (1 - flattening)**2	

	parameter (c1 = 0.993305242609)

	ggtogc = atan2(c1*sin(lat), cos(lat))
	return
	end
