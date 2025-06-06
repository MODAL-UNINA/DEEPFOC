      subroutine plotpl (cx, cy, dpidg, pi, rad, rmax, strkdg)
c
c plots fault plane on lower hemisphere stereo net
c
      real              cx                              
c                                                       ! x position of circle center
      real              cy                              
c                                                       ! y position of circle center
      real              dpidg                           
c                                                       ! dip angle in degrees
      real              pi                              
c                                                       ! pi
      real              rad                             
c                                                       ! pi/180
      real              rmax                            
c                                                       ! radius of circle
      real              strkdg                          
c                                                       ! strike angle in degrees

      real              ang                             
c                                                       ! angle in radians
      real              ainp(91)                        
c                                                       ! angle of incidence in radians
      real              arg                             
c                                                       ! dummy argument
      real              az                              
c                                                       ! azimuth
      real              con                             
c                                                       ! radius coefficient
      real              diprd                           
c                                                       ! dip angle in radians
      integer           i                               
c                                                       ! loop index
      integer           mi                              
c                                                       ! scratch index
      real              radius                          
c                                                       ! radius
      real              saz(91)                         
c                                                       ! azimuth in radians
      real              strkrd                          
c                                                       ! strike in radians
      real              taz                             
c                                                       ! scratch variable
      real              tpd                             
c                                                       ! scratch variable
      real              x                               
c                                                       ! x plot position
      real              y                               
c                                                       ! y plot position
c
      strkrd = strkdg*rad
      diprd = dpidg*rad
      tpd = tan(pi*.5 - diprd)**2
c
c case of vertical plane
c
      if (dpidg .eq. 90.0) then
        x = rmax*sin(strkrd) + cx
        y = rmax*cos(strkrd) + cy
        call plot (x, y, 3)
        x = rmax*sin(strkrd + pi) + cx
        y = rmax*cos(strkrd + pi) + cy
        call plot (x, y, 2)
        return
      end if
c
c compute angle of incidence, azimuth
c
      do 10 i = 1, 90
        ang = float(i - 1)*rad
        arg = sqrt((cos(diprd)**2)*(sin(ang)**2))/cos(ang)
        saz(i) = atan(arg)
        taz = tan(saz(i))**2
        arg = sqrt(tpd + tpd*taz + taz)
        ainp(i) = acos(tan(saz(i))/arg)
  10  continue
      saz(91) = 90.*rad
      ainp(91) = pi*.5 - diprd
c
c plot plane
c
      con = rmax*sqrt(2.)
      do 20 i = 1, 180
        if (i .le. 91) then
          mi = i
          az = saz(i) + strkrd
        else
          mi = 181 - i
          az = pi - saz(mi) + strkrd
        end if
        radius = con*sin(ainp(mi)*0.5)
        x = radius*sin(az) + cx
        y = radius*cos(az) + cy
        if (i .eq. 1) then
          call plot (x, y, 3)
        else
          call plot (x, y, 2)
        end if
20    continue
c
      return
      end
