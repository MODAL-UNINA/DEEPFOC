      subroutine tpplt1 (cx, cy, da1, dd1, hite, pi, pltpol, rad, rmax,
     & sa1, wt)
c
c plot p and t axes
c
      real              cx                              
c                                                       ! x position of circle center
      real              cy                              
c                                                       ! y position of circle center
      real              da1                             
c                                                       ! dip angle
      real              dd1                             
c                                                       ! dip direction
      real              hite                            
c                                                       ! height of p,t symbol
      real              pi                              
c                                                       ! pi
      character*1       pltpol                          
c                                                       ! flag: y(n)=do (not) plot first motion data
      real              rad                             
c                                                       ! pi/180
      real              rmax                            
c                                                       ! radius of circle
      real              sa1                             
c                                                       ! rake
      real              wt                              
c                                                       ! weight assigned to pick quality in program fpfit
c
      real              ainp                            
c                                                       ! angle of incidence of p axis
      real              aint                            
c                                                       ! angle of incidence of t axis
      real              ang                             
c                                                       ! angle of plot symbol
      real              azp                             
c                                                       ! azimuth of p axis
      real              azt                             
c                                                       ! azimuth of t axis
      character*4       blank                           
c                                                       ! blank
      real              da2                             
c                                                       ! dip angle of auxiliary plane
      real              dd2                             
c                                                       ! dip direction of auxiliary plane
      real              sa2                             
c                                                       ! strike of auxiliary plane
c
      parameter (ang = 0.0, blank = '    ')
c
c find auxilliary plane
c
      call auxpln (dd1, da1, sa1, dd2, da2, sa2)
c
c find p and t axes
c
      call tandp (ainp, aint, azp, azt, da1, da2, dd1, dd2, sa1,
     & sa2, pi, rad)
c
c plot symbols
c
      call pltsm1 (aint, azt, cx, cy, hite, blank, pi, rad, rmax, 'T',
     & wt, 0)
      if (pltpol .eq. 'y') call pltsm1 (ainp, azp, cx, cy, hite, blank,
     & pi, rad, rmax, 'P', wt, 0)
c
      return
      end
