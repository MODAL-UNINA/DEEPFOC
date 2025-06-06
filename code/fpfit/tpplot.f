      subroutine tpplot (cx, cy, da1, dd1, hite, pi, rad, rmax, sa1, wt)
c
c plot p and t axes
c
      real              cx                              
c                                                       ! x position of circle center
      real              cy                              
c                                                       ! y position of circle center
      real              da1                             
c                                                       ! dip angle of principle plane
      real              dd1                             
c                                                       ! dip direction of principle plane
      real              hite                            
c                                                       ! height of p,t symbol
      real              pi                              
c                                                       ! pi
      real              rad                             
c                                                       ! pi/180
      real              rmax                            
c                                                       ! radius of circle
      real              sa1                             
c                                                       ! rake of principle plane
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
c                                                       ! dip angle of auxilliary plane
      real              dd2                             
c                                                       ! dip direction of auxilliary plane
      real              sa2                             
c                                                       ! strike of auxilliary plane
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
      call pltsym (ainp, azp, cx, cy, hite, blank, pi, rad, rmax, 'P', 
     & wt)
      call pltsym (aint, azt, cx, cy, hite, blank, pi, rad, rmax, 'T', 
     & wt)
c
      return
      end
