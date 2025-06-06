cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Subroutine cursor(x,y)

      common /xpltscale/mpx,mpy,xinch,yinch,dpi,lastxx,lastyy
      real xinch,yinch,dpi
      integer mpx,mpy,lastxx,lastyy
      common /nxbakerp/xorig,yorig
      real xorig,yorig
      real scalefactor
      common/factor0/ scalefactor
C
      call get_pointxw(nx,ny,nbutton)
C
C     Convert terminal coordinates to inches
C
      x=real(nx)*xinch/real(mpx) - xorig
      y=real(ny)*yinch/real(mpy) - yorig
      x = x/scalefactor
      y = y/scalefactor
c     print *,'CURSOR: x,y =',x,y

      Return
      End
