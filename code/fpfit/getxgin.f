cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Subroutine getxgin(a,x,y,xhite,yhite,mode)

C     GETXGIN - gets a point from an x window

      Character a*(*),tcom*1
      common /xpltscale/mpx,mpy,xinch,yinch,dpi,lastxx,lastyy
      real xinch,yinch,dpi
      integer mpx,mpy,lastxx,lastyy
C
C     set Graphic Input (GIN) mode for one event
C
      call get_pointxw(nx,ny,nbutton)
C
C     decide if point is in command template
C
      mnx=nx
      mny=ny
      call rsintemplate(mnx,mny,tcom,nmode)
c     print *,'GETXGIN: tcom, mode = ',tcom,nmode
      mode=1
      if (nmode.eq.0) then
        a=tcom
        mode=0
      end if
C
C     Convert terminal coordinates to inches
C
      x=real(nx)*xinch/real(mpx)
      y=real(ny)*yinch/real(mpy)
c     print *,'GETXGIN: x,y =',x,y
      Return
      End
