c
c     rsintemplate - return the command string if nx,ny points are in template
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine rsintemplate(nx,ny,tcom,mode)
c
c     nx,ny = position on screen
c     tcom = string to return command in
c     mode = 0 if in template
c            1 if not in template
c
      integer nx,ny,mode
      character*(*) tcom
c
      character*5 tlabs
      dimension tlabs(30)
      common/rstemplatec/tlabs
      integer ntbox
      real sizetbox
      common/rstemplate/ntbox,sizetbox
      integer jx1,jx2,jy1,jy2,jint,i,lenc
c
c     this common ties to xplot library
c
      common /xpltscale/mpx,mpy,xinch,yinch,dpi,lastxx,lastyy
      real xinch,yinch,dpi
      integer mpx,mpy,lastxx,lastyy
c
c     print *,'RSINTEMPLATE: start'
c
      jint=nint(sizetbox*dpi)
      jx1=mpx-jint
      jx2=mpx
      jy1=mpy-jint*ntbox
      jy2=mpy
c
      mode=1
      if ((nx.ge.jx1).and.(nx.le.jx2).and.(ny.ge.jy1).and.(ny.le.jy2))
     & then
        mode=0
        lenc=max(len(tcom),5)
        i=1+int(real(jy2-ny)/real(jint))
        tcom=tlabs(i)(1:lenc)
      end if
c     print *,'RSINTEMPLATE: end'
      return
      end

