cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c    rstemplate - very simple x-window template system for use by
c                 xgetginpt
c
      subroutine rsmaketemplate(labels,n,size)
      character*(*) labels
      dimension labels(n)
      real size
c
c     uses Plesha's xdevice calls to make a set of command boxes
c     along the right side of the xwindow
c
c     labels = array of box labels
c     n = number of boxes
c     size = size (in inches) of boxes to draw
c
      character*5 tlabs
      dimension tlabs(30)
      common/rstemplatec/tlabs
      integer ntbox
      real sizetbox
      common/rstemplate/ntbox,sizetbox
      integer jx1,jx2,jy1,jy2,jint,i,iy,offx,offy,lenc
      character*6 label
c
c     this common ties to xplot library
c
      common /xpltscale/mpx,mpy,xinch,yinch,dpi,lastxx,lastyy
      real xinch,yinch,dpi
      integer mpx,mpy,lastxx,lastyy
c
      do 10 i=1,n
        tlabs(i)=labels(i)
   10 continue
      ntbox=n
      sizetbox=size
c
c     make boxes
c
      jint=nint(size*dpi)
      jx1=mpx-jint
      jx2=mpx
      jy1=mpy-jint*n
      jy2=mpy
      call linexw(jx1,jy2,jx1,jy1,1,2)
      call linexw(jx2,jy2,jx2,jy1,1,2)
      do 15 i=1,n+1
        iy=mpy-(i-1)*jint
        call linexw(jx1,iy,jx2,iy,1,2)
  15  continue
c
c     write labels
c
      offy=nint(size*dpi/2.)-5
      do 20 i=1,n
        lenc=itlen(tlabs(i))
        label=tlabs(i)(1:lenc)//char(0)
        offx=nint(size*dpi/2.)-nint(7.*float(lenc)/2.)
        ix=jx1+offx
        iy=mpy-i*jint+offy
        call writexw(ix,iy,label)
  20  continue
c
      return
      end     
