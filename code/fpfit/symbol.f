cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Subroutine SYMBOL (x, y, hgt, text, angle, nc)

      Implicit   None

      Real       x, y, hgt, angle
      Real       xs, ys, ht
      Character*(*) text
      Integer    nc

      common /npswitch/ nxplotr, npplotr
      integer nxplotr, npplotr

      real scalefactor
      common/factor0/ scalefactor

      xs = x * scalefactor
      ys = y * scalefactor
      ht = hgt * scalefactor

      if (nxplotr.eq.1) then
        call nxsymbol(xs,ys,ht,text,angle,nc)
        call flushxw()
      endif
      if (npplotr.eq.1) then
        call npsymbol(xs,ys,ht,text,angle,nc)
      endif

      Return
      End
