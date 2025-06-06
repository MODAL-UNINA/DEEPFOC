cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Subroutine PLOT (x, y, ipen)

      Implicit   None

      Real       x, y
      Real       xp, yp
      Integer    ipen

      common /npswitch/ nxplotr, npplotr
      integer nxplotr, npplotr

      real scalefactor
      common/factor0/ scalefactor

      xp = x * scalefactor
      yp = y * scalefactor

      if (nxplotr.eq.1) then
        call nxplot(xp,yp,ipen)
        call flushxw()
      endif
      if (npplotr.eq.1) then
        call npplot(xp,yp,ipen)
      end if

      Return
      End
