cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Subroutine NEWPEN (ipen)

      Implicit   None

      Integer    ipen

      common /npswitch/ nxplotr, npplotr
      integer nxplotr, npplotr


      if (nxplotr.eq.1) then
        call nxnewpen(ipen)
        call flushxw()
      endif
      if (npplotr.eq.1) then
        call npnewpen(ipen)
      end if

      Return
      End
