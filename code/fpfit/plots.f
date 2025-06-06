cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Subroutine PLOTS (ibuf, nloc, ldev)

      Implicit   None

      integer iplotr
      integer nxplotr, npplotr
      common /npswitch/ nxplotr, npplotr

C     nxplotr = 1 - x window plotting
C     npplotr = 1 - Adobe Illus (postscript) plotting

      Integer    ibuf(*), nloc, ldev

      real scalefactor
      common/factor0/ scalefactor

      if (scalefactor.eq.0.0)  scalefactor = 1.0

c      print *, ' ** in plots... nx,np=', nxplotr, npplotr

      if (nxplotr.eq.0 .and. npplotr.eq.0) then
 80     print *,' '
        print *,'Enter 1 for X window plot'
        print *,'      2 for Postscript file'
        print *,'      3 for both'
        print *,'      0 to quit plotting'
        read(5,*)iplotr
        if (iplotr .eq. 1) then
          nxplotr = 1
          npplotr = 0
        else if (iplotr .eq. 2) then
          nxplotr = 0
          npplotr = 1
        else if (iplotr .eq. 3) then
          nxplotr = 1
          npplotr = 1
        else if (iplotr .eq. 0) then
          nxplotr = 0
          npplotr = 0
        else
          print *, ' ** Value not understood, try again...'
          goto 80
        endif
      endif

      if (nxplotr.eq.1) then
        call nxplots(ibuf,nloc,ldev)
        call flushxw()
      endif
      if (npplotr.eq.1) then
        call npplots(ibuf,nloc,ldev)
      end if

      Return

      End
