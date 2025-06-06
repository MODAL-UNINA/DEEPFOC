cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Subroutine NEWPLOT (iplotr)
      Implicit   None
      integer iplotr
      integer nxplotr, npplotr
      common /npswitch/ nxplotr, npplotr
      
c     close old plot (except X window)
      if (nxplotr.ne.1) then
        call plot(0.,0.,999)
      end if

 80     print *,' '
        print *,'Enter 1 for x window plot'
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
        else if (iplotr .eq. 4) then
          nxplotr = 0
          npplotr = 0
        else
          print *, ' ** Value not understood, try again...'
          goto 80
        endif

c     open new plot
      if ((nxplotr.ne.0).or.(npplotr.ne.0)) then
        call plots(0,0,0)
      endif

      return
      end
