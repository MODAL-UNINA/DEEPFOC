cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Subroutine CLOSEPLOT ()
      Implicit   None
      integer iplotr
      integer nxplotr, npplotr
      common /npswitch/ nxplotr, npplotr
      
      print *,' '
      print *,'Enter 0 to quit plotting'
      read (5,*) iplotr

      call sleep (iplotr)

c     close plot
      call plot(0.,0.,999.)

      return
      end
