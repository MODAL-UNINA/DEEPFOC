cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine centxt (x,y,ht,string,angle,lens)

c     Plots centered text.

      character*(*) string
      parameter (pi=3.1415927, deg2rad=0.17453293e-1, rad2deg=57.295779)

      xoff = ht * xlengstr (string)
c      print *, '  ** xoff =', xoff

      xs = x - 0.5*xoff*cos(angle*deg2rad)
      ys = y - 0.5*xoff*sin(angle*deg2rad)

      call symbol (xs,ys,ht,string,angle,lens)

      return
      end


