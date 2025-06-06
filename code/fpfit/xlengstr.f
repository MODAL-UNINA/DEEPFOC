cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      real function xlengstr (string)

c     Calculates the length of a string in the existing Adobe font.
c      (For 1 inch high characters?)

      character string*(*)
      parameter (fudge = 1.5)

      nc = lentrue (string)

      call AFMBB (string, nc, xlow, ylow, xhi, yhi)

      xlengstr = fudge * (xhi - xlo)

      return
      end


