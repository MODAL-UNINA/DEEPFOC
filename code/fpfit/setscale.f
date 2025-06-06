cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine setscale (scfactor)

c     Non-standard call to change scale.

      real scfactor
      real scalefactor
      common/factor0/ scalefactor

      scalefactor = scfactor

      return
      end
