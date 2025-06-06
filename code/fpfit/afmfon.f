      Subroutine AFMFON (string, nc)
C
      Real       ax, ay, xlow, ylow, xhi, yhi, cpx, cpy
      Integer    nc
      Character  string*(*)
C
C...  Device font metrics in Block Data AFMBD
C
      Real       fm(6,0:255)
      Integer    nfname
      Character  fname*80
      Common /AFMCM1/  nfname, fm
      Common /AFMCM2/  fname
C
      Real       x0, y0
      Integer    j, ic
C
C
C...  Return name of loaded font
C
      string = fname(1:nfname)
      nc     = nfname
      Return
C
C
      Entry AFMBB (string, nc, ax, ay, xlow, ylow, xhi, yhi, cpx, cpy)
C
C
C...  Return bounding-box dimensions for string
C
      xlow = 0.0
      ylow = 0.0
      xhi  = 0.0
      yhi  = 0.0
      x0 = 0.0
      y0 = 0.0
      Do 2000 j = 1,nc
         ic = ICHAR (string(j:j))
C...     Substitute ASCII Space character if illegal value from ICHAR
         If ((ic .lt. 0) .or. (ic .gt. 255)) Then
            ic = 32
         End If
         xlow = MIN (x0+fm(3,ic),xlow)
         ylow = MIN (y0+fm(4,ic),ylow)
         xhi  = MAX (x0+fm(5,ic),xhi )
         yhi  = MAX (y0+fm(6,ic),yhi )
         x0 = x0 + fm(1,ic) + ax
         y0 = y0 + fm(2,ic) + ay
 2000    Continue
      cpx = x0
      cpy = y0
      Return
C
      End


