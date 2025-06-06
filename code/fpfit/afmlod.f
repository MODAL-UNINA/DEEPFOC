      Subroutine AFMLOD (string, nc, afm)
C
C Load a font for the PostScript.for SYMBOL subroutine.
C
C NOTE: The application must either load the font name and metrics into COMMONs
C       /AFMCM1/ and /AFMCM2/ at run-time using AFMLOD before calling SUBROUTINE
C       SYMBOL, or use a BLOCK DATA module to initialize the commons, such as
C       Helvetica.for.
C
C     Real       afm(6,0:255)
      Real       afm(0:1535)
      Integer    nc
      Character  string*(*)
C
      Logical    newfon
      Common /AFMC0M/  newfon
      Save   /AFMC0M/
C
C     Real       fm(6,0:255)
      Real       fm(0:1535)
      Integer    nfname
      Character  fname*80
      Common /AFMCM1/  nfname, fm
      Common /AFMCM2/  fname
C
      Integer    j
C
C
      fname  = string
      nfname = nc
      Do 1000 j = 0,1535
         fm(j) = afm(j)
 1000    Continue
C...  Notify SYMBOL that a new font has been loaded
      newfon = .TRUE.
C
      Return
C
      End


