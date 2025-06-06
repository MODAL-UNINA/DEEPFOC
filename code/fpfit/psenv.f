

      Subroutine PSENV
C
C...  Setup device name and other parameters in PostScript.com using environment
C...  variables:
C
C...  POSTSCRIPT_DEVICE_NAME     No default -- must be specified as:
C
C...                             ILLUSTRATOR  Adobe Illustrator 88
C...                             EPSF         Encapsulated PostScript V3.0
C...                             LINOTRONIC   Linotype Linotronic (7200 dpi)
C...                             LASERWRITER  Apple LaserWriter (300 dpi)
C...                             Other        Arbitrary PostScript device
C
C...  POSTSCRIPT_FILE_NAME       If not specified, defaults to:
C
C...                             ILLUSTRATOR  Illustrator.epsf
C...                             EPSF         PostScript.epsf
C...                             LINOTRONIC   Linotronic.ps
C...                             LASERWRITER  LaserWriter.ps
C...                             Other        POSTSCRIPT_DEVICE_NAME.ps
C
C...  POSTSCRIPT_DEVICE_DPI      For Other device only, defaults to 7200
C
C...  ICHANL/JCHANL              Fortran unit numbers; allocated dynamically
C...                                by INTEGER FUNCTION ALLOCU.
C
C
      Implicit   None
C
C     Include    'PostScript.com'
C
C PostScript.com -- Include file for PostScript device selection at run-time.
C
      Character  DEVICE*32, DEVNAM*80
      Integer    DEVDPI, ICHANL, JCHANL
      Logical    ISEPSF, ISAI
      Common /PSDEV1/ DEVICE, DEVNAM
      Common /PSDEV2/ DEVDPI, ICHANL, JCHANL, ISEPSF, ISAI
      Save   /PSDEV1/, /PSDEV2/
C
C
      Integer    ALLOCU
C
      Integer    EOS
      Parameter  ( EOS = 0 )
C
      Integer    i, j, nd, nf, np, lun
      Character  psdev*32, psfile*80, psdpi*8, ch
C
C
      Call GETENV ('POSTSCRIPT_DEVICE_NAME',psdev)
      DEVICE = ' '
      nd = 0
      Do 1000 j = 1,LEN(psdev)
         ch = psdev(j:j)
         If (ch .eq. CHAR(EOS)) Then
            Goto 1100
         End If
         If ( LGE(ch,'a') .and. LLE(ch,'z') ) Then
            ch = CHAR ( ICHAR(ch) - ICHAR('a') + ICHAR('A') )
         End If
         DEVICE(j:j) = ch
         If (ch .ne. ' ') Then
            nd = j
         End If
 1000    Continue
 1100 If (nd .eq. 0) Then
         DEVICE = 'LASERWRITER'
         nd = 11
      End If
C
      Call GETENV ('POSTSCRIPT_FILE_NAME',psfile)
      DEVNAM = ' '
      nf = 0
      Do 1200 j = 1,LEN(psfile)
         ch = psfile(j:j)
         If (ch .eq. CHAR(EOS)) Then
            Goto 1300
         End If
         DEVNAM(j:j) = ch
         If (ch .ne. ' ') Then
            nf = j
         End If
 1200    Continue
C
 1300 ISAI   = DEVICE .eq. 'ILLUSTRATOR'
      ISEPSF = ISAI .or. ( DEVICE .eq. 'EPSF' )
C
      If (ISAI) Then
C
C Illustrator.inc - PostScript.for include file for Adobe Illustrator.
C
C     Character  DEVICE*(*), DEVNAM*(*)
C     Integer    DEVDPI
C     Parameter  (DEVICE='ILLUSTRATOR')
C     Parameter  (DEVNAM='Illustrator.epsf')
C     Parameter  (DEVDPI=7200)
C
         If (nf .eq. 0) Then
            DEVNAM = 'Illustrator.epsf'
         End If
         DEVDPI = 7200
C
      Else If (ISEPSF) Then
C
C EPSF.inc - PostScript.for include file for Encapsulated PostScript.
C
C     Character  DEVICE*(*), DEVNAM*(*)
C     Integer    DEVDPI
C     Parameter  (DEVICE='EPSF')
C     Parameter  (DEVNAM='PostScript.epsf')
C     Parameter  (DEVDPI=7200)
C
         If (nf .eq. 0) Then
            DEVNAM = 'PostScript.epsf'
         End If
         DEVDPI = 7200
C
      Else If (DEVICE .eq. 'LINOTRONIC') Then
C
C Linotronic.inc - PostScript.for include file for the Linotype Series 100.
C
C     Character  DEVICE*(*), DEVNAM*(*)
C     Integer    DEVDPI
C     Parameter  (DEVICE='LINOTRONIC')
C     Parameter  (DEVNAM='Linotronic.ps')
C     Parameter  (DEVDPI=7200)
C
         If (nf .eq. 0) Then
            DEVNAM = 'Linotronic.ps'
         End If
         DEVDPI = 7200
C
      Else If (DEVICE .eq. 'LASERWRITER') Then
C
C LaserWriter.inc - PostScript.for include file for the Apple LaserWriter.
C
C     Character  DEVICE*(*), DEVNAM*(*)
C     Integer    DEVDPI
C     Parameter  (DEVICE='LASERWRITER')
C     Parameter  (DEVNAM='LaserWriter.ps')
C     Parameter  (DEVDPI=300)
C
         If (nf .eq. 0) Then
            DEVNAM = 'LaserWriter.ps'
         End If
         DEVDPI = 300
C
      Else
C
         If (nf .eq. 0) Then
            DEVNAM = psdev(1:nd) // '.ps'
         End If
         Call GETENV ('POSTSCRIPT_DEVICE_DPI',psdpi)
         np = 0
         Do 1400 j = 1,LEN(psdpi)
            If (psdpi(j:j) .eq. CHAR(EOS)) Then
               Goto 1500
            End If
            If (ch .ne. ' ') Then
               np = j
            End If
 1400       Continue
 1500    DEVDPI = 0
         If (np .gt. 0) Then
            i = LEN(psdpi)
            Do 1600 j = np,1,-1
               psdpi(i:i) = psdpi(j:j)
               i = i - 1
 1600          Continue
            psdpi(1:i) = ' '
            Read (psdpi,'(I8)',Err=1700) DEVDPI
            Goto 1800
 1700       DEVDPI = 0
 1800       Continue
         End If
         If (DEVDPI .le. 0) Then
            DEVDPI = 7200
         End If
C
      End If
C
      lun = ALLOCU (ICHANL)
      lun = ALLOCU (JCHANL)
C
      Return
C
      End


