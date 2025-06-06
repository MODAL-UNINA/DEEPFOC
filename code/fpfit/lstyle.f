      Subroutine LSTYLE (istyle)
C
C     Include    'PostScript.inc'
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
      Integer    MAXSTY
      Parameter  (MAXSTY = 5)
C
      Integer    istyle
C
      Integer    ipctr
      Logical    isopen
      Common /IOC0M/  ipctr, isopen
      Save   /IOC0M/
C...  If DRAW or NEWPEN, virgin: .TRUE. -> .FALSE. (new page)
C...  If DRAW, dirty: .FALSE. -> .TRUE.
      Logical    island, virgin, dirty, ismove
      Integer    dpi, page, kount, pfonch, dfonch
      Real       pxlow, pylow, pxhi, pyhi, dxlow, dylow, dxhi, dyhi
      Real       x999, y999, bbpen, pscale, spx, orgx, orgy
      Common /PSC0M/  island, virgin, dirty, ismove, dpi, page, kount,
     1                pxlow, pylow, pxhi, pyhi, dxlow, dylow, dxhi,
     2                dyhi, x999, y999, bbpen, pfonch, dfonch, pscale,
     3                spx, orgx, orgy
      Character  pfonts*1024, dfonts*1024
      Common /PSC0N/  pfonts, dfonts
      Save   /PSC0M/, /PSC0N/
C
      Integer    jstyle, j
      Real       styles(5,MAXSTY), s
      Data       styles/5*0., .016,.023,3*0., .094,.078,.031,.078,0.,
     1                  .031,.063,3*0., .125,.094,3*0./
C
C...  No calls allowed before Call PLOTS (0,0,0) or after Call PLOT (x,y,999)
      If (.not. isopen) Then
         Goto 9000
      End If
C
      If (virgin) Then
         Call PSPAGE
      Else
C...     Is there a path open?
         If (kount .gt. 0) Then
C...        Yes, draw the current path
            Call PUTLIN ('S',1)
            kount = 0
         End If
      End If
C
C...  Select one of five line styles (1..5) (1=solid)
C
      jstyle = ABS (MOD(istyle-1,MAXSTY)) + 1
C
      Call PUTSTR ('[',1)
      j = 1
 1000 s = styles(j,jstyle)
      If (s .ne. 0.) Then
         If (ISAI) Then
            Call PUTFLT (NINT(dpi * s)/100.)
         Else
            Call PUTINT (NINT(dpi * s))
         End If
         j = j + 1
         Goto 1000
      End If
      Call PUTLIN ('] 0 d',5)
C
 9000 Return
C
      End


