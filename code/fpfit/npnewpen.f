      Subroutine npNEWPEN (ipen)
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
      Integer    ipen
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
      Real       rpen
      Integer    jpen
C
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
C...  Scale "pen width" so its the same as on a 300 dpi LaserWriter
C
      jpen = ipen
      If (jpen .lt. 1) Then
         jpen = 1
      End If
      If (jpen .gt. 20) Then
         jpen = 20
      End If
      rpen = jpen * (dpi/300.)
      If (ISAI) Then
         Call PUTFLT (rpen/100.)
      Else
         Call PUTFLT (rpen)
      End If
      Call PUTLIN ('w',1)
      bbpen = 0.5 * jpen/300.
C
 9000 Return
C
      End


