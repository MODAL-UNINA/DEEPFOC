      Subroutine npPLOT (x, y, ipen)
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
      Integer    LPBUF, MAXVRT
      Parameter  ( LPBUF  =  80 )
      Parameter  ( MAXVRT = 256 )
C
      Real       x, y
      Integer    ipen
C
      Integer    ipctr
      Logical    isopen
      Common /IOC0M/  ipctr, isopen
      Save   /IOC0M/
C
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
      Real       xx, yy, xlast, ylast
      Save       xlast, ylast
      Integer    i, j, ix, iy, ip, xfloor, yfloor, xceil, yceil
      Integer    lastix, lastiy, lastdx, lastdy
      Save       lastix, lastiy, lastdx, lastdy
C
C
C...  No calls allowed before Call PLOTS (0,0,0) or after Call PLOT (x,y,999)
C
      If (.not. isopen) Then
         Goto 9000
      End If
      If (virgin) Then
         Call PSPAGE
      End If
C
C...  Scale the x and y coordinates
C
      xx = x + orgx
      yy = y + orgy
      ix = NINT (dpi * xx)
      iy = NINT (dpi * yy)
      ip = ABS (ipen)
C
      If (ip .eq. 2) Then
C
C...     DRAW command
C
C...     Update bounding box dimensions
         If (.not. dirty) Then
            If (ismove) Then
               pxlow = xlast - bbpen
               pylow = ylast - bbpen
               pxhi  = xlast + bbpen
               pyhi  = ylast + bbpen
               pxlow = MIN (xx-bbpen,pxlow)
               pylow = MIN (yy-bbpen,pylow)
               pxhi  = MAX (xx+bbpen,pxhi )
               pyhi  = MAX (yy+bbpen,pyhi )
            Else
               pxlow = xx - bbpen
               pylow = yy - bbpen
               pxhi  = xx + bbpen
               pyhi  = yy + bbpen
            End If
C...        Mark this frame dirty
            dirty = .TRUE.
         Else
            If (ismove) Then
               pxlow = MIN (xlast-bbpen,pxlow)
               pylow = MIN (ylast-bbpen,pylow)
               pxhi  = MAX (xlast+bbpen,pxhi )
               pyhi  = MAX (ylast+bbpen,pyhi )
            End If
            pxlow = MIN (xx-bbpen,pxlow)
            pylow = MIN (yy-bbpen,pylow)
            pxhi  = MAX (xx+bbpen,pxhi )
            pyhi  = MAX (yy+bbpen,pyhi )
         End If
C...     Appending to previous path?
         If (kount .gt. 0) Then
C...        Yes, room for more vertices in current path?
            If (kount .ge. MAXVRT) Then
C...           No, draw the current path
               Call PUTLIN ('S',1)
               kount = 0
            End If
         End If
C...     Move to the previous vertex first if starting a new path (must
C...     be absolute coordinates) or if previous call was a MOVE.
         If (kount .eq. 0) Then
            Call ABSMOV (lastix,lastiy)
            kount = kount + 1
         Else If ((.not. ISAI) .and. ismove) Then
            Call OPTMOV (lastix,lastiy,lastdx,lastdy)
            kount = kount + 1
         End If
C...     Append the new vertex to the end of the current path
         If (ISAI) Then
            Call ABSDRW (ix,iy)
         Else
            Call OPTDRW (ix,iy,lastix,lastiy)
         End If
         kount = kount + 1
         ismove = .FALSE.
C
      Else If ((ip .eq. 3) .or. (ip .eq. 999)) Then
C
C...     MOVE or END PLOT command
C
C...     Is there a path open?
         If (kount .gt. 0) Then
C...        Is it Abobe Illustrator (absolute MOVEs/DRAWs only)?
            If (ISAI) Then
C...           Yes, draw the current path
               Call PUTLIN ('S',1)
               kount = 0
            Else
C...           No, if last op this path was a DRAW, save the coordinates
               If (.not. ismove) Then
                  lastdx = lastix
                  lastdy = lastiy
               End If
            End If
         End If
         xlast  = xx
         ylast  = yy
         ismove = .TRUE.
C
C...     Print the page if this is the END OF FRAME/RUN
C
         If (ip .eq. 999) Then
            If (.not. virgin) Then
C...           Is there a path open?
               If (kount .gt. 0) Then
C...              Yes, draw the current path
                  Call PUTLIN ('S',1)
                  kount = 0
               End If
C...           Restore VM to start-of-page state (done in "everypage")
               If (.not. ISEPSF) Then
                  Call PUTLIN ('restore',7)
               End If
C...           Display the page (even if something has not been written on it)
               If (.not. ISAI) Then
C...              Technically, Encapsulated PostScript files shouldn't have any
C...              "showpage"'s in them.  But the specification recognizes that
C...              too many documents already have "showpage" in them, so it is
C...              up to the importing application to redefine "showpage" so its
C...              effect is benign.  The advantage in having "showpage" in the
C...              file is that the document can be previewed by sending the file
C...              to a PostScript device, just like a standard PostScript file.
                  Call PUTLIN ('showpage',8)
               End If
               If (.not. ISEPSF) Then
                  Call PUTLIN ('%%PageTrailer',13)
                  xfloor = INT (pxlow*72)
                  yfloor = INT (pylow*72)
                  xceil  = INT (pxhi*72)
                  If (xceil .lt. pxhi*72) Then
                     xceil = xceil + 1
                  End If
                  yceil  = INT (pyhi*72)
                  If (yceil .lt. pyhi*72) Then
                     yceil = yceil + 1
                  End If
                  Call PUTSTR ('%%PageBoundingBox:',18)
                  Call PUTINT (xfloor)
                  Call PUTINT (yfloor)
                  Call PUTINT (xceil )
                  Call PUTINT (yceil )
                  Call PUTLIN (' ',0)
               End If
               If (page .eq. 1) Then
                  dxlow = pxlow
                  dylow = pylow
                  dxhi  = pxhi
                  dyhi  = pyhi
                  dfonch = 0
               Else
                  dxlow = MIN (pxlow,dxlow)
                  dylow = MIN (pylow,dylow)
                  dxhi  = MAX (pxhi, dxhi )
                  dyhi  = MAX (pyhi, dyhi )
               End If
C
               If (.not. ISEPSF) Then
                  Call PUTSTR ('%%PageResources:',16)
                  If (pfonch .gt. 1) Then
                     Call PUTSTR ('font',4)
                  End If
               End If
               i = 1
 2000          If (i .lt. pfonch) Then
                  j = INDEX (pfonts(i:pfonch),' ')
                  If (.not. ISEPSF) Then
                     If (ipctr+i+j-2 .gt. LPBUF) Then
                        Call PUTLIN (' ',0)
                        Call PUTSTR ('%%+ font',8)
                     End If
                     Call PUTSTR (pfonts(i:i+j-2),j-1)
                  End If
                  Call APPLST (pfonts(i:i+j-2),dfonts,dfonch)
                  i = i + j
                  Goto 2000
               End If
               If (.not. ISEPSF) Then
                  Call PUTLIN (' ',0)
               End If
C
               lastix = 0
               lastiy = 0
               virgin = .TRUE.
            End If
C
         End If
C
C...     Flush buffers and close file if this is the END OF RUN
C...     (or the END OF FRAME in the Encapsulated PostScript version)
C
         If (ipen .eq. 999) Then
            Call PSTRLR
            Goto 9000
         End If
C
      Else
C
C...     Ignore unrecognized ipen
C
         Goto 9000
C
      End If
C
C...  Save last coordinate positions for initiation of a sequence of DRAW
C...  commands or the continuation of DRAW commands that span multiple paths.
      lastix = ix
      lastiy = iy
C...  Re-origin, if requested
      If ((ipen .eq. -2) .or. (ipen .eq. -3)) Then
         orgx = xx
         orgy = yy
      End If
C
 9000 Return
C
      End


