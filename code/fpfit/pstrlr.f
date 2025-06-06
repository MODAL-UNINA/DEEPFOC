      Subroutine PSTRLR
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
      Integer    LPBUF
      Parameter  ( LPBUF  =  80 )
C
      Integer    ipctr
      Logical    isopen
      Common /IOC0M/  ipctr, isopen
      Character  pbuf*(LPBUF)
      Common /IOC0N/  pbuf
      Save   /IOC0M/, /IOC0N/
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
      Integer    i, j, xfloor, yfloor, xceil, yceil
C
C
C...  Start of Document Trailer  .........................................
C
C
      Call PUTLIN ('%%Trailer',9)
      If (.not. ISAI) Then
         Call PUTLIN ('end',3)
      End If
      xfloor = INT (dxlow*72)
      xceil  = INT (dxhi*72)
      If (xceil .lt. dxhi*72) Then
         xceil = xceil + 1
      End If
      yfloor = INT (dylow*72)
      yceil  = INT (dyhi*72)
      If (yceil .lt. dyhi*72) Then
         yceil = yceil + 1
      End If
      If (.not. ISEPSF) Then
         Call PUTSTR ('%%BoundingBox:',14)
         Call PUTINT (xfloor)
         Call PUTINT (yfloor)
         Call PUTINT (xceil )
         Call PUTINT (yceil )
         Call PUTLIN (' ',0)
         Call PUTSTR ('%%Pages:',8)
         Call PUTINT (page)
         Call PUTLIN ('1',1)
      End If
C
      If (.not. ISEPSF) Then
         Call PUTSTR ('%%DocumentNeededResources:',26)
         If (dfonch .gt. 1) Then
            Call PUTSTR ('font',4)
         End If
      End If
      i = 1
 2000 If (i .lt. dfonch) Then
         j = INDEX (dfonts(i:dfonch),' ')
         If (.not. ISEPSF) Then
            If (ipctr+i+j-2 .gt. LPBUF) Then
               Call PUTLIN (' ',0)
               Call PUTSTR ('%%+ font',8)
            End If
            Call PUTSTR (dfonts(i:i+j-2),j-1)
         End If
         i = i + j
         Goto 2000
      End If
      If (.not. ISEPSF) Then
         Call PUTLIN (' ',0)
      End If
C
      Call PUTLIN ('%%EOF',5)
C
C
C...  End of Document Trailer    .........................................
C
C
      Call FLUSH_it
C
      If (ISEPSF) Then
C
C...     Re-read EPSF documents to fixup header
C
         End File (Unit=JCHANL)
         Rewind (Unit=JCHANL)
C
 5200    Read (JCHANL,End=5900) ipctr, pbuf
C
C...     Fixup %%BoundingBox:
C
         If ((ipctr .eq. LEN('%%BoundingBox: (atend)')+1) .and.
     1       (pbuf(1:ipctr-1) .eq. '%%BoundingBox: (atend)')) Then
            ipctr = 15
            Call PUTINT (xfloor)
            If (ISAI) Then
C...           Old versions of Adobe Illustrator don't like any spaces
C...           after the colon (see also "%%TemplateBox:0 0 0 0" above)
               Do 5300 j = 16,ipctr-1
                  pbuf(j-1:j-1) = pbuf(j:j)
 5300             Continue
               ipctr = ipctr - 1
            End If
            Call PUTINT (yfloor)
            Call PUTINT (xceil )
            Call PUTINT (yceil )
C
C...     Fixup %%DocumentNeededResources: (%%DocumentFonts: for Illustrator)
C
         Else If
     1      ((ipctr .eq. LEN('%%DocumentNeededResources: (atend)')+1)
     2         .and.
     3       (pbuf(1:ipctr-1) .eq.
     4          '%%DocumentNeededResources: (atend)')) Then
C...        Discard line if no fonts are needed
            If (dfonch .le. 0) Then
               Goto 5200
            End If
            If (ISAI) Then
               pbuf(1:17) = '%%DocumentFonts: '
               ipctr = 17
            Else
               pbuf(28:32) = 'font '
               ipctr = 32
            End If
            i = 1
 5400       If (i .lt. dfonch) Then
               j = INDEX (dfonts(i:dfonch),' ')
               If (ipctr+i+j-2 .gt. LPBUF) Then
                  Write (ICHANL,'(A)') pbuf(1:ipctr-1)
                  ipctr = 1
                  Call PUTSTR ('%%+ font',8)
               End If
               Call PUTSTR (dfonts(i:i+j-2),j-1)
               i = i + j
               Goto 5400
            End If
C
C...     Fixup %%IncludeResource: font
C
         Else If ((ipctr .eq. LEN('%%IncludeResource: font')+1) .and.
     1            (pbuf(1:ipctr-1) .eq. '%%IncludeResource: font')) Then
C...        Discard line if no fonts are needed
            If (dfonch .le. 0) Then
               Goto 5200
            End If
            i = 1
 5500       If (i .lt. dfonch) Then
               j = INDEX (dfonts(i:dfonch),' ')
               ipctr = 24
               If (ipctr+i+j-2 .gt. LPBUF) Then
                  Write (ICHANL,'(A)') pbuf(1:ipctr-1)
                  ipctr = 1
                  Call PUTSTR ('%%+',3)
               End If
               Call PUTSTR (dfonts(i:i+j-2),j-1)
               Write (ICHANL,'(A)') pbuf(1:ipctr-1)
               i = i + j
               Goto 5500
            End If
            Goto 5200
C
C...     Otherwise, pass line through as-is
C
         End If
C
         Write (ICHANL,'(A)') pbuf(1:ipctr-1)
         Goto 5200
C
 5900    Close (Unit=JCHANL)
C
      End If
      Close (Unit=ICHANL)
C
C...  Disallow any further access to the file now
C
      isopen = .FALSE.
C
      Return
C
      End


