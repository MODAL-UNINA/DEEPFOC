      Subroutine PSPAGE
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
      Logical    newfon
      Common /AFMC0M/  newfon
      Save   /AFMC0M/
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
C
      page = page + 1
      If (ISEPSF .and. (page .gt. 1)) Then
C
C...     Attempt to draw more than one page in an Encapsulated PostScript or
C...     Adobe Illustrator document
C
         Write (*,601)
  601    Format (/' Warning:  Multiple pages are not allowed in an',
     1            ' Encapsulated PostScript document.'/
     2            '           Further plotting calls will be ignored.'/)
         Call PSTRLR
         Goto 9000
      End If
C...  Initialize list of fonts use on this page
      pfonch = 0
      If (ISAI) Then
C...     Use rounded end caps and inflection elbows for polylines
         Call PUTSTR ('1 J 1 j',7)
C...     Use black for path strokes and character fills
         Call PUTLIN ('0 G 0 g',7)
      Else If (ISEPSF) Then
C...     Initialize PostScript environment for the page
         Call PUTLIN ('everypage',9)
      Else
         Call PUTSTR ('%%Page:',7)
         Call PUTINT (page)
         Call PUTINT (page)
         Call PUTLIN (' ',0)
C...     Initialize PostScript environment for next page
         Call PUTLIN ('%%BeginPageSetup',16)
*        If (DEVICE .eq. 'LASERWRITER') Then
C...        Print user name and time stamp for identification
C...        Note: this label is not included in the bounding-box
C...        6-point Courier is roughly 20 characters per inch
*           Call APPLST ('Courier',pfonts,pfonch)
*           Call PUTLIN ('%%IncludeResource: font Courier',31)
*           Call PUTSTR ('print_ident (  Page',19)
*           Call PUTINT (page)
*           Call PUTLIN (') show',6)
*        End If
         If (island) Then
            Call PUTLIN ('landscape',9)
         Else
            Call PUTLIN ('portrait',8)
         End If
         Call PUTLIN ('everypage',9)
         Call PUTLIN ('%%EndPageSetup',14)
         Call PUTLIN ('%%PageResources: (atend)',24)
         Call PUTLIN ('%%PageBoundingBox: (atend)',26)
      End If
C...  Initialize the page bounding-box for "no marks on page"
      pxlow = 0.0
      pylow = 0.0
      pxhi  = 0.0
      pyhi  = 0.0
C...  "showpage" reinitializes stroke width to 1 user unit
      bbpen = 0.5 / 72.
C...  No current font
      pscale = 0.0
      newfon = .TRUE.
C...  No current path
      kount = 0
C...  Relative origin (0,0)
      orgx = 0.
      orgy = 0.
      virgin = .FALSE.
      dirty  = .FALSE.
C
 9000 Return
C
      End


