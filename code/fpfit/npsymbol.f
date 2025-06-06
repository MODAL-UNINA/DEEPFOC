      Subroutine npSYMBOL (x, y, hgt, itext, angle, nc)
C
      External   PLOT
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
C...  Scale factor to convert Hershey hgt to PostScript font point size
      Real       FUDGE
      Parameter  ( FUDGE  = 1.5 )
C...  Fractional X kerning to make PostScript font spacing closer to Hershey
      Real       XKERN
      Parameter  ( XKERN  = .06 )
C...  To deal with IBM AIX's C-style escape character nonsense (ASCII "\")
      Integer    BSLASH, ENDASH
      Parameter  ( BSLASH  =  92 )
      Parameter  ( ENDASH  = 177 )
C
      Real       x, y, hgt, angle
      Integer*4  itext(*)
      Integer    nc
C
      Logical    newfon
      Common /AFMC0M/  newfon
      Save   /AFMC0M/
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
      Real       align, rangle, rsin, rcos, fscale, xs0, ys0, xs, ys
      Real       bx(2), by(2), bbx(2), bby(2), cpx, cpy, savex, savey
      Integer    i, j, n, istr, nch, nsp, nstr, afnch
      Character  cbuf*255, afname*80, ch
C
C
      align = 0.0
      Goto 1000
C
C
      Entry CENTXT (x, y, hgt, itext, angle, nc)
C
C
      align = -0.5
      Goto 1000
C
C
      Entry RGTTXT (x, y, hgt, itext, angle, nc)
C
C
      align = -1.0
C
C
C...  No calls allowed before Call PLOTS (0,0,0) or after Call PLOT (x,y,999)
 1000 If (.not. isopen) Then
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
C...  Make sure string will fit in cbuf
      nch = MIN (nc,255)
C
C...  Restore previous X or Y coordinate if special call (=999)
      If (x .eq. 999.) Then
         xs0 = x999
      Else
         xs0 = x + orgx
      End If
      If (y .eq. 999.) Then
         ys0 = y999
      Else
         ys0 = y + orgy
      End If
C
C...  Expand markers
C
      If (nch .lt. 0) Then
         If (ISAI) Then
C...        Group graphic elements into a single composite object
            Call PUTSTR ('u',1)
         End If
C...     To prevent adding origin offset again in SUBROUTINE PLOT
         savex = orgx
         savey = orgy
         orgx = 0.0
         orgy = 0.0
         Call PLSYMB (xs0,ys0,hgt,itext,angle,nch,PLOT)
C...     Restore origin offset
         orgx = savex
         orgy = savey
         If (ISAI) Then
C...        Is there a path open?
            If (kount .gt. 0) Then
C...           Yes, draw the current path
               Call PUTSTR ('S',1)
               kount = 0
            End If
C...        End of composite object
            Call PUTLIN ('U',1)
         End If
C
C...  Use PostScript fonts for strings
C
      Else
C
C...     If font or point size has changed since last call, select new font
         fscale = hgt * dpi * FUDGE
         If ((newfon) .or. (fscale .ne. pscale)) Then
            Call AFMFON (afname,afnch)
            n = pfonch
            Call APPLST (afname(1:afnch),pfonts,pfonch)
            If (.not. ISEPSF) Then
               If (pfonch .gt. n) Then
                  Call PUTSTR ('%%IncludeResource: font',23)
                  If (ipctr+afnch .gt. LPBUF) Then
                     Call PUTLIN (' ',0)
                     Call PUTSTR ('%%+',3)
                  End If
                  Call PUTLIN (afname,afnch)
               End If
            End If
            Call PUTSTR ('/'//afname,afnch+1)
            If (ISAI) Then
               Call PUTFLT (fscale/100.)
            Else
               Call PUTFLT (fscale)
            End If
            Call PUTSTR ('0',1)
            If (ISAI) Then
               Call PUTFLT (XKERN*fscale/100.)
            Else
               Call PUTFLT (XKERN*fscale)
            End If
            Call PUTLIN ('0 z',3)
            pscale = fscale
            newfon = .FALSE.
C...        Width of a "space" character (forced to be same as a digit)
            Call AFMBB ('0',1,XKERN,0.0,bx(1),by(1),bx(2),by(2),spx,cpy)
         End If
C
C...     Convert Hollerith text string into Fortran-77 character variable
         If (nch .gt. 0) Then
            Write (cbuf,'(64A4)') (itext(j), j = 1,(nch+3)/4)
         Else
            cbuf(1:1) = CHAR (itext(1))
            nch = 1
         End If
C
C...     Substitute an "endash" for ASCII "hyphen" to make it appear like a
C...     Hershey text string (PostScript "minus" is not available in the
C...     StandardEncoding Encoding vector)
         Do 1040 j = 1,nch
            If (cbuf(j:j) .eq. '-') Then
C...           PostScript "endash" in the StandardEncoding Encoding vector
               cbuf(j:j) = CHAR (ENDASH)
            End If
 1040       Continue
C
         rangle = ( ATAN(1.) / 45. ) * angle
         rsin = SIN (rangle)
         rcos = COS (rangle)
C
C...     Update bounding box dimensions
C
         xs = 0.
         ys = 0.
         istr = 1
C...     Space characters are handled as a special case to allow for alignment
C...     of digits w/ or w/o leading spaces.  Thus, SUBROUTINE AFMBB will not
C...     correctly compute the bounding-box when the string contains spaces.
 2000    nsp = 0
 2100    If (istr .le. nch) Then
            If (cbuf(istr:istr) .eq. ' ') Then
               nsp = nsp + 1
               istr = istr + 1
               Goto 2100
            End If
         End If
C...     Update current pen position using the dimensions of a "space"
         xs = xs + nsp*spx
         If (istr .gt. nch) Then
            Goto 2500
         End If
C...     Next non-blank substring
         i = istr
         nstr = 0
 2200    If (i .le. nch) Then
            If (cbuf(i:i) .ne. ' ') Then
               nstr = nstr + 1
               i = i + 1
               Goto 2200
            End If
         End If
         Call AFMBB (cbuf(istr:istr+nstr-1),nstr,XKERN,0.0,
     1               bx(1),by(1),bx(2),by(2),cpx,cpy)
         If (istr .eq. 1) Then
            bbx(1) = bx(1)
            bby(1) = by(1)
            bbx(2) = bx(2)
            bby(2) = by(2)
         Else
            bbx(1) = MIN (bbx(1),bx(1))
            bby(1) = MIN (bby(1),by(1))
            bbx(2) = MAX (bbx(2),bx(2))
            bby(2) = MAX (bby(2),by(2))
         End If
         xs = xs + cpx
         ys = ys + cpy
         istr = istr + nstr
         Goto 2100
C
C...     Adjust starting pen position for left, right, or centered
C
 2500    xs0 = xs0 + align * ( hgt * FUDGE * ( xs*rcos - ys*rsin ) )
         ys0 = ys0 + align * ( hgt * FUDGE * ( xs*rsin + ys*rcos ) )
C
C...     Update bounding box dimensions
C
         Do 2700 j = 1,2
            Do 2600 i = 1,2
               xs = xs0 + hgt * FUDGE * ( bbx(i)*rcos - bby(j)*rsin )
               ys = ys0 + hgt * FUDGE * ( bbx(i)*rsin + bby(j)*rcos )
               If (.not. dirty) Then
                  pxlow = xs
                  pylow = ys
                  pxhi  = xs
                  pyhi  = ys
C...              Mark this frame dirty
                  dirty = .TRUE.
               Else
                  pxlow = MIN (xs,pxlow)
                  pylow = MIN (ys,pylow)
                  pxhi  = MAX (xs,pxhi )
                  pyhi  = MAX (ys,pyhi )
               End If
 2600          Continue
 2700       Continue
C
C...     Draw the string
C
 3000    xs = xs0
         ys = ys0
         istr = 1
 3100    If (ISAI) Then
C
C...        Special case spaces for Adobe Illustrator since PostScript fonts do
C...        not have an "enspace" character to allow for alignment of digits w/
C...        or w/o leading spaces, and Illustrator does not allow us to kern the
C...        space character as a special case (as does the PostScript "awidth-
C...        show" operator)
C
            nsp = 0
 3200       If (istr .le. nch) Then
               If (cbuf(istr:istr) .eq. ' ') Then
                  nsp = nsp + 1
                  istr = istr + 1
                  Goto 3200
               End If
            End If
            If (nsp .gt. 0) Then
C...           Update current pen position using the dimensions of a "space"
               xs = xs + ( nsp * ( hgt * FUDGE * spx*rcos ) )
               ys = ys + ( nsp * ( hgt * FUDGE * spx*rsin ) )
            End If
C
C...        Next non-blank substring
C
            If (istr .gt. nch) Then
               Goto 8000
            End If
            i = istr
            nstr = 0
 3300       If (i .le. nch) Then
               If (cbuf(i:i) .ne. ' ') Then
                  nstr = nstr + 1
                  i = i + 1
                  Goto 3300
               End If
            End If
            Call AFMBB (cbuf(istr:istr+nstr-1),nstr,XKERN,0.0,
     1                  bx(1),by(1),bx(2),by(2),cpx,cpy)
C
         Else
C
C...        Standard PostScript (uses the "awidthshow" operator for strings)
C
            nstr = nch
C
         End If
C
         Call PUTSTR ('[',1)
         Call PUTFLT ( rcos)
         Call PUTFLT ( rsin)
         Call PUTFLT (-rsin)
         Call PUTFLT ( rcos)
         If (ISAI) Then
            Call PUTFLT (NINT(dpi * xs)/100.)
            Call PUTFLT (NINT(dpi * ys)/100.)
         Else
            If ((x .eq. 999.) .or. (istr .gt. 1)) Then
               Call PUTSTR ('x999',4)
            Else
               Call PUTINT (NINT(dpi * xs))
            End If
            If ((y .eq. 999.) .or. (istr .gt. 1)) Then
               Call PUTSTR ('y999',4)
            Else
               Call PUTINT (NINT(dpi * ys))
            End If
         End If
         Call PUTSTR ('] e',3)
         Call PUTINT (nstr)
C...     Make sure there's enough room for at least a one-character string
C...     (" (\xxx)")
         If (ipctr .gt. 1) Then
            If (ipctr .ge. LPBUF-8) Then
               Call FLUSH_it
            End If
         End If
C...     Substitute "\(" and "\)" for parentheses and split long strings across
C...     multiple lines using PostScript end-of-line continuation ("\<EOL>")
         Do 6300 j = 0,nstr+1
            If (ipctr .gt. 1) Then
               If (ipctr .ge. LPBUF-4) Then
                  pbuf(ipctr:ipctr) = CHAR (BSLASH)
                  ipctr = ipctr + 1
                  Call FLUSH_it
               End If
            End If
            If (j .eq. 0) Then
C...           PostScript string object begins with "("
               pbuf(ipctr:ipctr) = ' '
               ipctr = ipctr + 1
               ch = '('
            Else If (j .eq. nstr+1) Then
C...           PostScript string object ends with ")"
               ch = ')'
            Else
               ch = cbuf(istr+j-1:istr+j-1)
               If ((ch .eq. '(') .or. (ch .eq. ')')) Then
C...              Put PostScript escape character ("\") in front of "(" and ")"
                  pbuf(ipctr:ipctr) = CHAR (BSLASH)
                  ipctr = ipctr + 1
               Else If (ISAI .and. (ch .eq. CHAR(ENDASH))) Then
C...              Replace PostScript "endash" with Adobe Illustrator "endash"
C...              (Adobe Illustrator uses its own (MacIntosh) encoding vector)
                  ch = CHAR (208)
               End If
            End If
C...        Substitute PostScript escape sequence if character is not printable
            n = ICHAR (ch)
            If ((n .lt. ICHAR(' ')) .or. (n .gt. ICHAR('~'))) Then
               i = ipctr + 3
               pbuf(i:i) = CHAR (MOD(n,8)+ICHAR('0'))
               i = i - 1
               n = n / 8
               pbuf(i:i) = CHAR (MOD(n,8)+ICHAR('0'))
               i = i - 1
               n = n / 8
               pbuf(i:i) = CHAR (    n   +ICHAR('0'))
               i = i - 1
               pbuf(i:i) = CHAR (BSLASH)
               ipctr = ipctr + 4
            Else
               pbuf(ipctr:ipctr) = ch
               ipctr = ipctr + 1
            End If
 6300       Continue
         Call PUTLIN ('t T',3)
C
         If (ISAI) Then
            xs = xs + ( hgt * FUDGE * ( cpx*rcos - cpy*rsin ) )
            ys = ys + ( hgt * FUDGE * ( cpx*rsin + cpy*rcos ) )
            istr = istr + nstr
            Goto 3100
         End If
C
      End If
C
 8000 x999 = xs
      y999 = ys
C
 9000 Return
C
      End


