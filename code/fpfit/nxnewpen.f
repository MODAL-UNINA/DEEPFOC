cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Subroutine nxNEWPEN (ipen)
C
      Implicit   None
C
c     Include    'PostScript.inc'
C
c     Logical    ISAI
c     Parameter  ( ISAI   = DEVICE .eq. 'ILLUSTRATOR' )
C
      Integer    ipen
C
      Integer    jpen
C
      Integer    ipctr
      Logical    isopen
      Common /IOC0M/  ipctr, isopen
C...  If DRAW or NEWPEN, virgin: .TRUE. -> .FALSE. (new page)
C...  If DRAW, dirty: .FALSE. -> .TRUE.
c     Logical    island, virgin, dirty, ismove
c     Integer    dpi, page, kount
c     Real       pxlow, pylow, pxhi, pyhi, dxlow, dylow, dxhi, dyhi
c     Common /PSC0M/  island, virgin, dirty, ismove, dpi, page, kount,
c    1                pxlow, pylow, pxhi, pyhi, dxlow, dylow, dxhi, dyhi
c     Save   /PSC0M/
C
C...  No calls allowed before Call PLOTS (0,0,0) or after Call PLOT (x,y,999)
      If (.not. isopen) Then
         Goto 9000
      End If
C
      jpen = ipen
      jpen = mod(jpen,8)
      if (jpen.eq.0) jpen=8
      call flushxw
c      call setcolorxw(jpen)
c      call setlinexw (1,jpen,0)
      call flushxw
C
 9000 Return
C
      End
