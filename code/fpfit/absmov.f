      Subroutine ABSMOV (ix, iy)
C
C...  Append the new vertex to the end of the current path
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
      Integer    ix, iy, lastix, lastiy
C
      Integer    i, j, k, ncabs, ncrel, nai
      Logical    useabs
      Character  absstr*13, relstr*13, aistr*15, abs*2, rel*2
C
C
      abs = ' m'
      Goto 1000
C
C
      Entry ABSDRW (ix, iy)
C
C
      abs = ' L'
C
 1000 Call VECSTR (ix,iy,absstr,ncabs)
      useabs = .TRUE.
      Goto 4000
C
C
      Entry OPTMOV (ix, iy, lastix, lastiy)
C
C
      abs = ' m'
      rel = ' v'
      Goto 2000
C
C
      Entry OPTDRW (ix, iy, lastix, lastiy)
C
C
      abs = ' L'
      rel = ' r'
C
C...  Absolute coordinates
 2000 Call VECSTR (ix,iy,absstr,ncabs)
C...  Relative coordinates
      Call VECSTR (ix-lastix,iy-lastiy,relstr,ncrel)
C
C...  Use the shorter of the two
      If (ncabs .lt. ncrel) Then
         useabs = .TRUE.
      Else If (ncrel .lt. ncabs) Then
         useabs = .FALSE.
C...  They're both the same length, so use absolute coordinates,
C...  provided they're on-scale
      Else If (absstr(ncabs:ncabs) .ne. '999999 999999') Then
         useabs = .TRUE.
      Else
         useabs = .FALSE.
      End If
C
 4000 If (useabs) Then
         ncabs = ncabs + 2
         absstr(ncabs-1:ncabs) = abs
         If (ISAI) Then
            nai = ncabs
            Do 4100 j = 1,nai
               aistr(j:j) = absstr(j:j)
 4100          Continue
         Else
            Call PUTSTR (absstr,ncabs)
         End If
      Else
         ncrel = ncrel + 2
         relstr(ncrel-1:ncrel) = rel
         If (ISAI) Then
            nai = ncrel
            Do 4200 j = 1,nai
               aistr(j:j) = relstr(j:j)
 4200          Continue
         Else
            Call PUTSTR (relstr,ncrel)
         End If
      End If
      If (ISAI) Then
C...     Convert coordinates from 7200 dpi to 72 dpi (knowing format is I6.3)
         i = nai + 3
         k = nai + 1
         Do 4300 j = nai,1,-1
            If (aistr(j:j) .eq. ' ') Then
C...           Insert a decimal point when loop reaches j-3
               k = j - 3
            End If
            If (j .eq. k) Then
C...           Eliminate non-significant zeroes
               If (aistr(i:i+1) .eq. '00') Then
                  i = i + 2
               Else
                  If (aistr(i+1:i+1) .eq. '0') Then
                     aistr(i+1:i+1) = aistr(i:i)
                     i = i + 1
                  End If
                  i = i - 1
                  aistr(i:i) = '.'
               End If
            End If
            i = i - 1
            aistr(i:i) = aistr(j:j)
 4300       Continue
         Call PUTSTR (aistr(i:),nai+3-i)
      End If
C
 9000 Return
C
      End


