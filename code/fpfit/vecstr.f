      Subroutine VECSTR (x, y, str, nc)
C
C...  Size of str must accomodate two 6 digit integer coordinates
C...  (Note:  For Adobe Illustrator, the format used is I6.3 so that the
C...  coordinates can be converted from 7200 dpi to 72 dpi by inserting
C...  a decimal point before the second-to-the-last character.)
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
      Integer    x, y, nc
      Character  str*13
C
      Integer    i, j
      Character  t*1
C
C
      If (ISAI) Then
         Write (str,101,Err=9900) x, y
  101    Format (I6.3, ',', I6.3)
      Else
         Write (str,102,Err=9900) x, y
  102    Format (I6, ',', I6)
      End If
      i = 1
      Do 1000 j = 1,LEN(str)-1
         t = str(j:j)
         If (t .ne. ' ') Then
            If (t .eq. ',') Then
               t = ' '
            End If
            str(i:i) = t
            i = i + 1
         End If
 1000    Continue
 1900 str(i:) = str(LEN(str):LEN(str))
C
 9000 nc = i
      Return
C
 9900 str = '999999 999999'
      i = LEN(str)
      Goto 9000
C
      End


