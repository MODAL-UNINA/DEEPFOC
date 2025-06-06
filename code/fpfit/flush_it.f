      Subroutine FLUSH_it
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
C
      If (ipctr .gt. 1) Then
         If (ISEPSF) Then
            Write (JCHANL) ipctr, pbuf
         Else
            Write (ICHANL,'(A)') pbuf(1:ipctr-1)
         End If
         ipctr = 1
      End If
C
      Return
C
      End


