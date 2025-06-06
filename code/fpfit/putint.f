      Subroutine PUTINT (i)
C
      Integer    LPBUF
      Parameter  ( LPBUF  =  80 )
C
      Real       r
      Integer    i, nc
      Character  cn*(*), c1*1
C
      Integer    ipctr, lci
      Logical    isopen
      Common /IOC0M/  ipctr, isopen
      Character  pbuf*(LPBUF)
      Common /IOC0N/  pbuf
      Save   /IOC0M/, /IOC0N/
C
      Integer    j, ni
      Logical    newlin
C...  NOTE:  LEN(ci) must be the same number of characters written by
C...  Format statements 101 and 102
      Character  ci*8
C
C
      Write (ci,101) i
C...  NOTE:  Format 101 must write LEN(ci) characters
  101 Format (I8)
      lci = LEN(ci)
      Goto 1000
C
C
      Entry PUTFLT (r)
C
C
      Write (ci,102) r
C...  NOTE:  Format 102 must write LEN(ci) characters
  102 Format (F8.3)
      lci = LEN(ci)
  990 If (ci(lci:lci) .eq. '0') Then
         lci = lci - 1
         Goto 990
      End If
      If (ci(lci:lci) .eq. '.') Then
         lci = lci - 1
C...     A leading zero before the period is optional in Fortran-77
C...     Correct for r=0 and leading zeros suppressed
         If (ci(lci:lci) .eq. ' ') Then
            ci(lci:lci) = '0'
         End If
      End If
C
 1000 Do 1100 j = lci,1,-1
         If (ci(j:j) .eq. ' ') Then
            Goto 1200
         End If
 1100    Continue
C...  Fortran-77 specifies j = 0 at this point
C     j = 0
 1200 ni = lci - j
      If (ipctr .gt. 1) Then
         If (ipctr+ni .gt. LPBUF) Then
            Call FLUSH_it
         End If
      End If
      If (ipctr .gt. 1) Then
         pbuf(ipctr:ipctr) = ' '
         ipctr = ipctr + 1
      End If
      pbuf(ipctr:ipctr+ni-1) = ci(j+1:lci)
      ipctr = ipctr + ni
      Goto 9000
C
C
      Entry PUTSTR (cn, nc)
C
C
      newlin = .FALSE.
      Goto 3000
C
C
      Entry PUTLIN (cn, nc)
C
C
      newlin = .TRUE.
C
 3000 If (nc .gt. 0) Then
         If (ipctr .gt. 1) Then
            If (ipctr+nc .gt. LPBUF) Then
               Call FLUSH_it
            End If
         End If
         If (ipctr .gt. 1) Then
            pbuf(ipctr:ipctr) = ' '
            ipctr = ipctr + 1
         End If
         pbuf(ipctr:ipctr+nc-1) = cn(1:nc)
         ipctr = ipctr + nc
      End If
      If (newlin) Then
         Call FLUSH_it
      End If
      Goto 9000
C
C
      Entry PUTCHR (c1)
C
C
      If (ipctr .gt. 1) Then
         If (ipctr+1 .gt. LPBUF) Then
            Call FLUSH_it
         End If
      End If
      If (ipctr .gt. 1) Then
         pbuf(ipctr:ipctr) = ' '
         ipctr = ipctr + 1
      End If
      pbuf(ipctr:ipctr) = c1
      ipctr = ipctr + 1
C
 9000 Return
C
      End


