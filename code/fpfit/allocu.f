      Function ALLOCU (lun)
C
C
C...  Subroutines  to  allocate,  deallocate,  and  reserve Fortran logical unit
C...  numbers (luns).  By default, units 21  through  99  are  in  the  pool  of
C...  available  luns.   To  specifically  reserve  other  luns, Call RESRVU and
C...  inspect the return status to determine if the lun was free at the time  of
C...  the  call.   To  make a lun available to the pool, Call DALOCU and inspect
C...  the return status to determine if the lun was allocated, reserved, or free
C...  at the time of the call.  To allocate an available lun, Call ALLOCU.
C
C
C     To allocate a lun:       lun  = ALLOCU (lun)
C
C          lun  - Fortran unit number allocated, if available, else -1
C
C     To deallocate a lun:     ierr = DALOCU (lun,ierr)
C
C          lun  - Fortran unit number to be deallocated (unreserved)
C          ierr - Return status:
C                   -1 = Illegal lun (<0 or >99)
C                    0 = Lun was allocated ("A")
C                    1 = Lun was already available (" ")
C                    2 = Lun was reserved ("R")
C
C     To reserve a lun:        ierr = RESRVU (lun,ierr)
C
C          lun  - Fortran unit number to be deallocated (unreserved)
C          ierr - Return status:
C                   -1 = Illegal lun (<0 or >99)
C                    0 = Lun was available (" ")
C                    1 = Lun was already reserved ("R")
C                    2 = Lun was allocated ("A")
C
C
C...  Fortran  allows  units from 0 through 99, inclusive.  We use the character
C...  in position lun+1 in luns to determine whether or not it is available  for
C...  allocation  ("  "),  if it is reserved ("R"), or if it is allocated ("A").
C...  The character is returned to the caller when the  lun  is  deallocated  to
C...  allow  a  determination  of its previous status (we don't complain here on
C...  the assumption that the user knows what he/she wants --- they can add code
C...  to validate the results themselves).
C
C
      Integer    ALLOCU, DALOCU, RESRVU
C
      Logical    isopen
      Character  luns*100
      Save       luns
C...  "Reserve" luns 1 through 20
      Data       luns(1:20)/'RRRRRRRRRRRRRRRRRRRR'/, luns(21:)/' '/
C
C
C...  Allocate a free lun
C
C
 1000 lun = INDEX(luns,' ')
C...  If there were no free luns, then INDEX returned 0
      If (lun .gt. 0) Then
         Inquire (Unit=lun-1,Opened=isopen,Err=1900)
         Goto 2000
 1900    isopen = .TRUE.
 2000    If (isopen) Then
            luns(lun:lun) = 'R'
            Goto 1000
         Else
            luns(lun:lun) = 'A'
         End If
      End If
C...  If lun is zero, it is converted to the illegal Fortran lun "-1", which
C...  is used to signal the caller that no more luns were available
      lun    = lun - 1
      ALLOCU = lun
      Return
C
C
C...  Deallocate a lun
C
C
      Entry DALOCU (lun,ierr)
C
      l = lun + 1
      If ((l .lt. 1) .or. (l .gt. LEN(luns))) Then
         ierr = -1
      Else
         If (luns(l:l) .eq. ' ') Then
            ierr = 1
         Else
            If (luns(l:l) .eq. 'A') Then
               ierr = 0
            Else
               ierr = 2
            End If
            luns(l:l) = ' '
         End If
      End If
      DALOCU = ierr
      Return
C
C
C...  Reserve a lun
C
C
      Entry RESRVU (lun,ierr)
C
      l = lun + 1
      If ((l .lt. 1) .or. (l .gt. LEN(luns))) Then
         ierr = -1
      Else
         If (luns(l:l) .eq. 'R') Then
            ierr = 1
         Else
            If (luns(l:l) .eq. ' ') Then
               ierr = 0
            Else
               ierr = 2
            End If
            luns(l:l) = 'R'
         End If
      End If
      RESRVU = ierr
      Return
C
      End


