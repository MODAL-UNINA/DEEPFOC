      Subroutine APPLST (string, list, nlist)
C
C Search for string in arbitrarily ordered list and append if not found
C
      Integer    nlist
      Character  string*(*), list*(*)
C
      Integer    i
C
C
      If (nlist .gt. 0) Then
         i = INDEX (list(1:nlist),string)
      Else
         i = 0
      End If
      If (i .eq. 0) Then
         If (nlist + LEN(string) + 1 .le. LEN(list)) Then
            nlist = nlist + 1
            list(nlist:nlist+LEN(string)-1) = string
            nlist = nlist + LEN(string)
            list(nlist:nlist) = ' '
         End If
      End If
C
      Return
C
      End


