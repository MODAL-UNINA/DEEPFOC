C
C UserID - Construct a user identification string for the PostScript document
C          heading.
C
C Note:  This version of UserID is written for SUN SunOS.  The result string is
C        formatted as "node_name:user_name (GID UID)".
C 
C
      Subroutine USERID (for)
C
      Implicit   None
C
      External   UNAME        !$PRAGMA C ( UNAME )
      Integer*4  GETLOGIN, GETGID, GETUID
      External   GETLOGIN     !$PRAGMA C ( GETLOGIN )
      External   GETGID       !$PRAGMA C ( GETGID )
      External   GETUID       !$PRAGMA C ( GETUID )
C
      Character  EOS*1
      Parameter  (EOS = '\0')
C
      Character  for*(*)
C
      Character  name(0:100)*1
      Character  sysname*9, nodename*9, nodeext*(65-9), release*9
      Character  version*9, machine*9
      Equivalence  (name( 0), sysname)
      Equivalence  (name( 9), nodename)
      Equivalence  (name(18), nodeext)
      Equivalence  (name(74), release)
      Equivalence  (name(83), version)
      Equivalence  (name(92), machine)
C
      Integer*4  gid, uid
      Integer    i, j, n
      Character  login*80, acct*10
      Pointer    (loginp, login)
C
C
C...  Node name
C
      i = 1
      Call UNAME (name)
      n = INDEX (nodename,EOS) - 1
      Do 1000 j = 1,n
         If (i .le. LEN(for)) Then
            for(i:i) = nodename(j:j)
            i = i + 1
         End If
 1000    Continue
      If (i .le. LEN(for)) Then
         for(i:i) = ':'
         i = i + 1
      End If
C
C...  Login name
C
      loginp = GETLOGIN ()
      n = INDEX (login,EOS) - 1
      Do 1100 j = 1,n
         If (i .le. LEN(for)) Then
            for(i:i) = login(j:j)
            i = i + 1
         End If
 1100    Continue
C
C...  Group ID
C
      gid = GETGID ()
      If (i .le. LEN(for)) Then
         for(i:i) = ' '
         i = i + 1
      End If
      Write (acct,110) gid
  110 Format (I10)
      If (i .le. LEN(for)) Then
         for(i:i) = '('
         i = i + 1
      End If
      Do 1200 j = 1,LEN(acct)
         If ((acct(j:j) .ne. ' ') .and. (i .le. LEN(for))) Then
            for(i:i) = acct(j:j)
            i = i + 1
         End If
 1200    Continue
C
C...  User ID
C
      uid = GETUID ()
      Write (acct,110) uid
      If (i .le. LEN(for)) Then
         for(i:i) = ' '
         i = i + 1
      End If
      Do 1300 j = 1,LEN(acct)
         If ((acct(j:j) .ne. ' ') .and. (i .le. LEN(for))) Then
            for(i:i) = acct(j:j)
            i = i + 1
         End If
 1300    Continue
      If (i .le. LEN(for)) Then
         for(i:) = ')'
      End If
C
      Return
C
      End


