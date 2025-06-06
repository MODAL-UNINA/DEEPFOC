cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C     INTEGER FUNCTION INEXTVAL
C
C     ATTEMPTS TO DECODE NEXT ATOM INTO A REAL VARIABLE
C
C       iposition = INEXTVAL (BUF,RVAL,IERR)
C
C      iposition = The last position read in BUF + 1
C
C           BUF  = Character string to decode into real numbers
C
C          RVAL  = Real array to hold real numbers obtained
C
C           IERR = 1 - GOT A VALUE
C                  0 - GOT A VALUE, THEN HIT END OF STRING
C                 -1 - DECODE ERROR
C                 -2 - DIDN'T GET A VALUE, HIT END OF STRING
C
C
C-
      Integer Function inextval(buf,rval,ierr)
      Character*(*) buf
      iblen=len(buf)
C
C     LOCATE FIRST NON-BLANK CHARACTER
C
      lpos=1
      Do 10 i=1,iblen
   10 If (buf(i:i).NE.' ')Go To 20
C
C     FALLS OUT HERE IF ALL BLANK - END OF BUFFER
C
      ierr=-2
      inextval=0
      Return
C
C     FOUND A NON-BLANK CHARACTER, CHECK FOR ','
C
   20 Continue
      lpos=i
      If (buf(lpos:lpos).NE.',')Go To 30
C
C     EQUALS ',' - SO RETURN VALUE OF ZERO
C
      rval=0.
      ierr=1
      inextval=lpos+1
      Return
C
C     FIND END OF TOKEN
C
   30 Continue
      Do 40 i=lpos,iblen
   40 If ((buf(i:i).EQ.' ').OR.(buf(i:i).EQ.','))Go To 50
C
C     REACHED END OF STRING
C
      i=i+1
C
C     REACHED END OF TOKEN
C
   50 Continue
      irpos=i-1
C
C     DECODE TOKEN
C
      Read (buf(lpos:irpos),110,err=60)rval
  110 Format (f20.0)
C
C     SUCCESSFUL DECODE - NORMAL EXIT
C
      ierr=1
C
C     SET CORRECT POSITION - PAST NEXT COMMA, OR AT NEXT NON-BLANK
C
      Do 55 j=i,iblen
   55 If (buf(j:j).NE.' ')Go To 57
C
C     HIT END OF STRING
C
      ierr=0
      inextval=0
      Return
C
C     IF POINTING TO COMMA, ADVANCE BY ONE
C
   57 Continue
      inextval=j
      If (buf(j:j).EQ.',')inextval=j+1
      Return
C
C     UNABLE TO DECODE
C
   60 Continue
      ierr=-1
      inextval=0
      Return
      End
