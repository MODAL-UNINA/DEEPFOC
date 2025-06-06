cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
CS    USGS Function IGETTOK
C     Version: 1.0
C     Technical Contact: Richard W. Saltus
C     Release: not released
C
C     Function IGETTOK
C
C     Program purpose:
C             - Selects next alpha-numeric token, special symbol,
C               or quoted string from BUF.
C               Puts token in TOK.  Returns ierr & itype to describe
C               token.
C
C     Instructions for use:
C           LEN = IGETTOK(BUF,TOK,IERR,ITYPE)
C
      INTEGER FUNCTION IGETTOK(BUF,TOK,IERR,ITYPE)
C-
C
C     Variables and parameters:
C
C                LEN = Length of BUF read.
C
C                BUF = Character string to search for token
C
C                TOK = Token found (Character string)
C
C               IERR = 1 , got a token
C                      0 , got a token, hit end of BUF
C                     -1 , didn't get a token, hit end of BUF
C
C               ITYPE = 1 , got an alpha-numeric token
C                       0 , got a quoted token
C                      -1 , got a single symbol
C
C         Called by: Style, IxQUEST subroutines, user program
C
C             Calls: ICTYPE
C
C      Commons used: none
C-
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Character*(*) buf,tok
      lbuf=len(buf)
      lmax=len(tok)
      ltok=0
C
      Do 10 i=1,lbuf
      If (ictype(buf(i:i)).LT.0) Then
C
C     got special symbol (non alpha-numeric)
C
         If (ltok.GT.0) Then
C
C       already have a token, this must end it
C
            igettok=i-1
            itype=1
            ierr=1
            Return
         Else
C
C       don't have a token yet
C
            If (buf(i:i).EQ.' ') Then
C
C         got a blank, skip it
C
               Continue
            Else If (buf(i:i).EQ.'''') Then
C
C         got a quote (begin quoted token)
C
               itemp=i
    5          Continue
               iend=index(buf(itemp+1:lbuf),'''')
               If (iend.EQ.0) Then
C
C           didn't find another quote
C
                  tok=buf(i:lbuf)
                  igettok=lbuf
                  itype=0
                  ierr=0
                  Return
               Else
C
C           did find another quote
C
                  nloc=itemp+iend+1
                  If (buf(nloc:nloc).EQ.'''') Then
C
C             paired quotes within quotes, keep looking for end
C
                     itemp=nloc
                     Go To 5
                  Else
C
C           found end of quoted token
C
                     tok=buf(i:itemp+iend)
                     igettok=itemp+iend
                     itype=0
                     If (i+iend.EQ.lbuf) Then
                        ierr=0
                     Else
                        ierr=1
                     End If
                  End If
                  Return
               End If
            Else
C
C       special symbol is token
C
               tok(1:1)=buf(i:i)
               itype=-1
               ierr=1
               igettok=i
               Return
            End If
         End If
      Else
C
C     got alpha-numeric
C
         ltok=ltok+1
         tok(ltok:ltok)=buf(i:i)
      End If
C
   10 Continue
C
C     reached end of buffer, all alpha-numerics
C
      itype=1
      igettok=lbuf
      If (ltok.EQ.0) Then
C
C       all blanks (no token)
C
         ierr=-1
      Else
C
C       got token
C
         ierr=0
      End If
      Return
      End
