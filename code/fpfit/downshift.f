cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C     DOWNSHIFT - Converts uppercase to lowercase.
C               All non-uppercase characters unchanged.
C
C     CALL DOWNSHIFT(A)
C
C             A = A character string of any length
C
C-
      SUBROUTINE DOWNSHIFT(A)
      CHARACTER*(*) A,UP*(26),DOWN*(26)
      UP='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
      DOWN='abcdefghijklmnopqrstuvwxyz'
      ILEN=LEN(A)
      DO 10 I=1,ILEN
        INUM=INDEX(UP,A(I:I))
        IF (INUM.NE.0) A(I:I)=DOWN(INUM:INUM)
  10      CONTINUE
      RETURN
      END
