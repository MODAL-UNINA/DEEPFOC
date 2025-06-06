cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C     UPSHIFT - Converts lowercase to uppercase.
C               All non-lowercase characters unchanged.
C
C     CALL UPSHIFT(A)
C
C             A = A character string of any length
C
C-
      SUBROUTINE UPSHIFT(A)
      CHARACTER*(*) A,UP*(26),DOWN*(26)
      UP='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
      DOWN='abcdefghijklmnopqrstuvwxyz'
      ILEN=LEN(A)
      DO 10 I=1,ILEN
        INUM=INDEX(DOWN,A(I:I))
        IF (INUM.NE.0) A(I:I)=UP(INUM:INUM)
  10      CONTINUE
      RETURN
      END
