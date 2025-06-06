      CHARACTER*1 FUNCTION UPPER (LOWCHR)
c
c	UPPER converts a lower case character to its upper
c  case equivalent on the VAX
C
c	USE:  char = upper (lowchr)
c
c		upchar = the output upper case character
c		lowchr = the input lower case character
C
      CHARACTER*1  LOWCHR,TEMP
C
C
C...  LOWER CASE A, Z, OFFSET TO UPPER CASE CHARACTER SET
C...  ASCII CHARACTER SET
C
      TEMP = LOWCHR
      IF (LGT(TEMP,'z')) GOTO 1000
      IF (LLT(TEMP,'a')) GOTO 1000
      TEMP = CHAR( ICHAR(LOWCHR) + ICHAR('A') - ICHAR('a') )
 1000 UPPER = TEMP
      RETURN
      END
