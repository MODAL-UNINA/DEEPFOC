CS    USGS Function ITLEN
C     Version: 1.0
C     Technical Contact: Richard W. Saltus
C     Release: not released
C
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C     Function ITLEN
C
C     Program purpose:
C         This function returns the length of a character string without
C         trailing blanks.
C
C     Instructions for use:
C         Use this function to find out the position of the last non-blank
C         character in a string.
      INTEGER FUNCTION ITLEN(STRING)
C-
C
C     Variables and parameters:
C
C     ireturn = ITLEN (string)
C
C        ireturn = the length of the string without trailing blanks
C         string = string to have trailing blanks removed from
C
C^
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      Character string*(*)
      i=len(string)
   10 Continue
      If (string(i:i).NE.' ')Go To 15
      i=i-1
      If (i.LE.0)Go To 15
      Go To 10
   15 Continue
      itlen=i
      Return
      End
