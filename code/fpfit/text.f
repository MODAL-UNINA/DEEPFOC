C
C Replacement entry points for the two different uses of CalComp SYMBOL.
C
      Subroutine TEXT (x, y, height, string, angle, nc)
C
C nc is (usually) positive or zero
C
      Character*(*) string, nmark
C
      Call SYMBOL (x,y,height,string,angle,nc)
      Return
C
      Entry MARKER (x, y, height, nmark, angle, nc)
C
C nc is (usually) negative
C
      Call SYMBOL (x,y,height,nmark,angle,nc)
      Return
C
      End
